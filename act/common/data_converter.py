from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from act.common.replay_buffer import ReplayBuffer
import os
import h5py
import torch

from act.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from act.common.pose_repr_util import convert_pose_mat_rep
from act.common.pytorch_util import dict_apply
from act.common.pose_util import pose_to_mat, mat_to_pose10d

from ur_ikfast.ur_ikfast import ur_kinematics

class ACTDataConverter:
    def __init__(self,
                 shape_meta: dict,
                 replay_buffer: ReplayBuffer,
                 hdf5_path: str,
                 camera_names: list,
                 rgb_keys: list,
                 depth_keys: list,
                 lowdim_keys: list,
                 key_horizon: dict, # action horizon
                 key_latency_steps: dict,
                 key_down_sample_steps: dict,
                 pose_repr: dict,
                 is_depth: bool
                 ):
        
        
        # load all length of episodes
        episodes_ends = replay_buffer.episode_ends[:]
        # format
        # [1st_ep_end_time 2nd_ep_end_time ...]
        # 1st_ep_end_time = 2nd_ep_satrt_time

        # define max episode length
        max_ep_length = 0

        # laod gripper width
        # gripper_width = replay_buffer['robot0_gripper_width'][:, 0]

        # create indics : current_idx, start_idx, end_idx
        indices = list()
        ep_indices = list()
        for i in range(len(episodes_ends)):
            # define start and end idx
            start_idx = 0 if i == 0 else episodes_ends[i-1]
            end_idx = episodes_ends[i]

            # find max episode length
            ep_length = end_idx - start_idx
            max_ep_length = max(ep_length, max_ep_length)

            ep_indices.append((i, start_idx, end_idx))


            for current_idx in range(start_idx, end_idx):
                indices.append((current_idx, start_idx, end_idx))
        
        print("Max ep length: ", max_ep_length)

        self.indices = indices
        self.ep_indices = ep_indices
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.shape_meta = shape_meta

        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.key_horizon = key_horizon

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr['obs_pose_repr']
        self.action_pose_repr = self.pose_repr['action_pose_repr']

        self.is_depth = is_depth

        self.sampler_lowdim_keys = list()
        for key in self.lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
        # print("Converter low dim keys : ", self.lowdim_keys) 
        # ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot0_eef_rot_axis_angle_wrt_start', 'robot0_force', 'robot0_torque']
        # print("Converter sampler low dim keys : ", self.sampler_lowdim_keys)
        # ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot0_force', 'robot0_torque']

        print("Extracting replay buffer")
        self.extract_replay_buffer(replay_buffer)

        
        shape_dict = dict()
        for key, attr in shape_meta['obs'].items():
            shape = attr.get('shape')
            if key.endswith('rgb'):
                shape_dict[key] = shape
            elif key.endswith('depth'):
                shape_dict[key] = shape
            elif key.endswith('pos'):
                shape_dict[key] = shape
            elif key.endswith('angle'):
                shape_dict[key] = shape
            elif key.endswith('width'):
                shape_dict[key] = shape
            elif key.endswith('start'):
                shape_dict[key] = shape
            elif key.endswith('force'):
                shape_dict[key] = shape
            elif key.endswith('torque'):
                shape_dict[key] = shape
            else:
                raise NotImplementedError
        
        shape_dict['action'] = shape_meta['action']['shape']
        
        self.shape_dict = shape_dict

        self.max_ep_length = max_ep_length
        self.camera_names = camera_names
        self.hdf5_path = hdf5_path

        ur3_arm = ur_kinematics.URKinematics('ur3')
        self.ur3_arm = ur3_arm

        self.num_ep_batch = 2
        self.ep_batch_indices = list()
        self.ep_batch_size = len(self.ep_indices) // 2
        for ep_batch_idx in range(self.num_ep_batch):
            self.ep_batch_indices.append( self.ep_indices[ ep_batch_idx * self.ep_batch_size : (ep_batch_idx+1) * self.ep_batch_size ] )

    def __len__(self):
        # length of all data
        return len(self.indices)

    def extract_replay_buffer(self, replay_buffer):
        '''
            camera0_rgb
            robot0_demo_end_pose
            robot0_demo_start_pose
            robot0_eef_pos
            robot0_eef_rot_axis_angle
            robot0_force
            robot0_gripper_width
            robot0_torque
        '''
        
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                self.key_horizon[key] = self.shape_meta['obs'][query_key]['horizon']
                self.key_latency_steps[key] = self.shape_meta['obs'][query_key]['latency_steps']
                self.key_down_sample_steps[key] = self.shape_meta['obs'][query_key]['down_sample_steps']
                
        self.replay_buffer = dict()
        self.num_robot = 0

        for key in self.sampler_lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
            
            self.replay_buffer[key] = replay_buffer[key][:]
        
        for key in self.rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        if self.is_depth:
            for key in self.depth_keys:
                self.replay_buffer[key] = replay_buffer[key]
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)
        
        # print("Extracted replay buffer keys: ", self.replay_buffer.keys())
        # dict_keys(['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot0_force', 'robot0_torque', 'robot0_demo_end_pose', 'robot0_demo_start_pose', 'camera0_rgb', 'action'])


    def preprocess_episodes(self, ep_batch_idx = 0):
        '''
            process list
            - latency
            - downsample
            - interpolation
            - repeat frame before first grasp
            - image processing
            - relative pose
            - action processing

            Original code use slicing with certain idx
            for example
            input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            slice_start = 2
            current_idx = 9
            this_downsample_steps = 2
            output = input_arr[slice_start: current_idx + 1: this_downsample_steps] #[start : stop : step]
            => [2 4 6 8]

            However ACT will use whole episode
            
            => Calculate interpolated idx and slice the data array

        '''
        all_results = []
        if self.is_depth:
            obs_keys = self.rgb_keys + self.depth_keys + self.sampler_lowdim_keys
        else:
            obs_keys = self.rgb_keys + self.sampler_lowdim_keys
        # if self.ignore_rgb_is_applied:
        #     obs_keys = self.sampler_lowdim_keys
        
        # for (ep_idx, start_idx, end_idx) in self.ep_indices:
        for (ep_idx, start_idx, end_idx) in self.ep_batch_indices[ep_batch_idx]:
            print(f"Pre-Processing episode {ep_idx}/{len(self.ep_indices)}.")
            episode_result = dict()

            # observation
            for key in obs_keys:
                this_horizon = self.key_horizon[key]
                this_latency_steps = self.key_latency_steps[key]
                this_downsample_steps = self.key_down_sample_steps[key]

                # ep_idx th data array
                obs_arr = self.replay_buffer[key]

                # calculate idx for image
                if key in self.rgb_keys:
                    image_idx_set = self.calculate_idx(key, start_idx, end_idx, this_horizon, this_downsample_steps, this_latency_steps)
                    image_idx_set = sorted(list(image_idx_set))
                    image_start_idx, image_end_idx = image_idx_set[0], image_idx_set[-1]
                    
                    # save image data
                    episode_result[key] = obs_arr[image_start_idx:image_end_idx]

                elif key in self.depth_keys:
                    image_idx_set = self.calculate_idx(key, start_idx, end_idx, this_horizon, this_downsample_steps, this_latency_steps)
                    image_idx_set = sorted(list(image_idx_set))
                    image_start_idx, image_end_idx = image_idx_set[0], image_idx_set[-1]
                    
                    # save image data
                    episode_result[key] = obs_arr[image_start_idx:image_end_idx]

                # calcuate idx for low dim data
                else:
                    interpolated_obs_arr = self.calculate_idx(key, start_idx, end_idx, this_horizon, this_downsample_steps, this_latency_steps, obs_arr)
                    
                    # save low dim data
                    episode_result[key] = interpolated_obs_arr
            
            # action
            action_arr = self.replay_buffer['action']
            action_horizon = self.key_horizon['action']
            action_latency_steps = self.key_latency_steps['action']
            assert action_latency_steps == 0
            action_downsample_steps = self.key_down_sample_steps['action']
            action_idx_set = self.calculate_idx(key, start_idx, end_idx, action_horizon, action_downsample_steps, action_latency_steps, type='action')
            action_idx_set = sorted(list(action_idx_set))
            action_start_idx, action_end_idx = action_idx_set[0], action_idx_set[-1]
            episode_result['action'] = action_arr[action_start_idx:action_end_idx]

            all_results.append(episode_result)

        return all_results

    def postprocess_episodes(self, data):
        print("Post-Processing episode.")
        # data = self.preprocess_episodes()

        obs_dict = dict()

        for key in self.rgb_keys:
            if not key in data:
                continue

            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]

        if self.is_depth:
            for key in self.depth_keys:
                if not key in data:
                    continue
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
                # T,C,H,W
                del data[key]

        for key in self.sampler_lowdim_keys: 
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # print("postprocess obs dict keys: ", obs_dict.keys())
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.sampler_lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
        
        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{other_robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]
        
        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6]) # some data has no action at all
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)
        
        # torch_data = {
        #     'obs': dict_apply(obs_dict, torch.from_numpy),
        #     'action': torch.from_numpy(data['action'].astype(np.float32))
        # }
        # return torch_data
        return {'obs' : obs_dict, 'action': data['action']}

    def convert_episodes(self):
        # define shape of data
        for key in self.shape_dict:
            shape = self.shape_dict[key]
            if key.endswith('rgb'):
                image_shape = shape
            elif key.endswith('depth'):
                depth_shape = shape
            elif key.endswith('pos'):
                pos_shape = shape
            elif key.endswith('angle'):
                rot_shape = shape
            elif key.endswith('width'):
                width_shape = shape
            elif key.endswith('start'):
                start_shape = shape
            elif key.endswith('force'):
                force_shape = shape
            elif key.endswith('torque'):
                torque_shape = shape
            elif key.endswith('action'):
                action_shape = shape
            else:
                raise NotImplementedError
        
        # processing data
        print("Preprocessing...")
        for ep_batch_idx in range(self.num_ep_batch):
            all_preprocessed_data = self.preprocess_episodes(ep_batch_idx=ep_batch_idx)

            for ep_idx, episode_data in enumerate(all_preprocessed_data):
                print(f"Postprocessing episode {ep_idx}/{len(self.ep_indices)}.")
                postprocessed_data = self.postprocess_episodes(episode_data)

                print(f"Converting episode {ep_idx}/{len(self.ep_indices)}.")
                # define hdf5 save path
                dataset_path = os.path.join(self.hdf5_path, f'ep_{ep_idx + ep_batch_idx * self.ep_batch_size}.hdf5')

                # skip saving if file alreay exists
                if os.path.isfile(dataset_path):
                    print(f"Episode {ep_idx} already exists. Skipping...")
                    continue
                
                # define data dict
                data_dict = {
                    '/observations/eef_pos': [],
                    '/observations/eef_rot': [],
                    '/observations/eef_rot_start': [],
                    '/observations/gripper_width': [],
                    '/observations/force': [],
                    '/observations/torque': [],
                    '/action': []
                }

                for cam_name in self.camera_names:
                    data_dict[f'/observations/images/{cam_name}'] = []
                    data_dict[f'/observations/depth_images/{cam_name}'] = []

                # unpack data and save to data dict
                for k in postprocessed_data['obs'].keys():
                    if k.endswith('pos'):
                        data_dict['/observations/eef_pos'].append(postprocessed_data['obs'][k])
                    elif k.endswith('angle'):
                        data_dict['/observations/eef_rot'].append(postprocessed_data['obs'][k])
                    elif k.endswith('width'):
                        data_dict['/observations/gripper_width'].append(postprocessed_data['obs'][k])
                    elif k.endswith('force'):
                        data_dict['/observations/force'].append(postprocessed_data['obs'][k])
                    elif k.endswith('torque'):
                        data_dict['/observations/torque'].append(postprocessed_data['obs'][k])
                    elif k.endswith('start'):
                        data_dict['/observations/eef_rot_start'].append(postprocessed_data['obs'][k])
                    elif k.endswith('rgb'):
                        for cam_name in self.camera_names:
                            # data_dict[f'/observations/images/{cam_name}'].append(postprocessed_data['obs'][k])
                            data_dict[f'/observations/images/{cam_name}'].append((postprocessed_data['obs'][k]*255).astype(np.uint8))
                    elif k.endswith('depth'):
                        for cam_name in self.camera_names:
                            data_dict[f'/observations/depth_images/{cam_name}'].append((postprocessed_data['obs'][k]*255).astype(np.uint8))
                    else:
                        raise NotImplementedError

                data_dict['/action'].append(postprocessed_data['action'])

                # pad data
                for key, array in data_dict.items():
                    array = array[0]
                    current_length = array.shape[0]

                    if current_length < self.max_ep_length:
                        pad_size = self.max_ep_length - current_length
                        padding_shape = list(array.shape[1:])
                        padding = np.zeros((pad_size, *padding_shape), dtype=array.dtype)
                        array = np.concatenate([array, padding], axis=0)

                    data_dict[key] = [array]

                # convert each ep to hdf5 and save
                with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
                    root.attrs['sim'] = False

                    # make group
                    obs_group = root.create_group('observations')
                    image_group = obs_group.create_group('images')
                    depth_group = obs_group.create_group('depth_images')

                    # create dataset case
                    for cam_name in self.camera_names:
                        _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='uint8',
                                                 chunks=(1, *image_shape), )
                        # _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='float32',
                        #                          chunks=(1, *image_shape), )
                        _ = depth_group.create_dataset(cam_name, (self.max_ep_length, *depth_shape), dtype='uint8',
                                                 chunks=(1, *depth_shape), )

                    _ = obs_group.create_dataset('eef_pos', (self.max_ep_length, *pos_shape))
                    _ = obs_group.create_dataset('eef_rot', (self.max_ep_length, *rot_shape))
                    _ = obs_group.create_dataset('force', (self.max_ep_length, *force_shape))
                    _ = obs_group.create_dataset('torque', (self.max_ep_length, *torque_shape))
                    _ = obs_group.create_dataset('eef_rot_start', (self.max_ep_length, *start_shape))
                    _ = obs_group.create_dataset('gripper_width', (self.max_ep_length, *width_shape))
                    _ = root.create_dataset('action', (self.max_ep_length, *action_shape))

                    for name, array in data_dict.items():
                        root[name][...] = array[0]

        # all_preprocessed_data = self.preprocess_episodes()
        # for ep_idx, episode_data in enumerate(all_preprocessed_data):
        #     print(f"Postprocessing episode {ep_idx}/{len(self.ep_indices)}.")
        #     postprocessed_data = self.postprocess_episodes(episode_data)

        #     print(f"Converting episode {ep_idx}/{len(self.ep_indices)}.")
        #     # define hdf5 save path
        #     dataset_path = os.path.join(self.hdf5_path, f'ep_{ep_idx}.hdf5')

        #     # skip saving if file alreay exists
        #     if os.path.isfile(dataset_path):
        #         print(f"Episode {ep_idx} already exists. Skipping...")
        #         continue
            
        #     # define data dict
        #     data_dict = {
        #         '/observations/eef_pos': [],
        #         '/observations/eef_rot': [],
        #         '/observations/eef_rot_start': [],
        #         '/observations/gripper_width': [],
        #         '/observations/force': [],
        #         '/observations/torque': [],
        #         '/action': []
        #     }

        #     for cam_name in self.camera_names:
        #         data_dict[f'/observations/images/{cam_name}'] = []

        #     # unpack data and save to data dict
        #     for k in postprocessed_data['obs'].keys():
        #         if k.endswith('pos'):
        #             data_dict['/observations/eef_pos'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('angle'):
        #             data_dict['/observations/eef_rot'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('width'):
        #             data_dict['/observations/gripper_width'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('force'):
        #             data_dict['/observations/force'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('torque'):
        #             data_dict['/observations/torque'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('start'):
        #             data_dict['/observations/eef_rot_start'].append(postprocessed_data['obs'][k])
        #         elif k.endswith('rgb'):
        #             for cam_name in self.camera_names:
        #                 # data_dict[f'/observations/images/{cam_name}'].append(postprocessed_data['obs'][k])
        #                 data_dict[f'/observations/images/{cam_name}'].append((postprocessed_data['obs'][k]*255).astype(np.uint8))
        #         else:
        #             raise NotImplementedError
            
        #     data_dict['/action'].append(postprocessed_data['action'])

        #     # pad data
        #     for key, array in data_dict.items():
        #         array = array[0]
        #         current_length = array.shape[0]

        #         if current_length < self.max_ep_length:
        #             pad_size = self.max_ep_length - current_length
        #             padding_shape = list(array.shape[1:])
        #             padding = np.zeros((pad_size, *padding_shape), dtype=array.dtype)
        #             array = np.concatenate([array, padding], axis=0)
                
        #         data_dict[key] = [array]
            
        #     # convert each ep to hdf5 and save
        #     with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
        #         root.attrs['sim'] = False

        #         # make group
        #         obs_group = root.create_group('observations')
        #         image_group = obs_group.create_group('images')

        #         # create dataset case
        #         for cam_name in self.camera_names:
        #             _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='uint8',
        #                                      chunks=(1, *image_shape), )
        #             # _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='float32',
        #             #                          chunks=(1, *image_shape), )

        #         _ = obs_group.create_dataset('eef_pos', (self.max_ep_length, *pos_shape))
        #         _ = obs_group.create_dataset('eef_rot', (self.max_ep_length, *rot_shape))
        #         _ = obs_group.create_dataset('force', (self.max_ep_length, *force_shape))
        #         _ = obs_group.create_dataset('torque', (self.max_ep_length, *torque_shape))
        #         _ = obs_group.create_dataset('eef_rot_start', (self.max_ep_length, *start_shape))
        #         _ = obs_group.create_dataset('gripper_width', (self.max_ep_length, *width_shape))
        #         _ = root.create_dataset('action', (self.max_ep_length, *action_shape))

        #         for name, array in data_dict.items():
        #             root[name][...] = array[0]
    
    def ur_ik(self, tx_tag_tcp):
        tx_tag_base = np.array([
        [
          0.014500569635805594,
          -0.9997530090623589,
          -0.016842041176650398,
          -0.044059934138102635
        ],
        [
          0.9996692201244223,
          0.014853059193052465,
          -0.02099611793729384,
          -0.4799630648844079
        ],
        [
          0.02124108792096622,
          -0.016532014498134257,
          0.9996376887055468,
          -0.02314005620630681
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]

        ])

        tx_base_tag = np.linalg.inv(tx_tag_base)

        tx_base_tcp = tx_base_tag @ tx_tag_tcp

        joint_trajectory = np.zeros((tx_tag_tcp.shape[0], 6))

        for t, pose in enumerate(tx_base_tcp):
            curr_pose = pose[:3, :4]
            if t != 0:
                result = self.ur3_arm.inverse(curr_pose, False, q_guess=joint_trajectory[t-1, :])
            else:
                result = self.ur3_arm.inverse(curr_pose, False)
            
            joint_trajectory[t, :] = result

        return joint_trajectory

    def postprocess_episodes_joints(self, data):
        '''
            data : dict
            key : rgb, low dim, action

        '''
        obs_dict = dict()

        for key in self.rgb_keys:
            if not key in data:
                continue

            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]

        for key in self.sampler_lowdim_keys: 
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # convert pose to joint
        for robot_id in range(self.num_robot):
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            obs_joints = self.ur_ik(pose_mat)

            obs_dict[f'robot{robot_id}_joints'] = obs_joints

        # convert action pose to joint
        actions = list()
        for robot_id in range(self.num_robot):
            # action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6]) # some data has no action at all
            # action_joints = self.ur_ik(action_mat)
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            obs_joints = self.ur_ik(pose_mat)
            action_joints = obs_joints[:-1, :]
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_joints, action_gripper], axis=-1))
        
        data['action'] = np.concatenate(actions, axis=-1)

        return {'obs' : obs_dict, 'action': data['action']}

    def convert_episodes_joints(self):
        # define shape of data
        for key in self.shape_dict:
            shape = self.shape_dict[key]
            if key.endswith('rgb'):
                image_shape = shape
            elif key.endswith('pos'):
                pos_shape = shape
            elif key.endswith('angle'):
                rot_shape = shape
            elif key.endswith('width'):
                width_shape = shape
            elif key.endswith('start'):
                start_shape = shape
            elif key.endswith('force'):
                force_shape = shape
            elif key.endswith('torque'):
                torque_shape = shape
            elif key.endswith('action'):
                action_shape = shape
            else:
                raise NotImplementedError
            
        # processing data
        print("Preprocessing...")
        # all_preprocessed_data = self.preprocess_episodes()
        # for ep_idx, episode_data in enumerate(all_preprocessed_data):
            # print(f"Postprocessing episode {ep_idx}/{len(self.ep_indices)} with joints.")
            # postprocessed_data = self.postprocess_episodes_joints(episode_data)
            # print(f"Converting episode {ep_idx}/{len(self.ep_indices)}.")
            # define hdf5 save path
            # dataset_path = os.path.join(self.hdf5_path, f'j_ep_{ep_idx}.hdf5')

        for ep_batch_idx in range(self.num_ep_batch):
            all_preprocessed_data = self.preprocess_episodes(ep_batch_idx=ep_batch_idx)

            for ep_idx, episode_data in enumerate(all_preprocessed_data):
                print(f"Postprocessing episode {ep_idx}/{len(self.ep_indices)}.")
                postprocessed_data = self.postprocess_episodes_joints(episode_data)

                print(f"Converting episode {ep_idx}/{len(self.ep_indices)}.")
                dataset_path = os.path.join(self.hdf5_path, f'j_ep_{ep_idx + ep_batch_idx * self.ep_batch_size}.hdf5')
        
                # check joint value none
                action_joint = postprocessed_data['action']
                if np.any(np.isnan(action_joint)):
                    print("action joint contains NaN")
                    continue

                # skip saving if file alreay exists
                if os.path.isfile(dataset_path):
                    print(f"Episode {ep_idx} already exists. Skipping...")
                    continue
                
                # define data dict
                data_dict = {
                    '/observations/qpos': [],
                    '/observations/gripper_width': [],
                    '/observations/force': [],
                    '/observations/torque': [],
                    '/action': [] # qpos with gripepr
                }

                for cam_name in self.camera_names:
                    data_dict[f'/observations/images/{cam_name}'] = []

                # unpack data and save to data dict
                for k in postprocessed_data['obs'].keys():
                    if k.endswith('joints'):
                        data_dict['/observations/qpos'].append(postprocessed_data['obs'][k])
                    elif k.endswith('width'):
                        data_dict['/observations/gripper_width'].append(postprocessed_data['obs'][k])
                    elif k.endswith('force'):
                        data_dict['/observations/force'].append(postprocessed_data['obs'][k])
                    elif k.endswith('torque'):
                        data_dict['/observations/torque'].append(postprocessed_data['obs'][k])
                    elif k.endswith('rgb'):
                        for cam_name in self.camera_names:
                            # data_dict[f'/observations/images/{cam_name}'].append(postprocessed_data['obs'][k])
                            data_dict[f'/observations/images/{cam_name}'].append((postprocessed_data['obs'][k]*255).astype(np.uint8))
                    else:
                        pass
                    
                data_dict['/action'].append(postprocessed_data['action'])

                # pad data
                for key, array in data_dict.items():
                    array = array[0]
                    current_length = array.shape[0]

                    if current_length < self.max_ep_length:
                        pad_size = self.max_ep_length - current_length
                        padding_shape = list(array.shape[1:])
                        padding = np.zeros((pad_size, *padding_shape), dtype=array.dtype)
                        array = np.concatenate([array, padding], axis=0)

                    data_dict[key] = [array]

                # convert each ep to hdf5 and save
                with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
                    root.attrs['sim'] = False

                    # make group
                    obs_group = root.create_group('observations')
                    image_group = obs_group.create_group('images')

                    # create dataset case
                    for cam_name in self.camera_names:
                        _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='uint8',
                                                 chunks=(1, *image_shape), )
                        # _ = image_group.create_dataset(cam_name, (self.max_ep_length, *image_shape), dtype='float32',
                        #                          chunks=(1, *image_shape), )

                    _ = obs_group.create_dataset('qpos', (self.max_ep_length, 6)) # joint 6
                    _ = obs_group.create_dataset('force', (self.max_ep_length, *force_shape))
                    _ = obs_group.create_dataset('torque', (self.max_ep_length, *torque_shape))
                    _ = obs_group.create_dataset('gripper_width', (self.max_ep_length, *width_shape))
                    _ = root.create_dataset('action', (self.max_ep_length, 7)) # 7 : joint 6 + gripper width 1

                    for name, array in data_dict.items():
                        root[name][...] = array[0]

        
        

    def calculate_idx(self, key, start_idx, end_idx, horizon, downsample_steps, latency_steps, obs_arr = None, type = 'obs',):
        index_set = set()
        if type == 'action':
            for current_idx in range(start_idx, end_idx):
                slice_end = min(end_idx, current_idx + (horizon - 1) * downsample_steps + 1)
                downsampled_indices = list(range(current_idx, slice_end, downsample_steps))
                
                for idx in downsampled_indices:
                    index_set.add(int(idx))
            return index_set
        else:
            if key in self.rgb_keys:
                for current_idx in range(start_idx, end_idx):
                    assert latency_steps == 0
                    num_valid = min(horizon, (current_idx - start_idx) // downsample_steps + 1)
                    slice_start = current_idx - (num_valid - 1) * downsample_steps
                    downsampled_indices = list(range(slice_start, current_idx + 1, downsample_steps))

                    for idx in downsampled_indices:
                        index_set.add(int(idx))

                return index_set

            elif key in self.depth_keys:
                for current_idx in range(start_idx, end_idx):
                    assert latency_steps == 0
                    num_valid = min(horizon, (current_idx - start_idx) // downsample_steps + 1)
                    slice_start = current_idx - (num_valid - 1) * downsample_steps
                    downsampled_indices = list(range(slice_start, current_idx + 1, downsample_steps))

                    for idx in downsampled_indices:
                        index_set.add(int(idx))

                return index_set

            else: # low dim 
                output_dict = dict()
                for current_idx in range(start_idx, end_idx):
                    # latency idx from current idx
                    idx_with_latency = np.array(
                            [current_idx - idx * downsample_steps + latency_steps for idx in range(horizon)],
                            dtype=np.float32)
                    # reverse
                    idx_with_latency = idx_with_latency[::-1]
                    # clip
                    idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                    # interpolation
                    interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                    interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                    if 'rot' in key:
                        # Rotation handling
                        rot_preprocess, rot_postprocess = None, None
                        if key.endswith('quat'):
                            rot_preprocess = st.Rotation.from_quat
                            rot_postprocess = st.Rotation.as_quat
                        elif key.endswith('axis_angle'):
                            rot_preprocess = st.Rotation.from_rotvec
                            rot_postprocess = st.Rotation.as_rotvec
                        else:
                            raise NotImplementedError

                        slerp = st.Slerp(
                            times=np.arange(interpolation_start, interpolation_end),
                            rotations=rot_preprocess(obs_arr[interpolation_start:interpolation_end])
                        )
                        interpolated_values = rot_postprocess(slerp(idx_with_latency))
                    else:
                        # Linear interpolation
                        assert len(np.arange(interpolation_start, interpolation_end)) == len(obs_arr[interpolation_start:interpolation_end])
                        interp = si.interp1d(
                            x=np.arange(interpolation_start, interpolation_end),
                            y=obs_arr[interpolation_start:interpolation_end],
                            axis=0,
                            assume_sorted=True
                        )
                        interpolated_values = interp(idx_with_latency)

                    # Save unique data with unique idx
                    for i, idx in enumerate(idx_with_latency):
                        if idx not in index_set:
                            index_set.add(idx)
                            output_dict[idx] = interpolated_values[i]


                sorted_indices = sorted(output_dict.keys())
                # Save unique value as array
                interpolated_output = np.array([output_dict[idx] for idx in sorted_indices])

                return interpolated_output
        