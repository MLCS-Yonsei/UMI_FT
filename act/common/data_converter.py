from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from act.common.replay_buffer import ReplayBuffer
import os
import h5py

class ACTDataConverter:
    def __init__(self,
                 shape_meta: dict,
                 replay_buffer: ReplayBuffer,
                 hdf5_path: str,
                 camera_names: list,
                 rgb_keys: list,
                 lowdim_keys: list,
                 key_horizon: dict, # action horizon
                 key_latency_steps: dict,
                 key_down_sample_steps: dict,
                 ):
        
        # load all length of episodes
        episodes_ends = replay_buffer.episode_ends[:]
        # format
        # [1st_ep_end_time 2nd_ep_end_time ...]
        # 1st_ep_end_time = 2nd_ep_satrt_time

        # define max episode length
        max_ep_length = 0

        # laod gripper width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]

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
        
        self.replay_buffer = dict()
        self.num_robot = 0

        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
            
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
        
        shape_dict = dict()
        for key, attr in shape_meta['obs']:
            shape = attr.get('shape')
            if key.endswith('rgb'):
                shape_dict[key] = shape
            elif key.endswith('pos'):
                shape_dict[key] = shape
            elif key.endswith('angle'):
                shape_dict[key] = shape
            elif key.endswith('width'):
                shape_dict[key] = shape
            # elif key.endswith('force'):
            #     shape_dict[key] = shape
            # elif key.endswith('torque'):
            #     shape_dict[key] = shape
            else:
                raise NotImplementedError
        
        for key, attr in shape_meta['action']:
            shape = attr.get('shape')
            shape_dict['action'] = shape
            
        
        self.indices = indices
        self.ep_indices = ep_indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        self.shape_dict = shape_dict

        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.key_horizon = key_horizon

        self.max_ep_length = max_ep_length
        self.camera_names = camera_names
        self.hdf5_path = hdf5_path

    def __len__(self):
        # length of all data
        return len(self.indices)

    def process_episodes(self):
        '''
            Convert zarr data as HDF5
            TODO
            1. make action padding with self.max_ep_length
            2. make action sequence which requre config yaml

            For each timestep:
            observations
            - images
                - left : (3, 224, 224)
                (-right)
            - eef pos : 3
            - eef rot : 6
            - gripper width : 1
            - force : 3
            - torque: 3

            action
            - eef pos + eef rot axis angle + gripper width : 10

            Processing

            observations
            - images
                - left : (max_ep_length, 3, 224, 224)
                (-right)
            - eef pos : (max_ep_length, 3)
            - eef rot : (max_ep_length, 6)
            - gripper width : (max_ep_length, 1)
            - force : (max_ep_length, 3)
            - torque: (max_ep_length, 3)

            action
            - eef pos + eef rot axis angle + gripper width : (max_ep_length, 10)
        
        '''

        # Assume uni-manual
        data_dict = {
            '/observations/eef_pos': [],
            '/observations/eef_rot': [],
            '/observations/gripper_width': [],
            # 'observations/force': [],
            # 'observations/torque': [],
            '/action': []
        }

        for cam_name in self.camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
        

        # observation keys
        obs_keys = self.rgb_keys + self.lowdim_keys
        

        for ep in self.ep_indices:
            # current data
            ep_idx, start_idx, end_idx = ep

            # observation
            for key in obs_keys:
                this_latency_steps = self.key_latency_steps[key]
                this_downsample_steps = self.key_down_sample_steps[key]
                this_horizon = self.key_horizon[key]

                # ep_idx th data array
                obs_arr = self.replay_buffer[key][start_idx : end_idx]

                # save to data dict
                if key in self.rgb_keys:
                    for cam_name in self.camera_names:
                        data_dict[f'observations/images/{cam_name}'].append(obs_arr)
                elif key in self.lowdim_keys:
                    # TODO : process low dim data with latency

                    if key.endswith('pos'):
                        data_dict['/observations/eef_pos'].append(obs_arr)
                    elif key.endwith('axis_angle'):
                        data_dict['/observations/eef_rot'].append(obs_arr)
                    elif key.endswith('width'):
                        data_dict['/observations/gripper_width'].append(obs_arr)
                    # elif key.endswith('force'):
                    #     data_dict['/observations/force'].append(obs_arr)
                    # elif key.endswith('torque'):
                    #     data_dict['/observations/torque'].append(obs_arr)
                    else:
                        raise NotImplementedError
                    
            # action
            action_arr = self.replay_buffer['action']
            data_dict['/action'].append(action_arr)

            # shape of data
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
                # elif key.endswith('force'):
                #     force_shape = shape
                # elif key.endswith('torque'):
                #     torque_shape = shape
                elif key.endswith('action'):
                    action_shape = shape
                else:
                    raise NotImplementedError
        
            # Convert each ep to HDF5
            if os.path.isfile(self.hdf5_path_path + 'ep_' + str(ep_idx) +'.hdf5'):
                print(f'Dataset already exist.')

            with h5py.File(self.hdf5_path_path + 'ep_' + str(ep_idx) +'.hdf5', 'w' , rdcc_nbytes=1024**2*2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in self.camera_names:
                    _ = image.create_dataset(cam_name, (self.max_ep_length, image_shape[0], image_shape[1], image_shape[-1]), dtype='uint8',
                                             chunks=(1, image_shape[0], image_shape[1], image_shape[-1]), )

                _ = obs.create_dataset('eef_pos', (self.max_ep_length, pos_shape))
                _ = obs.create_dataset('eef_rot', (self.max_ep_length, rot_shape))
                _ = obs.create_dataset('gripper_width', (self.max_ep_length, width_shape))
                _ = root.create_dataset('action', (self.max_ep_length, action_shape))

                for name, array in data_dict.items():
                    root[name][...] = array

        



