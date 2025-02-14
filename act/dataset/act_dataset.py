import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
from act.codecs.imagecodecs_numcodecs import register_codecs
from act.dataset.base_dataset import BaseDataset
from act.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from act.common.pose_repr_util import convert_pose_mat_rep
from act.common.pytorch_util import dict_apply
from act.common.replay_buffer import ReplayBuffer
from act.common.data_converter import ACTDataConverter
from act.dataset.base_dataset import BaseDataset
from act.common.normalizer import LinearNormalizer
from act.common.pose_util import pose_to_mat, mat_to_pose10d

from act.common.sampler import ACTSampler
import h5py

register_codecs()


class ACTDataset(BaseDataset):
    def __init__(self,
        episode_indices,   # a list of episode IDs, e.g. [0, 1, 2, …]
        camera_names: list, # list of camera names (to be passed to converter & sampler)
        shape_meta: dict,
        dataset_path: str,  # path to the original (zarr) dataset
        hdf5_path: str,     # directory to write/read HDF5 files
        chunk_size: int,
        cache_dir: str = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: float = None,
        ):

        self.dataset_path = dataset_path
        self.episode_indices = episode_indices
        self.hdf5_path = hdf5_path
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.pose_repr = pose_repr
        self.temporally_independent_normalization = temporally_independent_normalization
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_duration = max_duration
        self.shape_meta = shape_meta
        
        self.num_robot = 0

        # Load replay buffer
        print("Loading zarr")
        replay_buffer = self.load_zarr()
        self.replay_buffer = replay_buffer

        # Solve key and attribute
        print("Solving key and attributes")
        rgb_keys, lowdim_keys, key_horizon, key_down_sample_steps, key_latency_steps = self.solve_key_attr(shape_meta=shape_meta)

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps

        
        # Define sampler
        sampler = ACTSampler(
            episode_indices=self.episode_indices,
            hdf5_path=hdf5_path,
            cam_names=camera_names,
        )

        self.sampler = sampler


    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sampler.sample_item(idx)

    def convert_zarr_to_hdf5(self):
        converter = ACTDataConverter(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            hdf5_path=self.hdf5_path,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            camera_names=self.camera_names,
            pose_repr = self.pose_repr
        )

        # Convert UMI zarr data to hdf5
        print("Converting data...")
        converter.convert_episodes()
        return converter

    def solve_key_attr(self, shape_meta):
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key) # SC: F/T is appended to lowdim_keys

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']
        return rgb_keys, lowdim_keys, key_horizon, key_down_sample_steps, key_latency_steps

    def load_zarr(self, cache_dir=None):
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(self.dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(self.dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(self.dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(self.dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )

        return replay_buffer

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        
        for episode_id in self.episode_indices:
            dataset_path = os.path.join(self.hdf5_path, f"ep_{episode_id}.hdf5")
            with h5py.File(dataset_path, 'r') as root:
                for key in self.lowdim_keys:
                    if key.endswith('pos'):
                        arr = root['/observations/eef_pos'][()]
                    elif key.endswith('angle'):
                        arr = root['/observations/eef_rot'][()]
                    elif key.endswith('width'):
                        arr = root[f'/observations/gripper_width'][()]
                    elif key.endswith('start'):
                        arr = root['/observations/eef_rot_start'][()]
                    elif key.endswith('force'):
                        arr = root['/observations/force'][()]
                    elif key.endswith('torque'):
                        arr = root['/observations/torque'][()]
                    else:
                        pass
                    data_cache[key].append(arr)
                
                action_arr = root[f'/action'][()]
                data_cache['action'].append(action_arr)
        
        for key in data_cache:
            data_cache[key] = np.stack(data_cache[key], axis=0)
            B, T = data_cache[key].shape[0], data_cache[key].shape[1]
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B * T, -1)
            

        action_data = data_cache['action']
        action_dim = action_data.shape[-1]
        if action_dim % self.num_robot != 0:
            raise ValueError("Action dimension is not divisible by num_robot.")
        dim_a = action_dim // self.num_robot

        action_normalizers = []
        for i in range(self.num_robot):
            # Assume the first 3 dims correspond to position.
            pos_stats = array_to_stats(action_data[..., i * dim_a : i * dim_a + 3])
            action_normalizers.append(get_range_normalizer_from_stat(pos_stats))
            # Assume the next dims (from 3 to dim_a-1) correspond to rotation.
            rot_stats = array_to_stats(action_data[..., i * dim_a + 3 : (i + 1) * dim_a - 1])
            action_normalizers.append(get_identity_normalizer_from_stat(rot_stats))
            # Assume the last dimension corresponds to the gripper.
            grip_stats = array_to_stats(action_data[..., (i + 1) * dim_a - 1 : (i + 1) * dim_a])
            action_normalizers.append(get_range_normalizer_from_stat(grip_stats))
        normalizer['action'] = concatenate_normalizer(action_normalizers)
        
        for key in self.lowdim_keys:
            stats = array_to_stats(data_cache[key])
            # Choose the normalization function based on key name.
            if key.endswith('pos') or ('pos_wrt' in key) or key.endswith('pos_abs'):
                norm_fn = get_range_normalizer_from_stat(stats)
                normalizer['eef_pos'] = norm_fn
            elif key.endswith('rot') or key.endswith('rot_axis_angle'):
                norm_fn = get_identity_normalizer_from_stat(stats)
                normalizer['eef_rot'] = norm_fn
            elif key.endswith('gripper_width'):
                norm_fn = get_range_normalizer_from_stat(stats)
                normalizer['gripper_width'] = norm_fn
            elif key.endswith('start'):
                norm_fn = get_range_normalizer_from_stat(stats)
                normalizer['eef_rot_start'] = norm_fn
            elif key.endswith('force'):
                norm_fn = get_range_normalizer_from_stat(stats)
                normalizer['force'] = norm_fn
            elif key.endswith('torque'):
                norm_fn = get_range_normalizer_from_stat(stats)
                normalizer['torque'] = norm_fn
            else:
                pass
                # raise RuntimeError(f"Unsupported low-dimensional key for normalization: {key}")
        
        for key in self.rgb_keys:
            normalizer['images'] = get_image_identity_normalizer()
        
        return normalizer



        

