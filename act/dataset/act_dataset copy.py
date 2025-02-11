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

from act.dataset.base_datast import BaseDataset
from act.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from act.common.pose_repr_util import convert_pose_mat_rep
from act.common.pytorch_util import dict_apply
from act.common.replay_buffer import ReplayBuffer
from act.common.sampler import ACTSequenceSampler, get_val_mask
from act.common.data_converter import ACTDataConverter
from act.dataset.base_dataset import BaseDataset
from act.common.normalizer import LinearNormalizer
from act.common.pose_util import pose_to_mat, mat_to_pose10d

from act.common.sampler import ACTSampler


class ACTDataset(BaseDataset):
    def __init__(self,
        episode_indices,   # a list of episode IDs, e.g. [0, 1, 2, …]
        shape_meta: dict,
        dataset_path: str,  # path to the original (zarr) dataset
        hdf5_path: str,     # directory to write/read HDF5 files
        camera_names: list, # list of camera names (to be passed to converter & sampler)
        chunk_size: int,
        cache_dir: str = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: float = None,
        do_convert: bool = False):

        

        # Load replay buffer
        replay_buffer = self.load_zarr(cache_dir=cache_dir)

        # Solve key and attribute
        rgb_keys, lowdim_keys, key_horizon, key_down_sample_steps, key_latency_steps = self.solve_key_attr(shape_meta=shape_meta)

        # Convert Zarr to hdf5
        converter = None
        if do_convert:
            converter = self.covert_zarr_to_hdf5(hdf5_path=hdf5_path)
        
        # Define sampler
        sampler = ACTSampler(
            episode_indices=episode_indices,
            hdf5_path=hdf5_path,
            cam_names=camera_names,
        )

        self.episode_indices = episode_indices
        self.dataset_path = dataset_path
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
        self.sampler = sampler
        self.converter = converter


    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
            sampler -> dataset -> dataloader -> model

            sampler
                
                {'obs': obs, 'action': action, 'is_pad': is_pad}

            dataset

                torch_data = {
                    'obs': dict_apply(obs_dict, torch.from_numpy),
                    'action': torch.from_numpy(data['action'].astype(np.float32))
                }
            
            model
                
                obs_dict = data['obs']
                action = data['action']

            
            define dataloader
            train_dataset = ACTDataset(train_indices, **self.cfg.task.dataset)
            train_dataloader = DataLoader(train_dataset, **self.cfg.dataloader)
            
            training loop
            for batch in train_dataloader:
                batch = dict_apply(batch, device)

                forward_dict = self.model(batch)
                loss = forward_dict['loss']
                loss.backward

            
            model input
            def __call__(self, data):
                obs_dict = data['obs']
                action = data['action']
                ...
            
            
        '''
        # sample data
        data = self.sampler.sample_item(idx)
        return data


    def covert_zarr_to_hdf5(self, hdf5_path):
        converter = ACTDataConverter(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            hdf5_path=hdf5_path,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
        )

        # Convert UMI zarr data to hdf5
        converter.process_episodes()
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