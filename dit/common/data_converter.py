#!/usr/bin/env python3

'''
    python data_converter.py /path/to/umi/dataset /path/to/save/converted/data --verbose
    python common/data_converter.py /home/soochul/GoPro_20250519_hdf5 /home/soochul/GoPro_20250519_DiT --verbose
    
'''
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


from typing import Optional
import numpy as np
import os
import h5py
import torch
import argparse
import pickle as pkl
from tqdm import tqdm
from typing import Iterator, Tuple, Any
import glob
import cv2
import json
import io
from collections import defaultdict
import random
from copy import deepcopy
import scipy.interpolate as si
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from dit.common.normalizer import LinearNormalizer
from dit.common.normalize_util import *


IMAGE_SIZE = (256, 256)
CAM_NAMES = ['images', 'depth_images']
LOW_DIM_KEYS = ['pos', 'angle', 'start', 'width', 'force', 'torque']

def _resize_and_encode(bgr_image, size=IMAGE_SIZE):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded


def _decode_bgr_image(img):
    if isinstance(img, np.ndarray):
        img = np.moveaxis(img, 0, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert RGB to BGR
        return img  # already decoded
    return cv2.imdecode(img, 1)

def _gaussian_norm(all_acs):
    print('Using gaussian norm')
    all_acs_arr = np.array(all_acs)
    mean = np.mean(all_acs_arr, axis=0)
    std =  np.std(all_acs_arr, axis=0)
    if not std.all(): # handle situation w/ all 0 actions
        std[std == 0] = 1e-17

    for a in all_acs:
        a -= mean
        a /= std

    return dict(loc=mean.tolist(), scale=std.tolist())


def _max_min_norm(all_acs):
    print('Using max min norm')
    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr, axis=0)
    min_ac = np.min(all_acs_arr, axis=0)

    mid = (max_ac + min_ac) / 2
    delta = (max_ac - min_ac) / 2

    for a in all_acs:
        a -= mid
        a /= delta
    return dict(loc=mid.tolist(), scale=delta.tolist())

def _get_normalizer(episode_paths):
    normalizer = LinearNormalizer()

    data_cache = {key: list() for key in LOW_DIM_KEYS + ['action']}

    for episode_path in tqdm(episode_paths):
        with h5py.File(episode_path, 'r') as f:
            for key in LOW_DIM_KEYS:
                if key.endswith('pos'):
                    arr = f['/observations/eef_pos'][()]
                elif key.endswith('angle'):
                    arr = f['/observations/eef_rot'][()]
                elif key.endswith('width'):
                    arr = f[f'/observations/gripper_width'][()]
                elif key.endswith('start'):
                    arr = f['/observations/eef_rot_start'][()]
                elif key.endswith('force'):
                    arr = f['/observations/force'][()]
                elif key.endswith('torque'):
                    arr = f['/observations/torque'][()]
                else:
                    pass

                data_cache[key].append(arr)
            
            action_arr = f['/action'][()]
            data_cache['action'].append(action_arr)
        
        for key in data_cache:
            data_cache[key] = np.stack(data_cache[key], axis=0)
            B, T = data_cache[key].shape[0], data_cache[key].shape[1]
            data_cache[key] = data_cache[key].reshape(B * T, -1)

        action_data = data_cache['action']
        action_dim = action_data.shape[-1]
        num_robot = 1
        dim_a = action_dim // num_robot

        # action
        action_normalizers = []
        for i in range(num_robot):
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

        # obs
        for key in LOW_DIM_KEYS:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('angle') or 'start' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('force'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('torque'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        
        # image & depth
        for key in CAM_NAMES:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer


def convert_dataset(base_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    episode_paths = glob.glob(os.path.join(base_path, '**/*.hdf5'), recursive=True)

    if len(episode_paths) == 0:
        raise ValueError(f"No .hdf5 files found in {base_path}")
    
    print(f"Found {len(episode_paths)} episodes in {base_path}")
    print(f"Will save converted data to {save_path}")

    print("Computing normalization parameters...")
    normalizer = _get_normalizer(episode_paths)

    # out_trajectories, all_actions = [], []
    print("Converting trajectories...")
    out_trajectories = []

    for episode_path in tqdm(episode_paths):
        each_trajectory = []

        with h5py.File(episode_path, 'r') as f:
            actions = f['/action'][:]
        
            for t, a in enumerate(actions):
                # all_actions.append(a) # for normalization later

                reward = 0 # dummy reward

                # obs = dict(state=f['observations']['qpos'][t])
                pos = f['/observations/eef_pos'][t]
                rot = f['/observations/eef_rot'][t]
                rot_start = f['/observations/eef_rot_start'][t]
                width = f['/observations/gripper_width'][t]
                
                obs_concat = np.concatenate((pos, rot, rot_start, width), axis=0)
                obs = dict(state=obs_concat)

                for idx, key in enumerate(CAM_NAMES):
                    bgr_img = _decode_bgr_image(f['observations'][key]['top'][t])

                    obs[f'enc_cam_{idx}'] = _resize_and_encode(bgr_img)

                each_trajectory.append((obs, a, reward))

        out_trajectories.append(each_trajectory)

    # action_dict = _max_min_norm(all_actions)

    # with open('ac_norm.json', 'w') as f:
    #     json.dump(action_dict, f)

    normalizer_path = os.path.join(save_path, 'normalizer.pkl')
    trajectories_path = os.path.join(save_path, 'buf.pkl')

    print(f"Saving normalizer to {normalizer_path}")
    with open(normalizer_path, 'wb') as f:
        pkl.dump(normalizer, f)

    print(f"Saving trajectories to {trajectories_path}")
    with open(trajectories_path, 'wb') as f:
        pkl.dump(out_trajectories, f)

    print(f"Conversion complete!")
    print(f"- Processed {len(out_trajectories)} episodes")
    print(f"- Total timesteps: {sum(len(traj) for traj in out_trajectories)}")
    print(f"- Files saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert UMI dataset to DiT format')
    parser.add_argument('base_path', type=str, 
                       help='Path to the UMI dataset directory containing .hdf5 files')
    parser.add_argument('save_path', type=str,
                       help='Directory where converted files will be saved')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Expand paths
    base_path = os.path.expanduser(args.base_path)
    save_path = os.path.expanduser(args.save_path)
    
    # Validate paths
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")
    
    if args.verbose:
        print(f"Base path: {base_path}")
        print(f"Save path: {save_path}")
    
    # Convert dataset
    convert_dataset(base_path, save_path)


if __name__ == '__main__':
    main()