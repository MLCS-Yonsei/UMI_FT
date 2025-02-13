from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st

import os
import h5py
import torch


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class ACTSampler:
    def __init__(self,
                 episode_indices,
                 hdf5_path,
                 cam_names,
                 ):
        self.episode_indices = episode_indices
        self.hdf5_path = hdf5_path
        self.cam_names = cam_names
        self.is_sim = None
    
    def __len__(self):
        return len(self.episode_indices)


    def sample_item(self, index):
        episode_id = self.episode_indices[index]
        dataset_path = os.path.join(self.hdf5_path, f'ep_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            start_ts = np.random.choice(episode_len)
            # get observation at start timestep only
            pos = root['/observations/eef_pos'][start_ts]
            rot = root['/observations/eef_rot'][start_ts]
            width = root['/observations/gripper_width'][start_ts]

            image_dict = dict()
            for cam_name in self.cam_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)
            
        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.cam_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # image_data = torch.from_numpy(all_cam_images)
        # pos_data = torch.from_numpy(pos).float()
        # action_data = torch.from_numpy(padded_action).float()
        # is_pad = torch.from_numpy(is_pad).bool()

        obs = {
            'eef_pos': torch.from_numpy(pos).float(),
            'eef_rot': torch.from_numpy(rot).float(),
            'gripper_width' : torch.from_numpy(width).float(),
            'images' : torch.from_numpy(all_cam_images),
        }

        action = torch.from_numpy(padded_action).float()

        is_pad = torch.from_numpy(is_pad).bool()

        # return image_data, pos_data, action_data, is_pad
        return {'obs': obs, 'action': action, 'is_pad': is_pad}





