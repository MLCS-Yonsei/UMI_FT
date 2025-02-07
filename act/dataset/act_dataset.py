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


class ACTDataset(BaseDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 cache_dir: Optional[str]=None,
                 pose_repr: dict={},
                 action_padding: bool=False,
                 temporally_independent_normalization: bool=False,
                 repeat_frame_prob: float=0.0,
                 seed: int=42,
                 val_ratio: float=0.0,
                 max_duration: Optional[float]=None
                 ):
        
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
    
    def get_validation_dataset(self):
        ...
    
    