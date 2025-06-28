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

from dit.codecs.imagecodecs_numcodecs import register_codecs
from dit.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from dit.common.pose_repr_util import convert_pose_mat_rep
from dit.common.pytorch_util import dict_apply
from dit.common.replay_buffer import ReplayBuffer
from dit.common.sampler_ft_depth import SequenceSamplerFTDepth, get_val_mask
from dit.dataset.base_dataset import BaseDataset
from dit.model.common.normalizer import LinearNormalizer
from dit.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()

