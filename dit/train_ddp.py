"""
cd umi_act/UMI_Ft
python -m dit.train_ddp.py

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
import torch.multiprocessing as mp
from dit.workspace.base_workspace import BaseWorkspace
from dit.common import misc

base_path = os.path.dirname(os.path.abspath(__file__))

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=os.path.join(base_path, "config"),
    config_name="finetune_ddp.yaml"
)
def main(cfg: OmegaConf):
    # Get the workspace class
    cls = hydra.utils.get_class(cfg.workspace._target_)
    
    # Check if we're doing multi-GPU training
    world_size = cfg.get('devices', 1)
    
    if world_size == 1:
        # Single GPU training - use original approach
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()
    else:
        # Multi-GPU training - need to use DDP approach
        if 'DDP' not in cfg.workspace._target_:
            print(f"Warning: Using {world_size} GPUs but workspace is not DDP-enabled.")
            print("Make sure to use TrainDiTWorkspaceDDP for multi-GPU training.")
        
        # The DDP workspace will handle the multiprocessing internally
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()