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

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

base_path = os.path.dirname(os.path.abspath(__file__))

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=os.path.join(base_path, "config"),
    config_name="finetune_ddp.yaml"
)
def main(cfg: OmegaConf):
    local_rank, rank, world_size = init_ddp()

    # override batch_size per‐GPU
    cfg.batch_size = max(cfg.batch_size // world_size, 1)


    # Get the workspace class
    cls = hydra.utils.get_class(cfg.workspace._target_)
    
    workspace: BaseWorkspace = cls(cfg, local_rank=local_rank, rank=rank, world_size=world_size)
    workspace.run()

    dist.destroy_process_group()

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()