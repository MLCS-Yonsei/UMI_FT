if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import OmegaConf
import pathlib
import tqdm
import numpy as np

from dit.workspace.base_workspace import BaseWorkspace
from dit.policy.dit_policy import DiffusionTransformerPolicy
from dit.trainers.base import BaseTrainer
from dit.task.task_ddp import BCTaskDDP
from dit.common import misc, transforms

OmegaConf.register_new_resolver("eval", eval, replace=True)


def setup_ddp(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training environment."""
    dist.destroy_process_group()



class TrainDiTWorkspaceDDP(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, rank=0, world_size=1, output_dir = None):
        super().__init__(cfg, output_dir=output_dir)

        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        if self.is_main_process:
            resume_model = misc.init_job(cfg)
        else:
            resume_model = None
        
        # Broadcast resume_model to all processes
        if world_size > 1:
            resume_model_list = [resume_model]
            dist.broadcast_object_list(resume_model_list, src=0)
            resume_model = resume_model_list[0]


        # set seed
        torch.manual_seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank + 1)

        self.model: DiffusionTransformerPolicy = hydra.utils.instantiate(cfg.agent)
        self.model = self.model.to(rank)

        if world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[rank], 
                output_device=rank,
                find_unused_parameters=True  # Set to False if you're sure all parameters are used
            )
        

        self.trainer: BaseTrainer = hydra.utils.instantiate(
            cfg.trainer, 
            model=self.model, 
            device_id=rank)


        self.task: BCTaskDDP = hydra.utils.instantiate(
            cfg.task, 
            batch_size=cfg.batch_size // world_size,  # Divide batch size
            num_workers=cfg.num_workers,
            rank=rank,
            world_size=world_size,
            use_distributed_sampler=(world_size > 1)
        )

        # create a gpu train transform (if used)
        self.gpu_transform = (
            transforms.get_gpu_transform_by_name(cfg.train_transform)
            if "gpu" in cfg.train_transform
            else None
        )

        # restore/save the model as required
        if resume_model is not None:
            misc.GLOBAL_STEP = self.trainer.load_checkpoint(resume_model)
        elif misc.GLOBAL_STEP == 0:
            self.trainer.save_checkpoint(cfg.checkpoint_path, misc.GLOBAL_STEP)
        

        if world_size > 1:
            global_step_tensor = torch.tensor(misc.GLOBAL_STEP, dtype=torch.long, device=rank)
            dist.broadcast(global_step_tensor, src=0)
            misc.GLOBAL_STEP = global_step_tensor.item()

        
        assert misc.GLOBAL_STEP >= 0, "GLOBAL_STEP not loaded correctly!"

        # register checkpoint handler and enter train loop
        if self.is_main_process:
            misc.set_checkpoint_handler(self.trainer, cfg.checkpoint_path)
            print(f"Starting at Global Step {misc.GLOBAL_STEP}")

        self.cfg = cfg

    def run(self):
        self.trainer.set_train()
        train_iterator = iter(self.task.train_loader)

        if self.is_main_process:
            pbar = tqdm.tqdm(range(self.cfg.max_iterations), postfix=dict(Loss=None))
        else:
            pbar = range(self.cfg.max_iterations)


        for itr in (
            pbar := tqdm.tqdm(range(self.cfg.max_iterations), postfix=dict(Loss=None))
        ):
            if itr < misc.GLOBAL_STEP:
                continue
            

            if (hasattr(self.task.train_loader, 'sampler') and 
                hasattr(self.task.train_loader.sampler, 'set_epoch')):
                self.task.train_loader.sampler.set_epoch(itr)
            

            # infinitely sample batches until the train loop is finished
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.task.train_loader)
                batch = next(train_iterator)

            # handle the image transform on GPU if specified
            if self.gpu_transform is not None:
                (imgs, obs), actions, mask = batch
                imgs = {k: v.to(self.trainer.device_id) for k, v in imgs.items()}
                imgs = {k: self.gpu_transform(v) for k, v in imgs.items()}
                batch = ((imgs, obs), actions, mask)

            self.trainer.optim.zero_grad()
            loss = self.trainer.training_step(batch, misc.GLOBAL_STEP)
            loss.backward()
            self.trainer.optim.step()

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(dict(Loss=loss.item()))

            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % self.cfg.schedule_freq == 0:
               self.trainer.step_schedule()

            if misc.GLOBAL_STEP % self.cfg.eval_freq == 0:
                self.trainer.set_eval()
                if self.is_main_process:
                    self.task.eval(self.trainer, misc.GLOBAL_STEP)
                # Wait for evaluation to complete on all processes
                if self.world_size > 1:
                    dist.barrier()
                self.trainer.set_train()

            if misc.GLOBAL_STEP >= self.cfg.max_iterations:
                if self.is_main_process:
                    self.trainer.save_checkpoint(self.cfg.checkpoint_path, misc.GLOBAL_STEP)
                return
            elif misc.GLOBAL_STEP % self.cfg.save_freq == 0:
                self.trainer.save_checkpoint(self.cfg.checkpoint_path, misc.GLOBAL_STEP)


def run_ddp_training(rank, world_size, cfg):
    """Function to run on each GPU process."""
    try:
        # Setup distributed training
        setup_ddp(rank, world_size)
        
        # Create and run workspace
        workspace = TrainDiTWorkspaceDDP(cfg, rank=rank, world_size=world_size)
        workspace.run()
        
    except Exception as e:
        if rank == 0:  # Only log on main process
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_ddp()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    world_size = cfg.get('devices', 1)
    
    if world_size == 1:
        # Single GPU training
        workspace = TrainDiTWorkspaceDDP(cfg, rank=0, world_size=1)
        workspace.run()
    else:
        # Multi-GPU training
        mp.spawn(run_ddp_training, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()


