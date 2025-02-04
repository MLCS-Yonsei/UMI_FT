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
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from act.policy.act_policy import ACTPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

from copy import deepcopy
from act.common.pytorch_util import compute_dict_mean, detach_dict


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainACTWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir = None):
        super().__init__(cfg, output_dir=output_dir)

        print(output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ACTPolicy = hydra.utils.instantiate(cfg.policy)

        # configure params
        param_groups = [
            {'params': self.model.model.parameters()},
        ]

        
        # configure optimizer
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )
        # TODO : modify ACT code for hydra
        # self.optimizer = self.model.configure_optimizers()

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']
        
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure wandb
        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        # configure env 
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer)
        device = self.model.device
        
        # TODO change to ACT style
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            train_history = []
            validation_history = []
            min_val_loss = np.inf
            best_ckpt_info = None

            for epoch in range(cfg.training.num_epochs):

                # validation
                with torch.inference_mode():
                    policy = accelerator.unwrap_model(self.model)
                    policy.eval()
                    epoch_dicts = []
                    for batch_idx, batch in enumerate(val_dataloader):
                        forward_dict = policy(batch)
                        epoch_dicts.append(forward_dict)
                    
                    # validation summary
                    epoch_summary = compute_dict_mean(epoch_dicts)
                    validation_history.append(epoch_summary)
                    epoch_val_loss = epoch_summary['loss']
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

                    print(f'Val loss:   {epoch_val_loss:.5f}')
                    summary_string = ''
                    for k, v in epoch_summary.items():
                        summary_string += f'{k}: {v.item():.3f} '
                    print(summary_string)


                # training
                self.model.train()
                self.optimizer.zero_grad()
                
                # TODO : define step log elements
                # step_log = dict()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # forward dict is a loss dict
                    forward_dict = self.model(batch)
                    loss = forward_dict['loss']
                    loss.backward()

                    # step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # log train history
                    train_history.append(detach_dict(forward_dict))
                    # step_log = {
                    #     'train_loss' : loss,
                    #     'global_step' : self.global_step,
                    #     'epoch' : self.epoch,
                    # }
                    # is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    # if not is_last_batch:
                    #     accelerator.log(step_log, step=self.global_step)
                    #     json_logger.log(step_log)
                    #     self.global_step += 1
                
                # training summary
                epoch_summary = compute_dict_mean(train_history[(batch_idx + 1)*epoch,(batch_idx+1)*(epoch+1)])
                epoch_train_loss = epoch_summary['loss']
                print(f'Train loss: {epoch_train_loss:.5f}')
                summary_string = ''
                for k, v in epoch_summary.items():
                    summary_string += f'{k}: {v.item():.3f} '
                print(summary_string)

                # Checkpointing
                if (epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
                    print(f'Training finished: val loss {min_val_loss:.6f} at epoch {best_epoch}')

                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    
                    # sanitize metric names
                    # metric_dict = dict()
                    # for key, value in step_log.items():
                    #     new_key = key.replace('/', '_')
                    #     metric_dict[new_key] = value
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

        accelerator.end_training()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainACTWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()