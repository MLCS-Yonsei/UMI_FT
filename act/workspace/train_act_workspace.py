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
from act.workspace.base_workspace import BaseWorkspace
from act.policy.act_policy import ACTPolicy
from act.dataset.base_dataset import BaseDataset
from act.env_runner.base_image_runner import BaseImageRunner
from act.common.checkpoint_util import TopKCheckpointManager
from act.common.json_logger import JsonLogger
from act.common.pytorch_util import dict_apply
from accelerate import Accelerator

from copy import deepcopy
from act.common.pytorch_util import compute_dict_mean, detach_dict
from act.dataset.act_dataset import ACTDataset
from act.common.data_converter import ACTDataConverter

import time


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
        self.num_episodes = self.cfg.num_episodes

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']
        
        self.convert_data = True # False

        
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


        # configure train and validation dataloader
        dataset, val_dataset, train_dataloader, val_dataloader = self.load_data()
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))


        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)

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
                epoch_start_time = time.time()

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

                    # print(f'Val loss:   {epoch_val_loss:.5f}')
                    summary_string = ''
                    for k, v in epoch_summary.items():
                        summary_string += f'{k}: {v.item():.3f} '
                    # print(summary_string)

                    val_log = {f"val/{k}": v.item() for k, v in epoch_summary.items()}
                    val_log['epoch'] = epoch
                    # Here we log validation metrics using the current global_step
                    accelerator.log(val_log, step=self.global_step)
                    json_logger.log(val_log)


                # training
                self.model.train()
                self.optimizer.zero_grad()
                
                # TODO : define step log elements
                # step_log = dict()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # normalizer is in a model forward
                    # forward dict is a loss dict
                    forward_dict = self.model(batch)
                    loss = forward_dict['loss']
                    loss.backward()

                    # step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # log train history
                    train_history.append(detach_dict(forward_dict))
                    step_log = {
                        'train_loss' : loss,
                        'l1_loss': forward_dict['l1'].item() if 'l1' in forward_dict else None,
                        'kl_loss': forward_dict['kl'].item() if 'kl' in forward_dict else None,
                        'global_step' : self.global_step,
                        'epoch' : self.epoch,
                    }
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                
                # training summary
                epoch_duration = time.time() - epoch_start_time
                print("Epoch duration: ", epoch_duration)

                epoch_summary = compute_dict_mean(train_history[(batch_idx + 1)*epoch : (batch_idx+1)*(epoch+1)])
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

                    # recover the DDP model (Distributed Data Parallel)
                    self.model = model_ddp

        accelerator.end_training()

    def load_data(self):
        train_ratio = 1 - self.cfg.task.dataset.val_ratio
        shuffled_indices = np.random.permutation(self.num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * self.num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * self.num_episodes):]
        
        print("Loading Dataset")

        train_dataset : ACTDataset
        train_dataset = hydra.utils.instantiate(self.cfg.task.dataset, episode_indices=train_indices, camera_names=self.cfg.camera_names)

        val_dataset : ACTDataset
        val_dataset = hydra.utils.instantiate(self.cfg.task.dataset, episode_indices=val_indices, camera_names=self.cfg.camera_names)

        if not self.convert_data:
            train_dataset.convert_zarr_to_hdf5()
            self.convert_data = True

        train_dataloader = DataLoader(train_dataset, **self.cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **self.cfg.val_dataloader)
        print('train dataset:', len(train_dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))
        return train_dataset, val_dataset, train_dataloader, val_dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainACTWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()