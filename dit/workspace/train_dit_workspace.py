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
import tqdm
import numpy as np

from dit.workspace.base_workspace import BaseWorkspace
from dit.policy.dit_policy import DiffusionTransformerPolicy
from dit.trainers.base import BaseTrainer
from dit.task.task import BCTask
from dit.common import misc, transforms

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiTWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir = None):
        super().__init__(cfg, output_dir=output_dir)

        resume_model = misc.init_job(cfg)

        # set seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed + 1)

        self.model: DiffusionTransformerPolicy = hydra.utils.instantiate(cfg.agent)
        self.trainer: BaseTrainer = hydra.utils.instantiate(cfg.trainer, model=self.model, device_id=0)
        self.task: BCTask = hydra.utils.instantiate(
            cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers
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
        assert misc.GLOBAL_STEP >= 0, "GLOBAL_STEP not loaded correctly!"

        # register checkpoint handler and enter train loop
        misc.set_checkpoint_handler(self.trainer, cfg.checkpoint_path)
        print(f"Starting at Global Step {misc.GLOBAL_STEP}")

        self.cfg = cfg

    def run(self):
        self.trainer.set_train()
        train_iterator = iter(self.task.train_loader)

        for itr in (
            pbar := tqdm.tqdm(range(self.cfg.max_iterations), postfix=dict(Loss=None))
        ):
            if itr < misc.GLOBAL_STEP:
                continue

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

            pbar.set_postfix(dict(Loss=loss.item()))
            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % self.cfg.schedule_freq == 0:
               self.trainer.step_schedule()

            if misc.GLOBAL_STEP % self.cfg.eval_freq == 0:
                self.trainer.set_eval()
                self.task.eval(self.trainer, misc.GLOBAL_STEP)
                self.trainer.set_train()

            if misc.GLOBAL_STEP >= self.cfg.max_iterations:
                self.trainer.save_checkpoint(self.cfg.checkpoint_path, misc.GLOBAL_STEP)
                return
            elif misc.GLOBAL_STEP % self.cfg.save_freq == 0:
                self.trainer.save_checkpoint(self.cfg.checkpoint_path, misc.GLOBAL_STEP)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiTWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()


