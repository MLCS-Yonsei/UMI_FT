# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dit.policy.base_policy import BaseAgent as BasePolicy
from dit.models.diffusion import _DiffusionTransformerNoiseNetwork

class DiffusionTransformerPolicy(BasePolicy):
    def __init__(
        self,
        features,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=None,
        token_dim=None,
        noise_net_kwargs=dict(),
    ):
        
        # initialize obs and img tokenizers
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
        )

        self.noise_net = _DiffusionTransformerNoiseNetwork(
            action_dim=ac_dim,
            action_chunk=ac_chunk,
            **noise_net_kwargs,
        )
        self._action_dim, self._action_chunk = ac_dim, ac_chunk

        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"

        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps

        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
    
    @property
    def action_chunk(self):
        return self._action_chunk

    @property
    def action_dim(self):
        return self._action_dim

    def forward(self, imgs, obs, action_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device

        s_t = self.tokenize_obs(imgs, obs)
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        # [B, T, D] T: action chunk size, D: action dim
        mask = mask_flat.reshape((B, self._action_chunk, self.action_dim))
        actions = action_flat.reshape((B, self.action_chunk, self.action_dim))
        
        noise = torch.randn_like(actions)

        # construct noise actions given actions, noise, and diffusion schedule
        noise_actions = self.diffusion_schedule.add_noise(actions, noise, timesteps)

        # get predictions
        _, noise_pred = self.noise_net(noise_actions, timesteps, s_t)

        # calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum(1)
        return loss.mean()
    
    def get_actions(self, imgs, obs, n_steps = None):
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        encoder_cache = None

        noise_actions = torch.randn(B, self.action_chunk, self.action_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert (
                n_steps <= self._train_diffusion_steps
            ), f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps
        
        encoder_cache = self.noise_net.forward_encoder(s_t)

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alpha_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )

        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net.forward_decoder(noise_actions, batched_timestep, encoder_cache)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        # return final action post diffusion
        return noise_actions