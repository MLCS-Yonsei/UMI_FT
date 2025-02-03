import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from diffusion_policy.model.common.normalizer import LinearNormalizer

import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override, 
                shape_meta: dict, 
                obs_encoder: ...,
                num_inference_steps=None,
                input_pertub=0.1,
                 # arch
                n_layer=7,
                n_head=8,
                n_emb=768,
                n_compress_emb=288,
                p_drop_attn=0.1,
                **kwargs
                ):
        super().__init__()
        # TODO : change args_override with hydra conf
        model, optimizer = build_ACT_model_and_optimizer(args_override)

        self.model = model # CVAE decoder
        self.optimizer = optimizer

        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

        self.obs_encoder = ... # self.model.cls_embed, self.model.encoder_action_proj, self.model.joint_proj

    
    def __call__(self, qpos, image, actions=None, is_pad=None):
        '''
            data = {

                'obs' : obs_dict <torch.from_numpy>

                'action' : action_list <torch.from_numpy>
            }

            obs_dict 
                - key : robot_eef_pos, robot_eef_rot_axis_angle, robot_gripper_width, robot_ robot_force, robot_torque
                - robot_eef_pos : 3d
                - robot_eef_rot_axis_angle: 6d
                - robot_gripper_width : 1d
                - robot_force: 3d
                - robot_torque: 3d

            action_list
                - concat ([ action_pose , action_gripper ])
                - action_pose : 10d
                - action_gripper : 1d

        '''
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat
    
    def configure_optimizers(self):
        return self.optimizer
    