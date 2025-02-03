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
        self.normalizer = LinearNormalizer()

        # define observation keys
        self.rgb_keys = []
        self.force_keys = []
        self.torque_keys = []
        self.lowdim_keys = []

        obs_shape_meta = shape_meta['obs']
        key_shape_map = dict()

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape

            if type == 'rgb':
                self.rgb_keys.append(key)
            elif type == 'low_dim':
                if key.endswith('force'):
                    self.force_keys.append(key)
                elif key.endswith('torque'):
                    self.torque_keys.append(key)
                else:
                    self.lowdim_keys.append(key)


    
    def __call__(self, obs_dict):
        '''
            input : dict type batch data

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
            
            model : DETRVAE

                - forward ( qpos, image, env_state, actions=None, is_pad=None )
                    qpos: batch, qpos_dim
                    image: batch, num_cam, channel, height, width
                    env_state: None
                    actions: batch, seq, action_dim
            
            B : batch size
            T : temporal sequence length, the number of time steps in sequence data
            C : channels : 3
            H : height : 224
            W : width : 224
            D: data dimension, the size of the feature vector for low dimensional data

        '''
        env_state = None
        images = None

        # process inputs including bi-manual case
        for key in self.rgb_keys:
            images = obs_dict[key]
        
        for key in self.lowdim_keys:
            low_dim_data = obs_dict[key]
        
        for key in self.force_keys:
            force_data = obs_dict[key]
        
        for key in self.torque_keys:
            torque_data = obs_dict[key]
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict['obs'])
        nactions = self.normalizer['action'].normalize(obs_dict['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        if nactions is not None: # training time
            actions = nactions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(low_dim_data, images, env_state, actions, is_pad)

            # TODO : change model architecture for force and torque
            # a_hat, is_pad_hat, (mu, logvar) = self.model(low_dim_data, images, force_data, torque_data, env_state, actions, is_pad)
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        
        else: # inference time
            a_hat, _, (_, _) = self.model(low_dim_data, images, env_state) # no action, sample from prior
            return a_hat
    
    def configure_optimizers(self):
        return self.optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld