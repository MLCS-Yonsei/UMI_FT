import sys
import os

# Get the absolute path of the UMI_FT directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add UMI_FT to Python path
sys.path.append(project_root)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch
import numpy as np
from act.detr.main import build_ACT_model_and_optimizer
from act.common.normalizer import LinearNormalizer



import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self,
                 lr,
                 kl_weight,
                 lr_backbone,
                 nheads,
                 state_dim,

                 # backbone
                 backbone,
                 dilation,
                 position_embedding,
                 camera_names,

                 # Transformer
                 encoder_layers,
                 decoder_layers,
                 dim_feedforward,
                 hidden_dim,
                 drop_out,
                 num_queries,
                 pre_norm,

                 # Segmentation
                 masks,

                 # not used
                 weight_decay,
                 lr_drop,
                 clip_max_norm,

                ):
        super().__init__()

        policy_config = {
            'lr': lr,
            'kl_weight': kl_weight,
            'lr_backbone': lr_backbone,
            'nheads': nheads,
            'state_dim' : state_dim,

            'backbone': backbone,
            'dilation': dilation,
            'position_embedding': position_embedding,
            'camera_names': camera_names,

            'enc_layers': encoder_layers,
            'dec_layers': decoder_layers,
            'dim_feedforward': dim_feedforward,
            'hidden_dim': hidden_dim,
            'dropout': drop_out,
            'num_queries': num_queries,
            'pre_norm': pre_norm,

            'masks': masks,

            'weight_decay': weight_decay,
            'lr_drop': lr_drop,
            'clip_max_norm': clip_max_norm

        }
        model, optimizer = build_ACT_model_and_optimizer(policy_config)

        self.model = model # CVAE decoder
        self.optimizer = optimizer

        self.kl_weight = kl_weight
        print(f'KL Weight {self.kl_weight}')

        self.normalizer = LinearNormalizer()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def configure_optimizers(self):
        return self.optimizer
    
    def __call__(self, data):
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

        obs_dict = data['obs']
        action = data['action'] 
        is_pad = data['is_pad']
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        images = nobs['images']
        # print("nobs: ", nobs)
        nactions = self.normalizer['action'].normalize(action)
        # print("nactions: ", nactions)
        batch_size = nactions.shape[0]

        # extract low dim nobs
        low_dim_keys = ['eef_pos', 'eef_rot', 'gripper_width']
        low_dim_data = torch.cat([nobs[key] for key in low_dim_keys if key in nobs], dim=-1)

        # print("low dim data: ", low_dim_data.shape) # bs, 10
        # print("image: ", images.shape) # bs, 1, 3, 224, 224
        # print("actions: ", nactions.shape) # bs, ep max length, 10

        if nactions is not None: # training time
            actions = nactions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            # print("sliced actions: ", actions.shape) # bs, chunk size, 10

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


def test_act_policy_forward():
    batch_size = 1
    channels = 3
    img_height = 480
    img_width = 640

    num_cam = 1
    obs_dict = {
        'camera0_rgb': torch.randn(batch_size, num_cam, channels, img_height, img_width),
        'robot0_force': torch.randn(batch_size, 3),
        'robot0_torque': torch.randn(batch_size,  3),
        'robot0_eef_pos': torch.randn(batch_size,  3),
        'robot0_eef_rot_axis_angle': torch.randn(batch_size,  6),
        'robot0_gripper_width': torch.randn(batch_size, 1),
    }
    action_dim = 10 # 3 + 6 + 1 (pos: 3, rot: 6, width: 1)
    action_seq = 2
    actions = torch.randn(batch_size, action_seq, action_dim)

    shape_meta = {
        'obs': {
            'camera0_rgb': {'shape': [3, 224, 224], 'type': 'rgb'},
            'robot0_force': {'shape': [3], 'type': 'low_dim'},
            'robot0_torque': {'shape': [3], 'type': 'low_dim'},
            'robot0_eef_pos': {'shape': [3], 'type': 'low_dim'},
            'robot0_eef_rot_axis_angle': {'shape': [6], 'type': 'low_dim'},
            'robot0_gripper_width': {'shape': [1], 'type': 'low_dim'},
        }
    }
    rgb_keys = []
    force_keys = []
    torque_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta['obs']
    key_shape_map = dict()
    for key, attr in obs_shape_meta.items():
        shape = tuple(attr['shape'])
        type = attr.get('type', 'low_dim')
        key_shape_map[key] = shape
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            if key.endswith('force'):
                force_keys.append(key)
            elif key.endswith('torque'):
                torque_keys.append(key)
            else:
                lowdim_keys.append(key)

    images = None
    low_dim_data = list()
    force_data = None
    torque_data = None
    for key in rgb_keys:
        images = obs_dict[key]
    
    for key in lowdim_keys:
        low_dim_data.append(obs_dict[key])
    low_dim_data = torch.cat(low_dim_data, dim=1) # (B, 3+6+1)
    
    for key in force_keys:
        force_data = obs_dict[key]
    
    for key in torque_keys:
        torque_data = obs_dict[key]

    policy_config = {
            'lr': 0.00001,
            'kl_weight': 10,
            'lr_backbone': 0.00001,
            'nheads': 8,
            'state_dim' : 10,

            'backbone': "resnet18",
            'dilation': False,
            'position_embedding': 'sine',
            'camera_names': ["top"],

            'enc_layers': 4,
            'dec_layers': 7,
            'dim_feedforward': 3200,
            'hidden_dim': 512,
            'dropout': 0.1,
            'num_queries': 2,
            'pre_norm': True,

            'masks': False, # it should be False

            'weight_decay': 0.0001,
            'lr_drop': 200,
            'clip_max_norm': 0.1,

            'ckpt_dir': 'path',
            'policy_class': "ACT",
            'task_name': 'act',
            "seed": 42,
            "num_epochs": 1,
            'eval': False,
            'onscreen_render': True,
            'temporal_agg': True,
            'chunk_size': 2,
            'batch_size': 1

        }
    print("load policy")
    act_policy, _  = build_ACT_model_and_optimizer(policy_config)
    env_state = None
    is_pad = np.zeros([batch_size, action_seq])
    is_pad = torch.from_numpy(is_pad).bool()
    print("run policy")
    a_hat, is_pad_hat, (mu, logvar) = act_policy(low_dim_data, images, env_state, actions, is_pad)
    # a_hat, is_pad_hat, (mu, logvar) = act_policy(low_dim_data, images, force_data, torque_data, env_state, actions, is_pad)

    print("test done!")

if __name__ == "__main__":
    test_act_policy_forward()