import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class CausalConv1D(nn.Conv1d):
    """
    A causal 1D convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ForceEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.frc_encoder = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=1),          # (B, 6, T) -> (B, 32, T)
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(2),                  # (B, 32, T) -> (B, 32, 1)
            nn.Conv1d(32, 2*z_dim, kernel_size=1),    # (B, 2*z_dim, 1)
            nn.LeakyReLU(0.1, inplace=True),
            # CausalConv1D(6, 16, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.1, inplace=True),
            # CausalConv1D(16, 32, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.1, inplace=True),
            # CausalConv1D(32, 64, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.1, inplace=True),
            # CausalConv1D(64, 128, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.1, inplace=True),
            # CausalConv1D(128, 2 * self.z_dim, kernel_size=2, stride=2),
            # nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.frc_encoder(force)


class TimmObsEncoderFTDepth(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            fuse_mode: str,
            model_rgb: dict,
            model_depth: dict,
            model_ft: dict,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encoding: str='learnable',

        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        self.rgb_backbone = self._build_backbone(
            model_cfg=model_rgb,
            global_pool=global_pool,
            use_group_norm=use_group_norm,
            downsample_ratio=downsample_ratio)
        
        if share_rgb_model and model_depth['name'] == model_rgb['name']:
            self.depth_backbone = self.rgb_backbone
        else:
            self.depth_backbone = self._build_backbone(
                model_cfg=model_depth,
                global_pool=global_pool,
                use_group_norm=use_group_norm,
                downsample_ratio=downsample_ratio
            )
        
        self.force_backbone = ForceEncoder(model_ft['feature_dim'])

        rgb_keys = list()
        depth_keys = list()
        force_keys = list()
        torque_keys = list()
        wrench_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                key_model_map[key] = self.rgb_backbone

                this_transform = transform
                key_transform_map[key] = this_transform

            elif type == 'depth':
                depth_keys.append(key)
                key_model_map[key] = self.depth_backbone
                this_transform = transform
                key_transform_map[key] = this_transform

            elif type == 'low_dim':
                if "force" in key:
                    force_keys.append(key)
                    wrench_keys.append(key)
                    key_model_map[key] = self.force_backbone
                elif "torque" in key:
                    torque_keys.append(key)
                    wrench_keys.append(key)
                    key_model_map[key] = self.force_backbone
                else:
                    if not attr.get('ignore_by_policy', False):
                        low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
            

        feature_dim_map = {
            # backbone : (dim, feat_map_h, feat_map_w)  (only needed for resnet/convnext)
            'resnet':   (512, 7, 7) if downsample_ratio == 32 else (256, 14, 14),
            'convnext': (1024, 7, 7)
        }
        backbone_name = model_rgb['name']
        if backbone_name.startswith('resnet'):
            feature_dim, h, w = feature_dim_map['resnet']
        elif backbone_name.startswith('convnext'):
            feature_dim, h, w = feature_dim_map['convnext']
        elif backbone_name.startswith('vit'):
            feature_dim = timm.create_model(backbone_name, pretrained=False).num_features
            h = w = None  # vit uses tokens
        else:
            raise NotImplementedError(backbone_name)
        
        self.feature_aggregation = feature_aggregation
        if backbone_name.startswith('vit'):
            if feature_aggregation and feature_aggregation != 'all_tokens':
                logger.warning("ViT uses CLS token; overriding feature_aggregation.")
            self.feature_aggregation = None
        else:
            feature_map_shape = [x // downsample_ratio for x in image_shape]
            if self.feature_aggregation == 'soft_attention':
                self.attention = nn.Sequential(
                    nn.Linear(feature_dim, 1, bias=False),
                    nn.Softmax(dim=1)
                )
            elif self.feature_aggregation == 'spatial_embedding':
                self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
            elif self.feature_aggregation == 'transformer':
                if position_encoding == 'learnable':
                    self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
                elif position_encoding == 'sinusoidal':
                    num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                    self.position_embedding = torch.zeros(num_features, feature_dim)
                    position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                    self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                    self.position_embedding[:, 1::2] = torch.cos(position * div_term)
                self.aggregation_transformer = nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                    num_layers=4)
            elif self.feature_aggregation == 'attention_pool_2d':
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=feature_dim,
                    num_heads=feature_dim // 64,
                    output_dim=feature_dim
                )
            logger.info(
                "number of parameters: %e", sum(p.numel() for p in self.parameters())
            )
        

        if fuse_mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim * 3, 1024), nn.ReLU(), nn.Linear(1024, 512)
            )

        elif fuse_mode == "modality-attention":
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0,
            )
            n_features = len(rgb_keys) * shape_meta["obs"]["camera0_rgb"]["horizon"] + len(depth_keys) * shape_meta["obs"]["camera0_depth"]["horizon"]+ len(force_keys) * shape_meta["obs"]["robot0_force"]["horizon"] # we will use wrench - force replace the wrench config
            self.linear_projection = nn.Linear(
                feature_dim * n_features, feature_dim
            )
            if position_encoding == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(n_features, feature_dim)
                )
        

        rgb_keys = sorted(rgb_keys)
        depth_keys = sorted(depth_keys)
        force_keys = sorted(force_keys)
        torque_keys = sorted(torque_keys)
        wrench_keys = sorted(wrench_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('depth keys:         ', depth_keys)
        print('force keys:         ', force_keys)
        print('torque keys:         ', torque_keys)
        print('wrench keys:         ', wrench_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name_rgb = model_rgb['name']
        self.model_name_depth = model_depth['name']
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.force_keys = force_keys
        self.torque_keys = torque_keys
        self.wrench_keys = wrench_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.fuse_mode = fuse_mode
        self.position_encoding = position_encoding

    def _build_backbone(self, model_cfg, global_pool, use_group_norm, downsample_ratio):
        name       = model_cfg['name']
        pretrained = model_cfg.get('pretrained', False)
        frozen     = model_cfg.get('frozen', False)

        assert global_pool == ''
        backbone = timm.create_model(
            model_name=name,
            pretrained=pretrained,
            global_pool=global_pool,
            num_classes=0
        )

        if name.startswith('resnet'):
            modules = list(backbone.children())
            idx = -2 if downsample_ratio == 32 else -3   # same logic as before
            backbone = torch.nn.Sequential(*modules[:idx])
        elif name.startswith('convnext'):
            # identical to your original branch
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        # (add vit/efficientnet branches if you need them)

        if use_group_norm and not pretrained:
            backbone = replace_submodules(
                backbone,
                predicate=lambda m: isinstance(m, nn.BatchNorm2d),
                func=lambda m: nn.GroupNorm(
                    num_groups=(m.num_features // 16) or 1, num_channels=m.num_features
                )
            )

        if frozen:
            for p in backbone.parameters():
                p.requires_grad = False

        return backbone
    
    def aggregate_feature(self, feature, type='rgb'):
        if type == 'rgb':
            model_name = self.model_name_rgb
        else:
            model_name = self.model_name_depth

        if model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        '''
            B : batch size
            T : temporal sequence length, the number of time steps in sequence data
            C : channels : 3
            H : height : 224
            W : width : 224
            D: data dimension, the size of the feature vector for low dimensional data

            camera0_rgb dict:  torch.Size([1, 2, 3, 224, 224])
            camera0_depth dict:  torch.Size([1, 2, 3, 224, 224])
            robot0_eef_pos dict:  torch.Size([1, 2, 3]) 
            robot0_eef_rot_axis_angle dict:  torch.Size([1, 2, 6])
            robot0_gripper_width dict:  torch.Size([1, 2, 1])
            robot0_eef_rot_axis_angle_wrt_start dict:  torch.Size([1, 2, 6])
            robot0_force dict:  torch.Size([1, 2, 3])
            robot0_torque dict:  torch.Size([1, 2, 3])
        '''
        features = list()
        modality_features = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature, type='rgb')
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            # print("vision feature shape: ", feature.shape) # (2, 768)
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.reshape(B, T, -1))
        
        for key in self.depth_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature, type='depth')
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            # print("depth feature shape: ", feature.reshape(B, T, -1).shape) # (1, 2, 768)
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.reshape(B, T, -1))
        
        for cam_idx, force_key in enumerate(self.force_keys):
            forces = obs_dict[force_key]
            B, T = forces.shape[:2]
            assert B == batch_size
            assert forces.shape[2:] == self.key_shape_map[force_key]

            torque_key = self.torque_keys[cam_idx]
            torques = obs_dict[torque_key]
            B, T = torques.shape[:2]
            assert B == batch_size
            assert torques.shape[2:] == self.key_shape_map[torque_key]

            wrench = torch.cat([forces, torques], dim=-1) # (B, T, 6) 
            B, T = wrench.shape[:2]
            assert B == batch_size
            assert wrench.shape[2:][0] == self.key_shape_map[force_key][0] + self.key_shape_map[torque_key][0]
            wrench = wrench.permute(0, 2, 1) # (B, 6, T)
            feature = self.key_model_map[force_key](wrench.float())[:, :, :] # (B, 768, T)
            # print("wrench feature shape: ", feature.shape) # (1, 768, 2)
            assert feature.shape[0] == B
            features.append(feature.reshape(B, -1))
            modality_features.append(feature.reshape(B, T, -1))



        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
            low_dim_features.append(data.reshape(B, -1))
        
        # print("modality features shape: ", torch.cat(modality_features, dim=1).shape) # (1, 5, 768)
        # print("features shape: ", torch.cat(features, dim=-1).shape) # (1, 3872)
        
        # concatenate all features
        if self.fuse_mode == "concat":
            result = torch.cat(features, dim=-1)
        elif self.fuse_mode == "mlp":
            result = self.mlp(torch.cat(modality_features, dim=-1))
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        elif self.fuse_mode == "modality-attention":
            in_embeds = torch.cat(modality_features, dim=1)  # [batch, n_features, D]
            if self.position_encoding == "learnable":
                if self.position_embedding.device != in_embeds.device:
                    self.position_embedding = self.position_embedding.to(feature.device)
                in_embeds = in_embeds + self.position_embedding
            out_embeds = self.transformer_encoder(in_embeds)  # [batch, n_features, D]
            result = torch.concat(
                [out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1
            )
            result = self.linear_projection(result)
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)

        return result
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device) # (B=1, horizon, shape)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoderFTDepth(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
