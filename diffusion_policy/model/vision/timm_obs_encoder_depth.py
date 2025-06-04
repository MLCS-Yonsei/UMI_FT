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
    

class TimmObsEncoderDepth(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_rgb: dict,
            model_depth: dict,
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
            position_encording: str='learnable',

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
        
        rgb_keys = list()
        depth_keys = list()
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
                if position_encording == 'learnable':
                    self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
                elif position_encording == 'sinusoidal':
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
        

        rgb_keys = sorted(rgb_keys)
        depth_keys = sorted(depth_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('depth keys:         ', depth_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name_rgb = model_rgb['name']
        self.model_name_depth = model_depth['name']
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

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
        features = list()
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
            features.append(feature.reshape(B, -1))
        
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
            features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

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
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoderDepth(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
