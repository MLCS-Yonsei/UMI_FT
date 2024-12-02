from typing import Optional, Tuple
import torch
import warnings
import math
from torch.nn import functional as F
from torch.distributions import Normal

import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import numpy as np

class VTTObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 img_size: int = 224,
                 img_patch_size: int = 14,
                 in_channel: int = 3,
                 n_emb: int = 384,
                 n_low_emb: int = 288,
                 n_compress_emb: int = 288,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 n_sensor: int = 1, # 2 for bi-manual
                 global_pool: str='',
                 **kwargs
                 ):
        super().__init__()

        # vision and f/t value -> VTT
        # other low dim value -> key projection map
        
        # key list for each variables
        self.rgb_keys = []
        self.force_keys = []
        self.torque_keys = []
        self.lowdim_keys = []

        # projection map for low dim value
        self.lowdim_projections = nn.ModuleDict()

        # shape map for each key
        key_shape_map = dict()

        self.shape_meta = shape_meta
        obs_shape_meta = shape_meta['obs'] # obs_dict

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

        # print(f"rgb keys: {self.rgb_keys}") # rgb keys: ['camera0_rgb']
        # print(f"force keys: {self.force_keys}") # force keys: ['robot0_force']
        # print(f"torque keys: {self.torque_keys}") # torque keys: ['robot0_torque']
        # print(f"lowdim keys: {self.lowdim_keys}") # lowdim keys: ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']

        # projection for low dim value
        for key in self.lowdim_keys:
            shape = obs_shape_meta[key]['shape']
            dim = int(np.prod(shape))
            # print(f"{key} has dim {dim}")
            # robot0_eef_pos has dim 3
            # robot0_eef_rot_axis_angle has dim 6
            # robot0_gripper_width has dim 1
            self.lowdim_projections[key] = nn.Linear(dim, n_low_emb)
        
        self.n_emb = n_emb
        self.key_shape_map = key_shape_map
        #print("key shape map: ", self.key_shape_map) 
        # key shape map:  {'camera0_rgb': (3, 224, 224), 
        # 'robot0_eef_pos': (3,), 
        # 'robot0_eef_rot_axis_angle': (6,), 
        # 'robot0_gripper_width': (1,), 
        # 'robot0_eef_rot_axis_angle_wrt_start': (6,), 
        # 'robot0_force': (3,), 
        # 'robot0_torque': (3,)}


        self.vtt_model = VTT(
            img_size=img_size,
            img_patch_size=img_patch_size,
            tactile_patches=n_sensor,
            in_channel=in_channel,
            embed_dim=n_emb,
            n_compress_emb=n_compress_emb,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        
        
    def forward(self, obs_dict):
        # B : batch size
        # T : temporal sequence length, the number of time steps in sequence data
        # C : channels : 3
        # H : height : 224
        # W : width : 224
        # D: data dimension, the size of the feature vector for low dimensional data
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process image and force/torque
        assert len(self.rgb_keys) == 1 # is it possible to change it to fit with image observation horizon?
        image_key = self.rgb_keys[0] # ex: camera0_rgb, camera1_rgb (bi-manual)
        images = obs_dict[image_key] # Shape: B, T, C, H, W
        B, T = images.shape[:2]
        # print(f"images T: {T}") 

        force_key = self.force_keys[0]
        forces = obs_dict[force_key] # Shape : B, T, 3
        # print("forces shape: ", forces.shape)
        B, T = forces.shape[:2]
        assert B == batch_size
        if len(forces.shape) == 2: # for last batch
            forces = forces.unsqueeze(1) 
        assert forces.shape[2:] == self.key_shape_map[force_key] # 3
        # print(f"force T: {T}") 

        torque_key = self.torque_keys[0]
        torques = obs_dict[torque_key] # Shape : B, T, 3
        # print("torques shape: ", torques.shape)
        B, T = torques.shape[:2]
        assert B == batch_size
        if len(torques.shape) == 2:
            torques = torques.unsqueeze(1) 
        assert torques.shape[2:] == self.key_shape_map[torque_key] # 3
        # print(f"torque T: {T}") 

        tactile = torch.cat([forces, torques], dim=-1) # Shape: B, T, 6

        vtt_output = self.vtt_model(images, tactile) 

        # print(f"VTT output shape: {vtt_output.shape}") # Shape: B, T, n_compress_emb

        # process low dim input
        lowdim_embeddings = []
        for key in self.lowdim_keys:
            data = obs_dict[key] # Shape: B, T, D
            B, T = data.shape[:2]
            # print(f"{key} T: {T}")
            assert B == batch_size
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            else:
                data = data.reshape(B, T, -1)
            assert data.shape[2:] == self.key_shape_map[key]
            emb = self.lowdim_projections[key](data) 
            # print(f"{key} embedding shape: {emb.shape}") # Shape: B, T, n_compress_emb
            lowdim_embeddings.append(emb)

        # concatenate all features along t
        embeddings = [vtt_output] + lowdim_embeddings
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device) # (B, T, D)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape


class VTT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        img_patch_size: int = 14,
        tactile_patches: int = 1,
        in_channel: int = 3,
        embed_dim: int = 384,
        n_compress_emb: int = 288,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Visuo-Tactile Transformer
        """

        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            img_patch_size=img_patch_size,
            tactile_patches=tactile_patches,
            in_channel=in_channel,
            embed_dim=embed_dim,
        )
        img_patches = self.patch_embed.img_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_patches + self.patch_embed.tactile_patches, embed_dim)
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # dropout probabilities for each block
        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)  # apply layer normalization

        # reduce dimensionality of patches(384 -> 96 -> 32)
        self.compress_patches = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim // 4, embed_dim // 12),
        )
        # reduce dimensionality of patches(258*32 -> 640 -> 288)
        self.compress_layer = nn.Sequential(
            nn.Linear(
                (img_patches + self.patch_embed.tactile_patches) * embed_dim // 12, 640
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(640, n_compress_emb),
        )

        # initialize parameters with truncated normal distribution to stabilize training
        trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Ensure that the positional encoding is compatible with the input tensor
        """
        npatch = x.shape[2] - 1  # number of patches
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError("Position Encoder does not match dimension")

    def prepare_tokens(self, x: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        """
        Prepare tokens for the transformer
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  #  [B, 1, nc, w, h]
        B, T, nc, w, h = x.shape
        x, patched_tactile = self.patch_embed(x, tactile)  # patching image and tactile
        x = torch.cat((x, patched_tactile), dim=2)  # concatenate image and tactile
        x = x + self.interpolate_pos_encoding(x, w, h)  # add positional encoding
        return x

    def forward(self, x: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        x = self.prepare_tokens(x, tactile)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        img_tactile = self.compress_patches(x)  # reduce dimensionality of patches
        B, T, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, T, -1)  # flatten patches
        img_tactile = self.compress_layer(
            img_tactile
        )  # reduce dimensionality of patches
        return img_tactile


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5  # scale factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, C = (
            x.shape
        )  # B: batch size, T: sequence length, N: number of patches, C: channels
        qkv = (
            self.qkv(x)
            .reshape(B * T, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B*T, num_heads, N, C//num_heads)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, N, C)
        attn = attn.view(B, T, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        tactile_dim: int = 6,
        img_patch_size: int = 14,
        tactile_patches: int = 2,
        in_channel: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_patches = int(
            (img_size / img_patch_size) * (img_size / img_patch_size)
        )  # 16 * 16 = 256
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channel, embed_dim, kernel_size=img_patch_size, stride=img_patch_size
        )
        self.tactile_patches = tactile_patches
        self.decode_tactile = nn.Sequential(
            nn.Linear(tactile_dim, self.tactile_patches * embed_dim)
        )

    def forward(
        self, image: torch.Tensor, tactile: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = image.shape
        image = image.view(B * T, C, H, W)
        patched_image = self.proj(image)
        patched_image = (
            patched_image.flatten(2).transpose(1, 2).view(B, T, -1, self.embed_dim)
        )

        tactile = tactile.view(B * T, -1)
        decoded_tactile = self.decode_tactile(tactile).view(
            B, T, self.tactile_patches, -1
        )
        return patched_image, decoded_tactile


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.MLP(x)
        return x


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def test_vtt_forward():
    batch_size = 1
    sequence_length = 2
    channels = 3
    img_height = 224
    img_width = 224
    tactile_dim = 6
    lowvar_dim = 10 # 3 + 6 + 1

    obs_dict = {
        'camera0_rgb': torch.randn(batch_size, sequence_length, channels, img_height, img_width),
        'robot0_force': torch.randn(batch_size, sequence_length, 3),
        'robot0_torque': torch.randn(batch_size, sequence_length, 3),
        'robot0_eef_pos': torch.randn(batch_size, sequence_length, 3),
        'robot0_eef_rot_axis_angle': torch.randn(batch_size, sequence_length, 6),
        'robot0_gripper_width': torch.randn(batch_size, sequence_length, 1),
    }

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

    encoder = VTTObsEncoder(shape_meta=shape_meta)
    embeddings = encoder(obs_dict)

    print("Embeddings shape:", embeddings.shape)
    print("test done!")


# if __name__ == "__main__":
#     test_vtt_forward()
