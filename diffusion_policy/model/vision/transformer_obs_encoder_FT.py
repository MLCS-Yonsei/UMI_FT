from typing import Optional, Tuple
import torch
import warnings
import math
from torch.nn import functional as F
from torch.distributions import Normal

import torch.nn as nn


class VTT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        img_patch_size: int = 14,
        tactile_patches: int = 2,
        in_channel: int = 3,
        embed_dim: int = 384,
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
            nn.Linear(640, 288),
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
        B, S, nc, w, h = x.shape
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
        B, S, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, S, -1)  # flatten patches
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
        B, S, N, C = (
            x.shape
        )  # B: batch size, S: sequence length, N: number of patches, C: channels
        qkv = (
            self.qkv(x)
            .reshape(B * S, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # (3, B*S, num_heads, N, C//num_heads)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
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
        B, S, C, H, W = image.shape
        image = image.view(B * S, C, H, W)
        patched_image = self.proj(image)
        patched_image = (
            patched_image.flatten(2).transpose(1, 2).view(B, S, -1, self.embed_dim)
        )

        tactile = tactile.view(B * S, -1)
        decoded_tactile = self.decode_tactile(tactile).view(
            B, S, self.tactile_patches, -1
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
    sequence_length = 5
    channels = 3
    img_height = 224
    img_width = 224
    tactile_dim = 6

    synthetic_images = torch.randn(
        batch_size, sequence_length, channels, img_height, img_width
    )

    synthetic_tactile = torch.randn(batch_size, sequence_length, tactile_dim)

    encoder = VTT()

    img_tactile = encoder(synthetic_images, synthetic_tactile)

    print("img_tactile feature shape:", img_tactile.shape)
    print("test done!")


if __name__ == "__main__":
    test_vtt_forward()
