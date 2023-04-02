# By Hanrong Ye for TaskPrompter
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on Swin Transformer

INTERPOLATE_MODE = 'bilinear'
import logging
import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import overlay_external_default_cfg, build_model_with_cfg
from timm.models.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import  _init_vit_weights
# from .timm_helpers import build_model_with_cfg

_logger = logging.getLogger(__name__)

from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x

BatchNorm2d = nn.SyncBatchNorm

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'swin_base_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    ),

    'swin_large_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_large_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    ),

    'swin_small_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    ),

    'swin_tiny_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    ),

    'swin_base_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_base_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),

    'swin_large_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_large_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),

}

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, p, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.p = p
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, spa_prompts, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        nW = B_ // spa_prompts.shape[0]
        prompts_len = self.p.prompts_len
        spa_prompts = spa_prompts[:, None, :, :].expand(-1, nW, -1, -1).clone().reshape(-1, prompts_len, self.dim)

        x = torch.cat([spa_prompts, x], dim=1)

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn_weight = (q @ k.transpose(-2, -1)) # (B*nW, nH, N, N), N=nPrompts+Wh*Ww
        attn = attn_weight * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn[:, :, self.p.prompts_len:, self.p.prompts_len:] = attn[:, :, self.p.prompts_len:, self.p.prompts_len:] + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) #+ mask.unsqueeze(1).unsqueeze(0)
            attn[:,:,:,self.p.prompts_len:, self.p.prompts_len:] = attn[:,:,:,self.p.prompts_len:, self.p.prompts_len:] + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        spa_prompts, x = sep_prompt(x, self.p.prompts_len) #  spa_prompts: (B*nW, T, C)
        spa_prompts = spa_prompts.reshape(B_ // nW, nW, prompts_len, C).mean(dim=1)

        return x, attn_weight, spa_prompts


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, LAST_BLOCK_FLAG, p, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.LAST_BLOCK_FLAG = LAST_BLOCK_FLAG
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(p,
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # pad to multiple of window size
        H, W = self.input_resolution
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        H += pad_b
        W += pad_r
        if pad_r == 0 and pad_b == 0:
            self.pad_size = None
        else:
            self.pad_size = (pad_b, pad_r)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.p = p
        self.resolution = [int(np.sqrt(p.chan_embed_dim)), int(np.sqrt(p.chan_embed_dim))]
        # task prompts processing
        pixel_no = int(input_resolution[0] * input_resolution[1])
        # channel-wise attention
        self.chan_nheads = p.chan_nheads
        self.attn_drop = nn.Dropout(0)
        self.chan_q = nn.Linear(p.chan_embed_dim, p.chan_embed_dim, bias=qkv_bias)
        self.chan_kv = nn.Linear(pixel_no, p.chan_embed_dim*2, bias=qkv_bias)
        self.chan_scale = p.chan_embed_dim ** -0.5
        self.token_trans = nn.Linear(dim, p.chan_embed_dim)
        if not LAST_BLOCK_FLAG:
            self.chan_proj = nn.Linear(p.chan_embed_dim, p.chan_embed_dim)
            self.token_trans1 = nn.Linear(p.chan_embed_dim, dim)


    def forward(self, x, task_prompts):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # task prompts norm
        ori_task_prompts = task_prompts
        spa_prompts = self.norm1(task_prompts)
        chan_prompts = self.token_trans(task_prompts)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad to the multiple of window size
        OH, OW = H, W
        if self.pad_size:
            pad_l, pad_t, pad_b, pad_r = 0, 0, self.pad_size[0], self.pad_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            H = OH + pad_b
            W = OW + pad_r

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn_weight, spa_prompts = self.attn(x_windows, spa_prompts, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        task_prompts = spa_prompts

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reshaping of attn_weight
        # attn_weight has shape (B*nW, nheads, T+wh*ww, T+wh*ww), N=T+HW
        attn_weight = attn_weight[:, :, :self.p.prompts_len, self.p.prompts_len:] # (B*nW, nheads, T, wh*ww)
        attn_weight = rearrange(attn_weight, '(b nWh nWw) nheads t (Wh Ww) -> b nheads t (nWh Wh) (nWw Ww)', b=B, nWh=H//self.window_size, nWw=W//self.window_size, Wh=self.window_size, Ww=self.window_size) # Wh: window height; nWh: number of window along height

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            attn_weight = torch.roll(attn_weight, shifts=(self.shift_size, self.shift_size), dims=(3, 4))
        else:
            x = shifted_x

        # un-pad
        if self.pad_size:
            x = x[:, :OH, :OW, :].contiguous()
            attn_weight = attn_weight[:, :, :, :OH, :OW].contiguous() # （B, nheads, T, OH, OW)

        
        x = x.view(B, OH * OW, C)

        raw_spa_attn = attn_weight
        # channel-wise attention
        chan_x = x
        chan_x = chan_x.permute(0,2,1) # (B, C, HxW)
        q = self.chan_q(chan_prompts) # (B, T, chant_dim)
        kv = self.chan_kv(chan_x).reshape(B, C, 2, -1) # (B, C, 2, chant_dim)
        k, v = kv[:,:,0,:], kv[:,:,1,:] # (B, C, chant_dim)

        nh = nw = int(np.sqrt(self.chan_nheads))
        win_h = self.resolution[0] // nh
        win_w = self.resolution[1] // nw
        q = rearrange(q, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)
        k = rearrange(k, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)
        v = rearrange(v, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)

        raw_chan_attn = (q @ k.transpose(-2, -1)) # (B, T, C)
        attn = raw_chan_attn * self.chan_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        chan_x = (attn @ v) # (B, T, chant_dim)
        chan_x = rearrange(chan_x, 'b (nh nw) t (h w) -> b t (nh h nw w)', nh=nh, nw=nw, h=win_h, w=win_w)
        raw_chan_attn = rearrange(raw_chan_attn, 'b (nh nw) t c -> b t c nh nw', nh=nh, nw=nw)

        raw_attn = [raw_spa_attn, raw_chan_attn]

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if not self.LAST_BLOCK_FLAG:
            chan_prompts = self.chan_proj(chan_x)
            task_prompts += self.token_trans1(chan_prompts)
            task_prompts = ori_task_prompts + self.drop_path(task_prompts)
            task_prompts = task_prompts + self.drop_path(self.mlp(self.norm2(task_prompts)))

        return x, raw_attn, task_prompts


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, p, num_heads, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.p = p
        task_no = len(p.TASKS.NAMES)

        self.process_chan_attn = nn.Linear(dim, 2*dim, bias=False)
        self.task_prompts_up = nn.Linear(dim, 2*dim, bias=False)
        self.spa_attn_ds = nn.Conv2d(num_heads*task_no, num_heads*task_no, kernel_size=3, padding=1, stride=2)

    def forward(self, x, task_prompts, attn_weight):
        """
        x: B, H*W, C
        attn_weight: B, nheads, T, H, W
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        # downsample attn
        raw_spa_attn, raw_chan_attn = attn_weight

        assert raw_spa_attn.shape[-2] == H
        assert raw_spa_attn.shape[-1] == W
        _, nheads, T, _, _ = raw_spa_attn.shape
        raw_spa_attn = raw_spa_attn.reshape(B, -1, H, W)
        raw_spa_attn = self.spa_attn_ds(raw_spa_attn)
        raw_spa_attn = raw_spa_attn.reshape(B, nheads, T, H//2, W//2)

        # process channel attention
        # raw_chan_attn: (b t c nh nw)
        raw_chan_attn = raw_chan_attn.transpose(2,-1)
        raw_chan_attn = self.process_chan_attn(raw_chan_attn)
        raw_chan_attn = raw_chan_attn.transpose(2,-1)
        attn_weight = [raw_spa_attn, raw_chan_attn]

        # downsample task_prompts
        task_prompts = self.task_prompts_up(task_prompts)

        return x, task_prompts, attn_weight

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, LAST_LAYER_FLAG, p, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(LAST_BLOCK_FLAG=True if i==depth-1 and LAST_LAYER_FLAG==True else False, p=p,
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(p, num_heads, input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, task_prompts):
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn_weight, task_prompts = blk(x, task_prompts)
        x_bef_down = x
        if self.downsample is not None:
            x, task_prompts, attn_weight = self.downsample(x_bef_down, task_prompts, attn_weight)
        return x, attn_weight, task_prompts

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TaskPrompterSwin(nn.Module):
    """ TaskPrompter built upon Swin Transformer
    """

    def __init__(self, p=None, img_size=224, patch_size=4, in_chans=3, 
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, weight_init='', **kwargs):
        """
        Args:
            p (dcit): parameters
            img_size (int | tuple(int)): Input image size. Default 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: 3
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        """
        super().__init__()

        self.p = p
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 96*8=768
        self.mlp_ratio = mlp_ratio

        # feature resolution of different scales
        self.resolution = []
        self.img_ds_ratio = self.p.img_ds_ratio
        for spa_dim in self.p.ori_spatial_dim:
            self.resolution.append([int(spa_dim[0]*self.img_ds_ratio), int(spa_dim[1]*self.img_ds_ratio)])
        img_size = [int(_*self.img_ds_ratio) for _ in img_size]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # multi-task prompt learning
        task_no = len(p.TASKS.NAMES)
        self.task_no = task_no
        self.all_tasks = p.TASKS.NAMES
        self.prompt_len = p.prompt_len
        self.prompts_len = task_no*p.prompt_len
        p.prompts_len = self.prompts_len
        self.task_prompts = nn.Parameter(torch.ones(self.prompts_len, embed_dim))
        trunc_normal_(self.task_prompts, mean=1., std=1.)

        self.fea_fuse = nn.ModuleList()
        self.fea_decode_spa = nn.ModuleList()
        self.fea_decode_chan = nn.ModuleList()
        for i_layer in range(self.num_layers):
            cur_embed_dim = p.backbone_channels[i_layer]
            tar_dim = p.backbone_channels[i_layer]
            final_embed_dim = p.backbone_channels[i_layer]
            prompt_dim = num_heads[i_layer]*p.prompt_len
            self.fea_fuse.append(nn.ModuleDict())
            self.fea_decode_spa.append(nn.ModuleDict())
            self.fea_decode_chan.append(nn.ModuleDict())
            for task in p.TASKS.NAMES:
                self.fea_fuse[i_layer][task] = nn.Sequential(nn.Conv2d(tar_dim*2, final_embed_dim, kernel_size=1), nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1), BatchNorm2d(final_embed_dim), nn.GELU(),  nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1))
                self.fea_decode_spa[i_layer][task] = nn.Sequential(nn.Conv2d(cur_embed_dim, tar_dim, kernel_size=1, padding=0))
                self.fea_decode_chan[i_layer][task] = nn.Sequential(nn.Conv2d(cur_embed_dim, tar_dim, kernel_size=1, padding=0))

        self.multi_scale_fuse = nn.ModuleDict({t: nn.Conv2d(sum(p.backbone_channels), p.final_embed_dim, kernel_size=3, padding=1) for t in p.TASKS.NAMES if t!='3ddet'})

        # build layers
        layers = []
        for i_layer in range(self.num_layers):
            layers += [BasicLayer(True if i_layer==self.num_layers-1 else False, p,
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patch_grid[0] // (2 ** i_layer), self.patch_grid[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            ]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = ''
        if weight_init.startswith('jax'):
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        # resize input image
        if not self.img_ds_ratio==1:
            x = F.interpolate(x, scale_factor=self.img_ds_ratio, mode='bilinear', align_corners=False)

        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # get task prompts
        task_prompts = self.task_prompts[None].expand(x.shape[0], -1, -1)

        task_fea = {task: [] for task in self.all_tasks}
        info = {} # pass information through the pipeline

        layers_no = len(self.layers)
        for il, _lay in enumerate(self.layers):
            x, attn_weight, task_prompts = _lay(x, task_prompts)
            if il < layers_no-1:
                spa_x = x 
                _cur_task_fea, info = self.cal_task_feature(spa_x, attn_weight, il, info)
                for t_idx, task in enumerate(self.p.TASKS.NAMES):
                    task_fea[task].append(_cur_task_fea[task])
        x = self.norm(x)  # B L C

        il=3
        spa_x = x
        _cur_task_fea, info = self.cal_task_feature(spa_x, attn_weight, il, info)
        for t_idx, task in enumerate(self.p.TASKS.NAMES):
            task_fea[task].append(_cur_task_fea[task])

        # fuse multi-scale feature for 2D tasks
        new_task_fea = {}
        for task in self.p.TASKS.NAMES:
            if task == '3ddet':
                new_task_fea[task] = task_fea[task]
            else:
                target_scale = task_fea[task][0].shape[-2:]
                _task_fea = torch.cat([F.interpolate(_, target_scale, mode=INTERPOLATE_MODE) for _ in task_fea[task]], dim=1)
                _task_fea = self.multi_scale_fuse[task](_task_fea)
                new_task_fea[task] =  _task_fea

        return new_task_fea, info

    def cal_task_feature(self, x, attn_weight, il, info):
        ''' Calculate task feature at each stage
        '''
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.resolution[il][0], w=self.resolution[il][1])
        task_fea = {}
        spa_attn, chan_attn = attn_weight
        chan_task_fea = {}

        for t_idx, task in enumerate(self.p.TASKS.NAMES):
            # task feature extraction with spatial attention
            cur_attn_weight = spa_attn[:, :, t_idx*self.prompt_len:(t_idx+1)*self.prompt_len, :] # (b, nheads, prompt_len, h*w)
            cur_attn_weight = rearrange(cur_attn_weight, 'b nh np h w -> b (nh np) h w', h=self.resolution[il][0], w=self.resolution[il][1])

            bs, nheads = cur_attn_weight.shape[0:2]
            cur_task_fea = []
            head_channel_no = self.p.backbone_channels[il] // nheads
            for hea in range(nheads):
                cur_head_attn = cur_attn_weight[:, hea:hea+1, :, :]
                cur_task_fea.append(cur_head_attn * x[:, head_channel_no*hea:head_channel_no*(hea+1), :, :])
            cur_task_fea = torch.cat(cur_task_fea, dim=1) + x
            cur_task_fea = self.fea_decode_spa[il][task](cur_task_fea)
            task_fea[task] = cur_task_fea

            # task feature extraction with channel-wise attention
            # chan_attn: (b t c nh nw)
            cur_attn_weight = chan_attn[:, t_idx] # (b, c, nh, nw) 
            bs, _, nh, nw = cur_attn_weight.shape
            nheads = nh * nw
            win_h = self.resolution[il][0] // nh
            win_w = self.resolution[il][1] // nw
            cur_task_fea = []

            for h_idx in range(nh):
                cur_row = []
                for w_idx in range(nw):
                    _patch = x[:, :, h_idx*win_h:(h_idx+1)*win_h, w_idx*win_w:(w_idx+1)*win_w]
                    _attn = cur_attn_weight[:, :, h_idx, w_idx]
                    _attn = _attn.unsqueeze(-1).unsqueeze(-1)
                    cur_row.append(_attn * _patch)
                cur_task_fea.append(torch.cat(cur_row, dim=3))
            cur_task_fea = torch.cat(cur_task_fea, dim=2) + x
            cur_task_fea = self.fea_decode_chan[il][task](cur_task_fea)
            chan_task_fea[task] = cur_task_fea

            combined_fea = torch.cat([task_fea[task], chan_task_fea[task]], dim=1)

            combined_fea = self.fea_fuse[il][task](combined_fea)
            task_fea[task] = combined_fea

        return task_fea, info



def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        
        # remove attn_map
        if 'attn_mask' in k:
            continue 
        out_dict[k] = v

    return out_dict

def taskprompter_create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)

    model = build_model_with_cfg(
        TaskPrompterSwin, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

def taskprompter_swin_base_patch4_window12_384(pretrained=False, **kwargs):
    """ TaskPrompter based on Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return taskprompter_create_swin_transformer('swin_base_patch4_window12_384', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
