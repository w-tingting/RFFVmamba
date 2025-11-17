import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn

from layers import Resnet, ConvBlock


in_channels = [0]


def _5d_to_3d(x, use_reshape=False):
    n, mc, C, H, W = x.shape
    if use_reshape:
        x = x.reshape(n, mc * C, H, W)
        x = x.reshape(n, mc * C, H * W)
    else:
        x = x.view(n, mc * C, H, W)
        x = x.view(n, mc * C, H * W)
    return x


class RFF_Token_Embed(nn.Module):
    def __init__(self, mc, kernel_type, block_conf, image_size=56, stage=1):
        super(RFF_Token_Embed, self).__init__()
        self.mc = mc
        self.kernel_type = kernel_type
        self.block_conf = block_conf
        # print("block_conf in RFF_Token_Embed:", self.block_conf)
        self.stage = stage
        

        self.res_normal = Resnet()
        # self.res_down = Resnet(stride=2)

        self.blocks = nn.ModuleList()
        self.res = nn.ModuleList()

        # in_channels = [0]
        for i, bc in enumerate(block_conf):

            padding = bc[1] // 2
            self.grid_size = torch.tensor((image_size - bc[1] + 2 * padding) / bc[2] + 1).floor()
            image_size = self.grid_size

            if bc[2] >= 2:
                in_chanels = 3 if i == 0 else (in_channels[-2] + in_channels[-1]) * self.mc
                res_down = Resnet(in_chanel=in_chanels, k_size=bc[1], stride=bc[2])
            else:
                res_down = self.res_normal
            self.res.append(res_down)

            if stage == 1:
                b = ConvBlock(
                    in_channels=3 if i == 0 else in_channels[-2] + in_channels[-1],
                    out_channels=bc[0],
                    k_size=bc[1],
                    stride=bc[2],
                    mc=self.mc,
                    kernel_type=self.kernel_type,
                    group=1,
                    head=i == 0
                )
            else:
                b = ConvBlock(
                    in_channels=in_channels[-3] + in_channels[-2] + in_channels[-1] if i == 1 else in_channels[-2] +
                                                                                                   in_channels[-1],
                    out_channels=bc[0],
                    k_size=bc[1],
                    stride=bc[2],
                    mc=self.mc,
                    kernel_type=self.kernel_type,
                    group=1,
                    head=False
                )

            self.blocks.append(b)
            in_channels.append(bc[0])

        if stage == 1:
            self.layer_norm = torch.nn.LayerNorm((in_channels[-2] + in_channels[-1]) * self.mc, eps=1e-5)
        else:
            self.layer_norm = torch.nn.LayerNorm((in_channels[-2] + in_channels[-1]) * self.mc, eps=1e-5)
        # EXCEPT FULLY RESNET
        # in_channels.append(0)

        # num_pathces
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x):
        # x : [n, mc, c, h, w]
        
        layer_results = []
        middle_feat = []
        # New Resnet
        if self.stage != 1:
            layer_results.append(x)

        for b in self.blocks:
            x = b(x)

            layer_results.append(x)
            if self.stage == 1:
                if len(layer_results) >= 2:
                    x = self.res[len(layer_results) - 1](layer_results[-2], x)
            else:
                x = self.res[len(layer_results) - 2](layer_results[-2], x)
            middle_feat.append(x)
        bs, mc, C, H, W = x.shape
        
        x = _5d_to_3d(x).transpose(1, 2)
        # Layer Norm
        x = self.layer_norm(x)
        return x, middle_feat, bs, mc, C, H, W

# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim) # 最后一个维度规范化

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) # dim=-1: (0,0)  dim=-2:(0,1) dim=-3:(0,1)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class _SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.proj = (nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        x_proj = self.proj(x)
        return x_proj * scale


class MiddleFeatureCrossAttention(nn.Module):
    """Fuse middle-stage features via parallel 1x1 conv and SE branches, then concatenate."""

    def __init__(self, out_channels: int, reduction: int = 4):
        super().__init__()
        self.out_channels = out_channels
        self.reduction = reduction
        self.initialized = False
        self.expected_features: Optional[int] = None
        self.conv_branches: nn.ModuleList = nn.ModuleList()
        self.se_branches: nn.ModuleList = nn.ModuleList()

    def _initialize(self, middle_features):
        if self.initialized:
            return
        num_features = len(middle_features)
        if num_features == 0:
            raise ValueError("middle_features must contain at least one tensor for fusion")

        device = middle_features[-1].device
        dtype = middle_features[-1].dtype
        self.expected_features = num_features

        base_channels = self.out_channels // num_features
        channel_splits = [base_channels] * num_features
        remainder = self.out_channels - sum(channel_splits)
        if remainder > 0:
            channel_splits[-1] += remainder

        for feat, out_ch in zip(middle_features, channel_splits):
            _, mc, channels, _, _ = feat.shape
            in_ch = mc * channels
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            se = _SqueezeExcite(in_ch, out_ch, reduction=self.reduction)
            conv = conv.to(device=device, dtype=dtype)
            se = se.to(device=device, dtype=dtype)
            self.conv_branches.append(conv)
            self.se_branches.append(se)

        self.initialized = True

    def initialize_from_features(self, middle_features):
        if not self.initialized:
            detached = [feat.detach() for feat in middle_features]
            self._initialize(detached)

    def forward(self, middle_features):
        if not self.initialized:
            self._initialize(middle_features)

        if self.expected_features is not None and len(middle_features) != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} middle feature tensors but received {len(middle_features)}"
            )

        target_h, target_w = middle_features[-1].shape[-2:]
        fused = []
        for conv, se, feat in zip(self.conv_branches, self.se_branches, middle_features):
            B, mc, C, H, W = feat.shape
            feat_4d = feat.view(B, mc * C, H, W)
            conv_out = conv(feat_4d)
            se_out = se(feat_4d)
            combined = conv_out + se_out
            if (H, W) != (target_h, target_w):
                combined = F.interpolate(combined, size=(target_h, target_w), mode="bilinear", align_corners=False)
            fused.append(combined)

        return torch.cat(fused, dim=1)


class EfficientChannelAttention(nn.Module):
    """Lightweight Efficient Channel Attention (ECA).

    Uses global average pooling + a small 1D conv across channels to model
    local cross-channel interactions and produces a channel-wise gating map.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.channels = channels
        k = k_size if k_size % 2 == 1 else (k_size + 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Conv1d expects input shape (B, C_in, L) — we use (B, 1, C)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        y = self.gap(x).view(b, 1, c)  # (B,1,C)
        y = self.conv(y)               # (B,1,C)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class EnhancedSpatialAttention(nn.Module):
    """Simple Enhanced Spatial Attention (ESA) approximation.

    Reduces channels, applies spatial convs to compute an attention map and
    multiplies it with the input feature map.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1)
        self.conv = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.spatial = nn.Conv2d(mid, 1, kernel_size=1)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        y = self.reduce(x)
        y = self.act(y)
        y = self.conv(y)
        y = self.act(y)
        y = self.spatial(y)
        y = self.sigmoid(y)
        return x * y


class MultiScaleOutProj(nn.Module):
    """Multi-scale receptive-field output projection.

    - Accepts channel-first `(B, C, H, W)` or channel-last `(B, H, W, C)` inputs.
    - Applies several depthwise conv branches with increasing dilation (multi-scale),
      projects each branch with a pointwise conv, concatenates and fuses to produce
      the requested output channels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, channel_first: bool = True, branch_dilations=(1, 2, 4)):
        super().__init__()
        self.channel_first = channel_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_dilations = tuple(branch_dilations)

        # depthwise branches + 1x1 proj per branch
        self.branches = nn.ModuleList()
        for d in self.branch_dilations:
            padding = d
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding, dilation=d, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # fusion: reduce concatenated channels back to out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * len(self.branches), out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _to_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.channel_first else x.permute(0, 3, 1, 2).contiguous()

    def _to_channel_last(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.channel_first else x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (B, C, H, W) or (B, H, W, C)
        was_channel_first = self.channel_first
        x_cf = self._to_channel_first(x)

        outs = []
        for b in self.branches:
            outs.append(b(x_cf))

        x_cat = torch.cat(outs, dim=1)
        out = self.fuse(x_cat)

        if not was_channel_first:
            out = self._to_channel_last(out)

        return out


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2Dv2(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        init_kwargs = dict(kwargs)
        init_kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.forward_type = forward_type
        if forward_type in ["v0", "v0seq"]:
            raise ValueError("forward_type 'v0' variants are no longer supported; use v2-based modes instead.")
        self.__initv2__(**init_kwargs)

    # support: v01-v05; v051d,v052d,v052dc; 
    # postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
    # history support: v2,v3;v31d,v32d,v32dc;
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        print("forward:",forward_type)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        # use MultiScaleOutProj to provide multi-scale receptive fields
        self.out_proj = MultiScaleOutProj(
            in_channels=self.d_inner,
            out_channels=self.d_model,
            bias=bias,
            channel_first=self.channel_first,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), 
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value




# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = True,
        # =============================
    _SS2D: type = SS2Dv2,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0 # True
        self.mlp_branch = mlp_ratio > 0 # False
        self.use_checkpoint = use_checkpoint # False
        self.post_norm = post_norm #False

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim) # 96
            self.op = _SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

block_conf = [
    [[8, 3, 1],
     [16, 3, 1],
     [16, 3, 1]],
    [[32, 3, 1],
     [32, 3, 1],
     [64, 3, 2]],
    [[64, 3, 1]]
]

from layers import Conv_RFF, Conv_Linear, FullyConnected, Resnet, ConvBlock
class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=43, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], # 96
        # =========================
        mc=1, 
        kernel_type="arccos", 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, # 0.2
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v3", # "v1", "v2", "v3" : v1
        patchembed_version: str = "v2", # "v1", "v2" : v1
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
    _SS2D=SS2Dv2,
        # =========================
        **kwargs,
    ):
        super().__init__()
        
        self.mc = mc
        self.kernel_type = kernel_type
        self.block_conf = block_conf
        
        
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"]) # True ln2d
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)] # [96,192,384,768]
       
        self.num_features = dims[-1]
        # self.num_features = 32
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule sum(depths)=2+2+9+2=15
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None) # ln
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None) # silu
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None) # gelu

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None # None

        # patch embedding
        # _make_patch_embed = dict(
        #     v1=self._make_patch_embed, 
        #     v2=self._make_patch_embed_v2,
        # ).get(patchembed_version, None)
        # self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        self.patch_embed = RFF_Token_Embed(self.mc, self.kernel_type, self.block_conf[0], stage=1)
        patch_merge_in = math.prod(self.patch_embed.layer_norm.normalized_shape)
        self.patch_merge = nn.Conv2d(patch_merge_in, self.dims[0], kernel_size=1)
        # two lightweight attention branches applied after patch merging
        if self.dims[0] % 2 != 0:
            raise ValueError("Embedding dimension must be even to split across ECA/ESA branches.")
        # self.branch_channels = self.dims[0] // 2
        self.branch_channels = self.dims[0]
        self.eca = EfficientChannelAttention(self.branch_channels)
        self.esa = EnhancedSpatialAttention(self.branch_channels)
        self.patch_residual_act = nn.SiLU()
        self.middle_cross_attn = MiddleFeatureCrossAttention(self.dims[0])

        with torch.no_grad():
            dummy_height = dummy_width = imgsize
            dummy = torch.zeros(
                1,
                in_chans,
                dummy_height,
                dummy_width,
                device=self.patch_merge.weight.device,
                dtype=self.patch_merge.weight.dtype,
            )
            _, dummy_middle_features, *_ = self.patch_embed(dummy)
            self.middle_cross_attn.initialize_from_features(dummy_middle_features)
            del dummy_middle_features

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))
            
        self.head_out_channels = 64
        # self.head_out_channels = 32
        kernel_type_lower = self.kernel_type.lower() if isinstance(self.kernel_type, str) else "rbf"
        # self.rff_feature_dim = self.head_out_channels * 16 * 16
        self.rff_feature_dim = self.head_out_channels * 4 * 4
        self.rff_layer_norm = torch.nn.LayerNorm(self.rff_feature_dim, eps=1e-5)
        self.head = Conv_RFF(
            self.num_features,
            self.head_out_channels,
            1,
            1,
            0,
            self.mc,
            self.kernel_type,
            1,
            False,
        )
        
        self.fully = FullyConnected(self.rff_feature_dim, self.num_classes, self.mc, False)

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        _SS2D=SS2Dv2,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        
        x, middle_features, batch_size, mc, channels, height, width = self.patch_embed(x)
        
        x = x.contiguous().view(batch_size, height, width, mc, channels)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, mc * channels, height, width)
        patch_merged = self.patch_merge(x)
        x = patch_merged
        
        # attn_feature = self.middle_cross_attn(middle_features)
        # if attn_feature.shape[-2:] != patch_merged.shape[-2:]:
        #     attn_feature = F.interpolate(
        #         attn_feature,
        #         size=patch_merged.shape[-2:],
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        
             
        # # # 98.77%
        # residual = self.patch_residual_act(patch_merged)
        # # pass merged patches through two attention branches and sum their outputs
        # eca_out = self.eca(attn_feature)
        # esa_out = self.esa(attn_feature)
        # x = eca_out + esa_out + residual
        
      
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        
        for layer in self.layers:
            x = layer(x)
  
        feat = x.unsqueeze(1).repeat(1, self.mc, 1, 1, 1)
         
        head_out = self.head(feat)
       
        bs = head_out.shape[0]
        head_out = head_out.reshape(bs, self.mc, -1)
        head_out = head_out.transpose(0, 1).contiguous() # (mc, bs, feat_dim)
        
        head_out = self.rff_layer_norm(head_out) # 5,64,256
        logits_mc = self.fully(head_out) # 5,64,43
        # classification head
        attn = F.softmax(logits_mc.mean(dim=(1)),dim=0).mean(dim=1)# 43
        attn = attn / attn.sum()
        x = torch.sum(logits_mc * attn.view(-1,1,1),dim=0)
        
        # x = logits_mc.mean(dim=0)
        
        # x = self.classifier(x)
        return x

    def flops(self, shape=(3, 32, 32), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


