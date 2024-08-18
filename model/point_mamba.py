from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from torch.nn.functional import softplus, gelu

import copy
import torch.nn.functional as F
import flash_attn

from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# from pointcept.models.utils.sds_conv import SpDepthWSepaConv3d,GroupSparseConv3d

# cross selective scan ===============================
import selective_scan_cuda
class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows

        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
        #     u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        #     # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
        # )
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

def compute_padding(offsets,patch_size = None):
    if patch_size is not None:
        bincounts = offset2bincount(offset=offsets).to(offsets.device)
        max_bin = ((max(bincounts)//patch_size) + 1) * patch_size
        pad_sizes = [max_bin - bincount for bincount in bincounts]
        padded_tensor = []
        new_offsets = torch.nn.functional.pad(offsets, (1, 0)).to(offsets.device)
        for i, bincount in enumerate(bincounts):
            padded_tensor.append(torch.arange(max_bin).to(offsets.device) % (bincount)+new_offsets[i]) 
        return padded_tensor, pad_sizes,max_bin
    else:
        bincounts = offset2bincount(offset=offsets).to(offsets.device)
        max_bin = max(bincounts).to(offsets.device)
        pad_sizes = [max_bin - bincount for bincount in bincounts]
        padded_tensor = []
        new_offsets = torch.nn.functional.pad(offsets, (1, 0)).to(offsets.device)
        for i, bincount in enumerate(bincounts):
            padded_tensor.append(torch.arange(max_bin).to(offsets.device) % (bincount)+new_offsets[i]) 
        return padded_tensor, pad_sizes,max_bin

def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    y_proj_weight: torch.Tensor=None,
    y_proj_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    conv1d: torch.nn.Module=None,
    softmax_version=False,
    order: torch.Tensor=None,
    inverse: torch.Tensor=None,
    offsets: torch.Tensor=None,
    nrows = -1,
    delta_softplus = True,
    patch_size = None,
    enable_inversion = False,
):
    
    _, N = A_logs.shape
    K, D, R = dt_projs_weight.shape

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    # xs = CrossScan.apply(x)
    padded_tensor, _ , max_bin = compute_padding(offsets=offsets,patch_size=patch_size)
    padded_tensor = torch.cat(padded_tensor)
    xs = torch.stack([x[:,idx][:,padded_tensor] for idx in order], dim=0)#(k,d,l_pad)
    xs = torch.stack(xs.chunk(len(offsets), dim=-1),dim=0)#(b,k,d,max_bin)
    if enable_inversion:
        reversed_x = torch.flip(xs, dims=[-1])
        xs = torch.cat([xs,reversed_x],dim=1)

    B,_,_,_ = xs.shape
    if patch_size is not None:
        xs = xs.view(-1, K, D,patch_size)
    _,K,D,L = xs.shape
    last_dim_size = xs.size(-1)

    if patch_size is None:
        assert last_dim_size == max_bin# 确保能够均匀切分
    original_index = [torch.argsort(idx) for idx in order]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    if patch_size is not None:
        xs = xs.view(-1, D * K, patch_size).to(torch.float)
        dts = dts.contiguous().view(-1, D * K, patch_size).to(torch.float)
    else:
        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if conv1d is not None:
        xs = conv1d(xs)
        
    # to enable fvcore.nn.jit_analysis: inputs[i].debugName
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, max_bin)

    bincounts = offset2bincount(offset=offsets)
    new_ys = []
    for i,bin in enumerate(bincounts):
        new_ys.append(ys[i,:,:,:bin])
    ys = torch.cat(new_ys,dim=-1)#(k,-1,l)  

    if y_proj_weight is not None:
        y =  torch.einsum("k d l, k r d ->r d l ", ys, y_proj_weight).squeeze(0)
    else:
        for k in range(K):
            if k == 0:
                y = ys[k][:,inverse[k]]
            elif k< 4:
                y += ys[k][:,inverse[k]]
            else:
                reversed_y = torch.flip(ys[k], dims=[-1])
                y += reversed_y[:,inverse[k%4]]

    if softmax_version:
        y = y.softmax(y, dim=-1).to(x.dtype)
        y = y.transpose(dim0=1, dim1=0).contiguous()
    else:
        y = y.transpose(dim0=1, dim1=0).contiguous()
        y = out_norm(y).to(x.dtype)
    
    return y

class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,#channel
        d_state=16,#隐藏维度
        ssm_ratio=2,
        dt_rank="auto",
        K = 1,
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv = False, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        dwconv_key = None,
        patch_size = None,
        enable_inversion = False,
        # ======================
        **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)#维度D
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.patch_size = patch_size
        self.rmsnorm = RMSNorm(self.d_model)

        # x proj; dt proj ============================
        self.enable_inversion = enable_inversion
        self.K = K
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        self.y_proj = [
            nn.Linear(self.d_inner, 1, **factory_kwargs)
            for _ in range(self.K)
        ]
        # self.y_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.y_proj], dim=0)) 
        self.y_proj_weight = None
        # self.y_proj_bias = nn.Parameter(torch.stack([t.bias for t in self.y_proj], dim=0)) 
        del self.y_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K 
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        enable_conv1d = True
        if enable_conv1d:
            kernel_size = 3
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner * self.K,
                out_channels=self.d_inner * self.K,
                bias=conv_bias,
                kernel_size=kernel_size,
                groups=self.d_inner,
                padding=kernel_size - 2,
                **factory_kwargs,
            )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

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
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
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
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
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
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, order: torch.Tensor,inverse: torch.Tensor,offsets: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,self.y_proj_weight,None,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), getattr(self, "conv1d", None), self.softmax_version, 
            order,inverse, offsets,nrows = nrows, patch_size = self.patch_size,enable_inversion= self.enable_inversion
        )
    
    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2
    # forward_core = forward_corev0

    def forward(self, point: Point, **kwargs):
        shortcut = point.feat
        point.feat = self.rmsnorm(point.feat)
        xz = self.in_proj(point.feat)
        order = point.serialized_order[:self.K]
        serialized_inverse = point.serialized_inverse[:self.K]
        offsets = point.offset
        
        x, z = xz.chunk(2, dim=-1) # (b, L, d)(l,d)
        x = x.permute(1,0).contiguous()
        # print(offsets)
        y = self.forward_core(x,order,serialized_inverse,offsets)
        if self.softmax_version:
            y = y * z
        else:
            y = y * F.gelu(z)
        out = self.out_proj(y)
        point.feat = shortcut + out
        return point

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The dimension of the input tensor.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        g (nn.Parameter): The learnable parameter used for scaling.

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.

        """
        return F.normalize(x, dim=-1) * self.scale * self.g
    
class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out
    
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GatedMLP(nn.Module):
    def __init__(self, dim=1024, expansion_factor=1):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = nn.Linear(dim, 2 * hidden, bias=False)
        self.shrink = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = gelu(gate) * x
        out = self.shrink(x)
        out = out + shortcut
        return out

class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point



class Block(nn.Module):
    def __init__(
        self,
        channels,
        K,
        atte_num_heads = None,
        drop_path=0.0,
        norm_layer=RMSNorm,
        pre_norm=True,
        cpe_indice_key=None,
        mamba_conv = True,
        patch_size = None,
        enable_inversion = False,
        
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.enable_lib = False
        self.cpe = nn.Sequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.pre_norm_layer = nn.Sequential(norm_layer(channels))
        self.norm1 = nn.Sequential(norm_layer(channels))
        self.norm2 = nn.Sequential(norm_layer(channels))
        atte_patch = 1024
        self.attn = SerializedAttention(
                    channels=channels,
                    patch_size=atte_patch,
                    num_heads=atte_num_heads,
                    enable_flash=True,
                    upcast_attention=False,
                    upcast_softmax = False,
                )
        
        # self.gatemlp = GatedMLP(dim = channels)
        self.mamba = SS2D(
            d_model=channels,
            K = K,
            dwconv_key = cpe_indice_key+'dwconv',
            d_conv = mamba_conv,
            patch_size = patch_size,
            enable_inversion = enable_inversion,
        )
        # self.mamba = Mamba(d_model=channels)
        self.gatemlp = GatedMLP(dim = channels)

        self.drop_path = nn.Sequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.pre_norm_layer(point)
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat

        point.feat = self.gatemlp(point.feat)

        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mamba(point))
        # point.feat = self.drop_path(self.mamba(point.feat.unsqueeze(0)).squeeze(0))
        point.feat = shortcut + point.feat


        # point.feat = self.gatemlp(point.feat)

        
        # if not self.pre_norm:
        #     point = self.norm1(point)

        # shortcut = point.feat
        # if self.pre_norm:
        #     point = self.norm2(point)
        # point.feat = self.gatemlp(point.feat)
        # point.feat = shortcut + point.feat
        # if not self.pre_norm:
        #     point = self.norm2(point)
        point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        enable_mamba = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = nn.Sequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = nn.Sequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point



class SerializedUnpooling(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent




class Embedding(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = nn.Sequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

class PointMambaRe(nn.Module):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z_trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 1, 1, 1),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(1, 1, 1, 1),
        dec_num_head=(4, 4, 8, 16),
        dec_channels=(64, 64, 128, 256),
        mamba_conv = True,
        patch_size=None,
        enable_inversion = False,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        cls_mode=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1


        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = nn.Sequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = nn.Sequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        K = 1,
                        atte_num_heads = enc_num_head[s],
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        pre_norm=pre_norm,
                        cpe_indice_key=f"stage{s}",
                        patch_size= patch_size,
                        enable_inversion = enable_inversion,
                    ),
                    name=f"block{i}_enc",
                )
                    
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = nn.Sequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = nn.Sequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            K = 1,
                            atte_num_heads = dec_num_head[s],
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            pre_norm=pre_norm,
                            cpe_indice_key=f"stage{s}",
                            mamba_conv=mamba_conv,
                            patch_size=patch_size,
                            enable_inversion = enable_inversion,
                        ),
                        name=f"block{i}_dec",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, feats, xyz, offset, batch, neighbor_idx, gt, query_base_y=None):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point.seg_label['stage0'] = point.segment

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
