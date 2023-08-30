import torch
import torch as th
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as Func
EPSILON = th.finfo(th.float32).eps
from typing import Optional


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """
    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, locations=[None, None], dropout=0.0, bias=True,
                 batch_first=True):
        super(SelfAttention, self).__init__()
        self.name = 'SelfAttention'
        self.locations = locations
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, output_dim)
        self.lnorm2 = nn.LayerNorm(output_dim)

    def forward(self, q, k, v):
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)
        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = self.lnorm1((v + attn_output).permute(1, 0, 2))
        attn_output = Func.relu(self.lnorm2(self.linear(attn_output))).transpose(1, 2)
        return attn_output


class DfOp(nn.Module):
    def __init__(
        self,
        df_bins: int,
        df_order: int = 5,
        df_lookahead: int = 0,
        method: str = "complex_strided",
        freq_bins: int = 0,
    ):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
        self.df_lookahead = df_lookahead
        self.freq_bins = freq_bins
        self.set_forward(method)

    def set_forward(self, method: str):
        # All forward methods should be mathematically similar.
        # DeepFilterNet results are obtained with 'real_unfold'.
        forward_methods = {
            "real_loop": self.forward_real_loop,
            "real_strided": self.forward_real_strided,
            "real_unfold": self.forward_real_unfold,
            "complex_strided": self.forward_complex_strided,
            "real_one_step": self.forward_real_no_pad_one_step,
            "real_hidden_state_loop": self.forward_real_hidden_state_loop,
        }
        if method not in forward_methods.keys():
            raise NotImplementedError(
                f"`method` must be one of {forward_methods.keys()}, but got '{method}'"
            )
        if method == "real_hidden_state_loop":
            assert self.freq_bins >= self.df_bins
            self.spec_buf: Tensor
            # Currently only designed for batch size of 1
            self.register_buffer(
                "spec_buf", torch.zeros(1, 1, self.df_order, self.freq_bins, 2), persistent=False
            )
        self.forward = forward_methods[method]

    def forward_real_loop(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version 0: Manual loop over df_order, maybe best for onnx export?
        b, _, t, _, _ = spec.shape
        f = self.df_bins
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )

        spec_f = torch.zeros((b, t, f, 2), device=spec.device)
        for i in range(self.df_order):
            spec_f[..., 0] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 0]
            spec_f[..., 0] -= padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 1]
            spec_f[..., 1] += padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 0]
            spec_f[..., 1] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 1]
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version1: Use as_strided instead of unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        # Complex numbers are not supported by onnx
        re = padded[..., 0] * coefs[..., 0]
        re -= padded[..., 1] * coefs[..., 1]
        im = padded[..., 1] * coefs[..., 0]
        im += padded[..., 0] * coefs[..., 1]
        spec_f = torch.stack((re, im), -1).sum(2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_unfold(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version2: Unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        padded = padded.unfold(dimension=1, size=self.df_order, step=1)  # [B, T, F, 2, O]
        padded = padded.permute(0, 1, 4, 2, 3)
        spec_f = torch.empty_like(padded)
        spec_f[..., 0] = padded[..., 0] * coefs[..., 0]  # re1
        spec_f[..., 0] -= padded[..., 1] * coefs[..., 1]  # re2
        spec_f[..., 1] = padded[..., 1] * coefs[..., 0]  # im1
        spec_f[..., 1] += padded[..., 0] * coefs[..., 1]  # im2
        spec_f = spec_f.sum(dim=2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_complex_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version3: Complex strided; definatly nicest, no permute, no indexing, but complex gradient
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        spec_f = torch.sum(torch.view_as_complex(padded) * torch.view_as_complex(coefs), dim=2)
        spec_f = torch.view_as_real(spec_f)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_no_pad_one_step(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version4: Only viable for onnx handling. `spec` needs external (ring-)buffer handling.
        # Thus, time steps `t` must be equal to `df_order`.

        # spec (real) [B, 1, O, F', 2]
        # coefs (real) [B, 1, O, F, 2]
        assert (
            spec.shape[2] == self.df_order
        ), "This forward method needs spectrogram buffer with `df_order` time steps as input"
        assert coefs.shape[1] == 1, "This forward method is only valid for 1 time step"
        sre, sim = spec[..., : self.df_bins, :].split(1, -1)
        cre, cim = coefs.split(1, -1)
        outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
        outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
        spec_f = torch.stack((outr, outi), dim=-1)
        return assign_df(
            spec[:, :, self.df_order - self.df_lookahead - 1],
            spec_f.unsqueeze(1),
            self.df_bins,
            alpha,
        )

    def forward_real_hidden_state_loop(self, spec: Tensor, coefs: Tensor, alpha: Tensor) -> Tensor:
        # Version5: Designed for onnx export. `spec` buffer handling is done via a torch buffer.

        # spec (real) [B, 1, T, F', 2]
        # coefs (real) [B, T, O, F, 2]
        b, _, t, _, _ = spec.shape
        spec_out = torch.empty((b, 1, t, self.freq_bins, 2), device=spec.device)
        for t in range(spec.shape[2]):
            self.spec_buf = self.spec_buf.roll(-1, dims=2)
            self.spec_buf[:, :, -1] = spec[:, :, t]
            sre, sim = self.spec_buf[..., : self.df_bins, :].split(1, -1)
            cre, cim = coefs[:, t : t + 1].split(1, -1)
            outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
            outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
            spec_f = torch.stack((outr, outi), dim=-1)
            spec_out[:, :, t] = assign_df(
                self.spec_buf[:, :, self.df_order - self.df_lookahead - 1].unsqueeze(2),
                spec_f.unsqueeze(1),
                self.df_bins,
                alpha[:, t],
            ).squeeze(2)
        return spec_out


def assign_df(spec: Tensor, spec_f: Tensor, df_bins: int, alpha: Optional[Tensor]):
    spec_out = spec.clone()
    if alpha is not None:
        b = spec.shape[0]
        alpha = alpha.view(b, 1, -1, 1, 1)
        spec_out[..., :df_bins, :] = spec_f * alpha + spec[..., :df_bins, :] * (1 - alpha)
    else:
        spec_out[..., :df_bins, :] = spec_f
    return spec_out


def spec_pad(x: Tensor, window_size: int, lookahead: int, dim: int = 0) -> Tensor:
    pad = [0] * x.dim() * 2
    if dim >= 0:
        pad[(x.dim() - dim - 1) * 2] = window_size - lookahead - 1
        pad[(x.dim() - dim - 1) * 2 + 1] = lookahead
    else:
        pad[(-dim - 1) * 2] = window_size - lookahead - 1
        pad[(-dim - 1) * 2 + 1] = lookahead
    return Func.pad(x, pad)


def as_strided(x: Tensor, window_size: int, lookahead: int, step: int = 1, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape.insert(dim + 1, window_size)
    x = spec_pad(x, window_size, lookahead, dim=dim)
    # torch.fx workaround
    step = 1
    stride = [x.stride(0), x.stride(1), x.stride(2), x.stride(3)]
    stride.insert(dim, stride[dim] * step)
    return torch.as_strided(x, shape, stride)