# This Script is currently private and only meant to serve as a reference to the reviewers of ASRU 2023

import torch
import torch as th
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from feature import DF_and_SteerVector, IPD_feature
from model_utils import SelfAttention

EPSILON = th.finfo(th.float32).eps


def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10 ** 6 if Mb else neles


class FeatureExtract(nn.Module):
    """
    creat features : LPS、IPD and Angle Feature
    """
    def __init__(self, channels=4, win_len=512, fft_len=512, win_inc=256, pair_index=None):
        super(FeatureExtract, self).__init__()
        self.ipd_extractor = IPD_feature(channels=channels, pair_idx=pair_index)

        self.df_extractor = DF_and_SteerVector(array_r=0.03,
                                               channels=channels,
                                               num_bins=fft_len//2+1,
                                               pair_idx=pair_index)
        self.stft = Spectrogram(n_fft=fft_len, win_length=win_len, hop_length=win_inc, power=None)

    def forward(self, inputs, doa):
        wav_stft = self.stft(inputs)
        ipd = self.ipd_extractor(wav_stft)
        mag = torch.abs(wav_stft)
        # lps = (2 * th.log(th.clamp(mag[:, 0], EPSILON)))  # lps: B x F x T,第0通道的lps
        df, steer_vector = self.df_extractor(wav_stft, doa)
        # inp = th.cat([mag[:, 0], ipd], 1)  # B x (M+1+1)*F x T

        return mag[:, 0], ipd, df, wav_stft, steer_vector


def get_power_spectral_density_matrix_self_with_cm_t(xs):
    psd = torch.einsum('...ct,...et->...tce', [xs, xs.conj()])
    return psd


def apply_beamforming_vector(beamform_vector, mix):
    es = th.einsum('bftc,bfct->bft', [beamform_vector.conj(), mix])
    return es


class DPT_BF(nn.Module):
    def __init__(self,
                 fft_len=512,
                 win_inc=256,
                 win_len=512,
                 pair_index=None,
                 channels=4,
                 dropout_rate=0.2,
                 pattern=False
                 ):
        super(DPT_BF, self).__init__()
        if pair_index is None:
            pair_index = [(0, 1), (0, 2), (0, 3)]
        self.dropout_rate = dropout_rate
        self.fft_len = fft_len
        self.channels = channels
        self.feat_channel_dim = len(pair_index) + 1 + 1
        self.pattern = pattern

        self.feature = FeatureExtract(win_len=512, fft_len=512, win_inc=256, pair_index=pair_index)
        self.inverse_stft = InverseSpectrogram(n_fft=fft_len, hop_length=win_inc, win_length=win_len)

        self.hid_dim = 256
        self.feat_emb = nn.Conv1d(5, self.hid_dim//2, 1)
        self.feat_emb_norm = nn.LayerNorm(self.hid_dim//2)
        self.feat_relu = nn.PReLU()

        self.scm_emb = nn.Conv1d(32, self.hid_dim//2, 1)
        self.scm_emb_norm = nn.LayerNorm(self.hid_dim//2)
        self.scm_relu = nn.PReLU()

        self.GRU_emb = nn.GRU(self.hid_dim, self.hid_dim, 2, batch_first=True)
        self.attn_t = SelfAttention(embed_dim=self.hid_dim//2, num_heads=4, output_dim=self.hid_dim//2,
                                    locations=[None, None], dropout=self.dropout_rate)
        self.attn_f = SelfAttention(embed_dim=self.hid_dim // 2, num_heads=4, output_dim=self.hid_dim // 2,
                                    locations=[None, None], dropout=self.dropout_rate)
        self.get_ws = nn.Conv1d(self.hid_dim//2, channels*2, 1)

    def forward(self, input, doa):
        mag, ipd, df, stft, steer_vector = self.feature(input, doa)
        ipd = torch.stack(ipd.chunk(chunks=3, dim=1), dim=1)
        inp = torch.cat([mag.unsqueeze(1), ipd, df.unsqueeze(1)], dim=1)
        B, C, F, T = stft.shape
        phi_yy_cplx = get_power_spectral_density_matrix_self_with_cm_t(stft.transpose(1, 2)).flatten(-2)
        phi_yy = torch.cat([phi_yy_cplx.real, phi_yy_cplx.imag], dim=-1)

        feat_emb = self.feat_emb(inp.transpose(1, 2).reshape(B * F, -1, T))
        feat_emb = self.feat_relu(self.feat_emb_norm(feat_emb.transpose(-1, -2)).transpose(-1, -2))

        phi_yy_emb = self.scm_emb(phi_yy.reshape(B*F, T, -1).transpose(-1, -2))
        phi_yy_emb = self.scm_relu(self.scm_emb_norm(phi_yy_emb.transpose(-1, -2)).transpose(-1, -2))

        fusion_emb = torch.cat([feat_emb, phi_yy_emb], dim=1).transpose(-1, -2)
        fusion_emb = self.GRU_emb(fusion_emb)[0]

        query, key_value = torch.chunk(fusion_emb, 2, dim=-1)
        ws_emb = self.attn_t(query.transpose(-1, -2), key_value.transpose(-1, -2), key_value.transpose(-1, -2))
        ws_emb = ws_emb.reshape(B, F, self.hid_dim//2, T).permute(0, 3, 2, 1).reshape(B*T, self.hid_dim//2, F)
        ws_emb = self.attn_f(ws_emb, ws_emb, ws_emb)
        ws_emb = ws_emb.reshape(B, T, self.hid_dim//2, F).permute(0, 3, 2, 1).reshape(B*F, self.hid_dim//2, T)
        ws = self.get_ws(ws_emb).reshape(B, F, self.channels*2, T)
        ws_cplx = (ws[:, :, :self.channels, :] + 1j*ws[:, :, self.channels:, :]).transpose(-1, -2)
        enhanced_spec = apply_beamforming_vector(ws_cplx, stft.transpose(1, 2))
        res = self.inverse_stft(enhanced_spec)
        if self.pattern:
            return res, ws_cplx
        return res, enhanced_spec


def testing():
    from thop import profile, clever_format
    x = th.rand(1, 4, 16000)
    nnet = DPT_BF()
    print("Net #param: {:.2f}".format(param(nnet)))
    y, _ = nnet(x, th.tensor([10]))
    print(y.shape)
    macs, params = profile(nnet, inputs=(x, th.tensor([10])))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)


if __name__ == "__main__":
    testing()
