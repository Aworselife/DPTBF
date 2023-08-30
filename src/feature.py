import torch
import torch as th
import torch.nn as nn


class IPD_feature(nn.Module):
    """
        Compute Inter-Channel Phase Difference
        input:
        the STFT of the batched Datas, B x C x F x T
        output:
        the IPD of the batched Datas, B x M*F x T, M is the num of mic pairs

        Args:
            channels: The num of mic channel
            pair_idx: The mic pair of Array, if None, traversal all pairs
            cos: compute the cos ipd to get better performance(may add sin also)
    """
    def __init__(self, channels, pair_idx=None, cos=True):
        super(IPD_feature, self).__init__()
        self.cos = cos
        if pair_idx is None:
            pair_idx = [(i, j) for i in range(channels) for j in range(i + 1, channels)]
        self.pair_idx = pair_idx

    def forward(self, spectrogram):
        phase = th.angle(spectrogram)
        ipd = []
        for i, j in self.pair_idx:
            ipd_per_pair = phase[:, i] - phase[:, j]
            ipd.append(ipd_per_pair)
        ipd = torch.cat(ipd, dim=1)
        if self.cos:
            ipd = th.cos(ipd)
        return ipd


class DF_and_SteerVector(nn.Module):
    """
        Compute the Directional/Angle Feature and Steer_Vector
        180 degree <---------> 0 degree
        input:
        the STFT of the batched Datas, B x C x F x T
        the DOA of the batched Datas, B
        output:
        The Directional/Angle Feature of the batched Datas, B x F x T

        Args:
            array_r: The array spacing of the microphone array, Here for Linear Array
            channels: The num of mic channel
            num_bins: The num of Frequency After STFT, n_fft//2+1
            pair_idx: The mic pair of Array, if None, traversal all the pairs
            c: speed of voice
            sr: sample rate of the signal

        Note: This class is for Linear Array, if you want to use other Array,
              change the topo for your array to compute your own tau
    """
    def __init__(self, array_r, channels, num_bins, pair_idx=None, c=343, sr=16000):
        super(DF_and_SteerVector, self).__init__()
        self.array_r = array_r
        self.channels = channels
        self.num_bins = num_bins
        self.c = c
        self.sr = sr
        if pair_idx is None:
            pair_idx = [(i, j) for i in range(channels) for j in range(i + 1, channels)]
        self.pair_idx = pair_idx

    def _get_steer_vector(self, doa):
        topo = torch.arange(self.channels, device=doa.device) * -1 * self.array_r
        dist = th.cos(doa.unsqueeze(-1) * th.pi / 180) * topo
        omega = th.pi * th.arange(self.num_bins, device=doa.device) * self.sr / (self.num_bins - 1)
        tau = (dist / self.c).unsqueeze(-1)
        steer_vector = th.exp(-1j * (omega * tau))
        return steer_vector

    def forward(self, spectrogram, doa):
        steer_vector = self._get_steer_vector(doa)
        arg_s, arg_t = th.angle(spectrogram), th.angle(steer_vector)
        df = []
        for i, j in self.pair_idx:
            delta_s = arg_s[:, i] - arg_s[:, j]
            delta_t = (arg_t[:, i] - arg_t[:, j]).unsqueeze(-1)
            df.append(th.cos(delta_s - delta_t))
        df = th.stack(df, dim=-1).mean(dim=-1)
        return df, steer_vector


if __name__ == "__main__":
    import torchaudio
    from torchaudio.transforms import Spectrogram
    import librosa.display
    import matplotlib.pyplot as plt
    df_get = DF_and_SteerVector(array_r=0.03, channels=4, num_bins=257, pair_idx=[(0, 1)])
    # input a wavfile, The first four channels are the origin multichannel noisy inputs.
    y, fs = torchaudio.load('..//BAC009S0720W0403-35-BAC009S0723W0142-140-0.131.wav')
    stft = Spectrogram(n_fft=512, win_length=512, hop_length=256, power=None)
    wav_stft = stft(y)
    ipt_spec = wav_stft[:4, ...]
    doas = torch.tensor([35, 145])
    res, _ = df_get(ipt_spec.unsqueeze(0).repeat(2, 1, 1, 1), doas)
    print(res[0].shape)
    plt.figure()
    librosa.display.specshow(res[0].detach().numpy())
    plt.show()

    ipd_get = IPD_feature(channels=4, pair_idx=[(0, 1)])
    ipd_result, tmp = ipd_get(ipt_spec.unsqueeze(0))
    print(ipd_result.shape)
    plt.figure()
    librosa.display.specshow(tmp[0][0].detach().numpy())
    plt.show()
