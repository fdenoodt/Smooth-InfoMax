import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.fft as fft
import torchaudio.transforms as T


class SpectralLoss(nn.Module):
    def __init__(self, n_fft=1024):  # should be higher than signal length
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True, dtype=None, layout=torch.strided,
                                        requires_grad=False).cuda()
        self.loss = nn.MSELoss()

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape

        batch_inputs = batch_inputs.squeeze(1)  # (batch_size, length)
        batch_targets = batch_targets.squeeze(1)  # (batch_size, length)

        # n_fft = bin size (number of frequency bins) -> if 16khz, each bin roughly 15hz

        input_spectograms = torch.stft(
            batch_inputs, self.n_fft, window=self.window, return_complex=True)  # returns complex tensor
        target_spectograms = torch.stft(
            batch_targets, self.n_fft, window=self.window, return_complex=True)  # returns complex tensor

        # Convert complex tensor to real tensor using torch.view_as_real
        input_spectograms = torch.view_as_real(input_spectograms).pow(2).sum(-1)
        target_spectograms = torch.view_as_real(target_spectograms).pow(2).sum(-1)

        return self.loss(input_spectograms, target_spectograms)


class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        return self.mse_loss(batch_inputs, batch_targets)


# TODO: WINDOW
class MSE_AND_SPECTRAL_LOSS(nn.Module):
    def __init__(self, n_fft=1024, lambd=1):
        super(MSE_AND_SPECTRAL_LOSS, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralLoss(n_fft)
        self.lambd = lambd

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        return self.mse_loss(batch_inputs, batch_targets) + (
                self.lambd * self.spectral_loss(batch_inputs, batch_targets))


class FFTLoss(nn.Module):
    # generated via chat gpt
    def __init__(self, fft_size=10240):
        super(FFTLoss, self).__init__()
        # The value of the FFT size should be chosen based on the properties of your signal, such as its sample rate and the frequency content you are interested in analyzing. In general, the FFT size determines the frequency resolution of the analysis, and a larger FFT size will provide better frequency resolution at the expense of time resolution.
        # For a signal with a sample rate of 16,000 Hz, you could choose an FFT size that is a power of two and is equal to or greater than the length of your signal. A common choice for audio signals is 2048 or 4096 samples, which would correspond to a frequency resolution of approximately 8 or 4 Hz, respectively. However, you may need to experiment with different FFT sizes to determine the best choice for your particular application.
        # In addition to the FFT size, the choice of window function can also affect the quality of the spectral analysis. The Hann window used in the example I provided is a good default choice, but you may want to try other window functions such as the Blackman-Harris or Kaiser windows to see if they improve the accuracy of your analysis.
        self.fft_size = fft_size
        self.window = torch.hann_window(fft_size, periodic=True).cuda()

    def forward(self, output, target):
        # Compute FFT of output and target signals
        output_fft = fft.rfft(output * self.window)
        target_fft = fft.rfft(target * self.window)

        # Compute magnitude and phase of FFT coefficients
        output_mag, output_phase = torch.abs(output_fft), torch.angle(output_fft)
        target_mag, target_phase = torch.abs(target_fft), torch.angle(target_fft)

        # Compute FFT loss based on magnitude and phase differences
        mag_loss = torch.mean(torch.abs(output_mag - target_mag))
        phase_loss = torch.mean(torch.abs(output_phase - target_phase))
        fft_loss = mag_loss + phase_loss

        return fft_loss


class MSE_AND_FFT_LOSS(nn.Module):
    def __init__(self, fft_size=10240, lambd=1):
        super(MSE_AND_FFT_LOSS, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.fft_loss = FFTLoss(fft_size)
        self.lambd = lambd

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        return self.mse_loss(batch_inputs, batch_targets) + (self.lambd * self.fft_loss(batch_inputs, batch_targets))


class MEL_LOSS(nn.Module):
    # https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#melspectrogram
    def __init__(self, n_fft=4096, sr=16000):
        super().__init__()

        win_length = None  # 1024
        hop_length = n_fft // 2
        n_mels = None  # 128

        self.criterion = nn.MSELoss()
        self.compute_mel_spectr = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            # win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            # n_mels=n_mels,
            mel_scale="htk",
        ).cuda()

    def power_to_db(self, melspec):
        # todo: check if can just call librosa.power_to_db(melspec, ref=1.0, amin=1e-10, top_db=80.0)
        # Alternative
        # inp_mel = 10 * torch.log10(inp_mel)

        amin = 1e-10 * torch.ones_like(melspec) # to avoid log(0)
        ref_value = torch.ones_like(melspec)

        log_spec = 10.0 * torch.log10(torch.maximum(amin, melspec))
        log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))
        return log_spec

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        inp_mel = self.compute_mel_spectr(batch_inputs)
        tar_mel = self.compute_mel_spectr(batch_targets)

        # to decibel scale
        inp_mel = self.power_to_db(inp_mel)
        tar_mel = self.power_to_db(tar_mel)
        # tar_mel = 10 * torch.log10(tar_mel)

        return self.criterion(inp_mel, tar_mel)


class MSE_AND_MEL_LOSS(nn.Module):
    def __init__(self, lambd=1):
        super(MSE_AND_MEL_LOSS, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mel_loss = MEL_LOSS()
        self.lambd = lambd

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        mse = self.mse_loss(batch_inputs, batch_targets)
        mel = self.mel_loss(batch_inputs, batch_targets)
        return mse + (self.lambd * mel)



