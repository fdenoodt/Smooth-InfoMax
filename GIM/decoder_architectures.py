# %%
from helper_functions import *
from decoder_architectures import *
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import torch.fft as fft


class GimDecoder(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name


class GimL4Decoder(GimDecoder):
    # (batch_size, 512, 129) -> (batch_size, 1, 10240)
    def __init__(self, hidden_channels=512, output_channels=1, output_dim=10240):
        super().__init__("GIM_L4_DECODER")

        # inp: (batch_size, 512, 129) = (B, Chann, Height)
        self.conv_trans_layer0 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=4, padding=2, output_padding=0, stride=2)

        # inp: (batch_size, 512, 256) = (B, Chann, Height)
        self.conv_trans_layer1 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=4, padding=2, output_padding=1, stride=2)
        # output_height = [(input_height - 1) * stride] + kernel_size[0] - [2 * padding] + output_padding
        # out_H = [(256 - 1) * 2] + 4 - [2 * 2] + 1 = 511

        self.conv_trans_layer2 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=10, padding=2, output_padding=1, stride=4)
        # out_H = [(511 - 1) * 4] + 10 - [2 * 2] + 1 = 2047

        self.conv_trans_layer3 = nn.ConvTranspose1d(
            hidden_channels, output_channels, kernel_size=12, padding=2, output_padding=2, stride=5)
        # out_H = [(2047 - 1) * 5] + 12 - [2 * 2] + 2 = 10240

        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.conv_trans_layer0(x))
        x = F.relu(self.conv_trans_layer1(x))
        x = F.relu(self.conv_trans_layer2(x))
        x = self.conv_trans_layer3(x)

        return x


class GimL3Decoder(GimDecoder):
    # (batch_size, 512, 256) -> (batch_size, 1, 10240)
    def __init__(self, hidden_channels=512, output_channels=1, output_dim=10240):
        super().__init__("GIM_L3_DECODER")

        # inp: (batch_size, 512, 256) = (B, Chann, Height)
        self.conv_trans_layer1 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=4, padding=2, output_padding=1, stride=2)
        # output_height = [(input_height - 1) * stride] + kernel_size[0] - [2 * padding] + output_padding
        # out_H = [(256 - 1) * 2] + 4 - [2 * 2] + 1 = 511

        self.conv_trans_layer2 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=10, padding=2, output_padding=1, stride=4)
        # out_H = [(511 - 1) * 4] + 10 - [2 * 2] + 1 = 2047

        self.conv_trans_layer3 = nn.ConvTranspose1d(
            hidden_channels, output_channels, kernel_size=12, padding=2, output_padding=2, stride=5)
        # out_H = [(2047 - 1) * 5] + 12 - [2 * 2] + 2 = 10240

        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.conv_trans_layer1(x))
        x = F.relu(self.conv_trans_layer2(x))
        x = self.conv_trans_layer3(x)

        return x


class GimL2Decoder(GimDecoder):
    # (batch_size, 512, 511) -> (batch_size, 1, 10240)
    def __init__(self, hidden_channels=512, output_channels=1, output_dim=10240):
        super().__init__("GIM_L2_DECODER")

        self.conv_trans_layer2 = nn.ConvTranspose1d(
            hidden_channels, hidden_channels, kernel_size=10, padding=2, output_padding=1, stride=4)
        # out_H = [(511 - 1) * 4] + 10 - [2 * 2] + 1 = 2047

        self.conv_trans_layer3 = nn.ConvTranspose1d(
            hidden_channels, output_channels, kernel_size=12, padding=2, output_padding=2, stride=5)
        # out_H = [(2047 - 1) * 5] + 12 - [2 * 2] + 2 = 10240

        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.conv_trans_layer2(x))
        x = self.conv_trans_layer3(x)

        return x


class GimL1Decoder(GimDecoder):
    # (batch_size, 512, 2047) -> (batch_size, 1, 10240)
    def __init__(self, hidden_channels=512, output_channels=1, output_dim=10240):
        super().__init__("GIM_L1_DECODER")

        self.conv_trans_layer3 = nn.ConvTranspose1d(
            hidden_channels, output_channels, kernel_size=12, padding=2, output_padding=2, stride=5)
        # out_H = [(2047 - 1) * 5] + 12 - [2 * 2] + 2 = 10240

        self.output_dim = output_dim

    def forward(self, x):
        x = self.conv_trans_layer3(x)

        return x


class SpectralLoss(nn.Module):
    # aided by ChatGPT
    def __init__(self, n_fft=1024):
        super(SpectralLoss, self).__init__()
        self.name = f"Spectral Loss FFT={n_fft}"
        self.n_fft = n_fft
        self.loss = nn.MSELoss()

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape

        # batch_inputs.shape: (batch_size, 1, length)
        batch_inputs = batch_inputs.squeeze(1)  # (batch_size, length)
        batch_targets = batch_targets.squeeze(1)  # (batch_size, length)

        input_spectograms = torch.stft(
            batch_inputs, self.n_fft, return_complex=False)  # only magnitude
        target_spectograms = torch.stft(
            batch_targets, self.n_fft, return_complex=False)  # only magnitude

        input_spectograms = input_spectograms.pow(2).sum(-1)
        target_spectograms = target_spectograms.pow(2).sum(-1)

        return self.loss(input_spectograms, target_spectograms)

class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        self.name = "MSE Loss"
        self.mse_loss = nn.MSELoss()

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        return self.mse_loss(batch_inputs, batch_targets)


class MSE_AND_SPECTRAL_LOSS(nn.Module):
    def __init__(self, n_fft=1024, lambd=1):
        super(MSE_AND_SPECTRAL_LOSS, self).__init__()
        self.name = f"MSE + scSpectral Loss FFT={n_fft} Lambda={lambd:.7f}"
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralLoss(n_fft)
        self.lambd = lambd

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        return self.mse_loss(batch_inputs, batch_targets) + (self.lambd * self.spectral_loss(batch_inputs, batch_targets))


class FFTLoss(nn.Module):
    # generated via chat gpt
    def __init__(self, fft_size=10240):
        super(FFTLoss, self).__init__()
        # The value of the FFT size should be chosen based on the properties of your signal, such as its sample rate and the frequency content you are interested in analyzing. In general, the FFT size determines the frequency resolution of the analysis, and a larger FFT size will provide better frequency resolution at the expense of time resolution.
        # For a signal with a sample rate of 16,000 Hz, you could choose an FFT size that is a power of two and is equal to or greater than the length of your signal. A common choice for audio signals is 2048 or 4096 samples, which would correspond to a frequency resolution of approximately 8 or 4 Hz, respectively. However, you may need to experiment with different FFT sizes to determine the best choice for your particular application.
        # In addition to the FFT size, the choice of window function can also affect the quality of the spectral analysis. The Hann window used in the example I provided is a good default choice, but you may want to try other window functions such as the Blackman-Harris or Kaiser windows to see if they improve the accuracy of your analysis.
        self.name = f"FFT Loss FFT={fft_size}"
        self.fft_size = fft_size
        self.window = torch.hann_window(fft_size, periodic=True).to(device)

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

if __name__ == "__main__":
    # (batch_size, 1, 10240)
    rnd = torch.rand((2, 512, 2047)).to(device)  # layer 1

    # %%
    signal1 = torch.rand((5, 1, 10240)).to(device)
    signal2 = torch.rand((5, 1, 10240)).to(device)

    criterion = FFTLoss(10240).to(device)  # must be power
    loss = criterion(signal1, signal2)


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # decoder = GimL1Decoder().to(device)

    # model_path = "./logs\\RMSE_decoder2_experiment\\optim_24.ckpt"
    # decoder.load_state_dict(torch.load(model_path, map_location=device))

    # rnd = torch.rand((2, 512, 2047)).to(device) # layer 1
    # rnd = torch.rand((2, 512, 511)).to(device) # layer 2
    # rnd = torch.rand((2, 512, 256)).to(device) # layer 3
    # rnd = torch.rand((96, 512, 129)).to(device) # layer 4
    # print(decoder(rnd).shape)


# %%


# def encode(audio, model, depth=1):
#     audios = audio.unsqueeze(0)
#     model_input = audios.to(device)

#     for idx, layer in enumerate(model.module.fullmodel):
#         context, z = layer.get_latents(model_input)
#         model_input = z.permute(0, 2, 1)

#         if(idx == depth - 1):
#             return z


# def encoder_lambda(xs_batch):
#     # Gim_encoder is outerscope variable
#     with torch.no_grad():
#         return encode(xs_batch, GIM_encoder, depth=1)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# opt['batch_size'] = 8

# GIM_encoder, _ = load_model(path='./g_drive_model/model_180.ckpt')
# GIM_encoder.eval()

# random.seed(0)

# # %%


# def element_from_data_loader(data_loader):
#     for step, batch in enumerate(data_loader):
#         return batch

# if __name__ == "__main__":
#     train_loader, _, _, _ = get_dataloader.get_de_boer_sounds_decoder_data_loaders(
#         opt,
#         GIM_encoder=encoder_lambda)


#     batch = element_from_data_loader(train_loader)
#     (org_audio, enc_audio, _, _, _) = batch
#     org_audio = org_audio.to(device)
#     enc_audio = enc_audio.to(device)


#     # %%
#     print(org_audio.shape)
#     print(enc_audio.shape)


# target shape: inp = torch.rand(([2, 512, 2047])).to('cuda')
# current shape: inp = torch.rand(([2, 2047, 512]))
# decoder = OneLayerDecoder().to('cuda')
# d = decoder(inp)


#     # %%

#     criterion = nn.MSELoss()

#     optimizer = torch.optim.Adam(decoder.parameters(), lr=1.5e-2)

#     loss = criterion(d, org_audio)
#     loss.backward()
#     optimizer.step()


# %%


# %%



