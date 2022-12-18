# %%
import torch.nn.functional as F
import torch.nn as nn
import os
import IPython.display as ipd
from options import OPTIONS as opt
from utils import model_utils
import torch
import numpy as np
from utils import logger
from arg_parser import arg_parser
from data import get_dataloader
import os
from models import full_model
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_layers():
    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #
    pass


def plot_spectrogram(signal, name):
    # plot_spectrogram(audios[0].to('cpu').numpy()[0], "bibaga")
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for {name}")
    plt.xlabel("Time")
    plt.show()


def load_model(path):
    # Code comes from: def load_model_and_optimizer()
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    padding = [2, 2, 2, 2, 1]
    enc_hidden = 512
    reg_hidden = 256

    calc_accuracy = False
    num_GPU = None

    # Initialize model.
    model = full_model.FullModel(
        opt,
        kernel_sizes=kernel_sizes,
        strides=strides,
        padding=padding,
        enc_hidden=enc_hidden,
        reg_hidden=reg_hidden,
        calc_accuracy=calc_accuracy,
    )

    # Run on only one GPU for supervised losses.
    if opt["loss"] == 2 or opt["loss"] == 1:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(
        opt, model, num_GPU=num_GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])
    model.load_state_dict(torch.load(path))

    return model, optimizer


def play_sound(audio):
    # ipd.Audio(os.path.join(
    #     r"C:\GitHub\thesis-fabian-denoodt\GIM\datasets\gigabo\train", f"{filenames[0]}.wav"), rate=44100)

    # audios[0].shape
    # ipd.Audio(audios[0], rate=44100)
    ipd.Audio(audio, rate=16000)


def plot_fft():
    wave = audios[0][0].to('cpu').numpy()
    X = np.fft.fft(wave)
    X_mag = np.absolute(X)
    plt.figure(figsize=(18, 10))
    plt.plot(X_mag)  # magnitude spectrum
    plt.xlabel('Frequency (Hz)')


def encode(audio, model, depth=1):
    audios = audio.unsqueeze(0)
    model_input = audios.to(device)

    for idx, layer in enumerate(model.module.fullmodel):
        context, z = layer.get_latents(model_input)
        model_input = z.permute(0, 2, 1)

        if(idx == depth - 1):
            return z

        # model_input = z.permute(0, 2, 1)
        # latent_rep = context.permute(0, 2, 1).cpu().numpy()


# %%

GIM_encoder, _ = load_model(path='./g_drive_model/model_180.ckpt')
GIM_encoder.eval()


def encoder_lambda(xs_batch):
    # No auto differentaiton support error:
    # Chat gpt:
    # It looks like the issue is with the encode function being called with a gradient-enabled tensor as the audios argument, which is causing the stack function to be called with a gradient-enabled tensor as an argument. This is causing the RuntimeError because the stack function does not support automatic differentiation when one of its arguments requires a gradient.
    # One solution would be to disable gradients for the audios tensor before calling the encode function. You can do this by using the torch.no_grad context manager or the detach method. Here's what the modified code would look like using the torch.no_grad context manager:

    # Gim_encoder is outerscope variable
    with torch.no_grad():
        return encode(xs_batch, GIM_encoder, depth=1)


# chat gtp request: create a torch 1dimensional convolutional decoder class (one layer) that goes from dimension: 2047 with 512 channels to 1 channel and 10240 dimensions
# class Conv1DDecoder(nn.Module):
#     def __init__(self, input_channels=512, output_channels=1, output_dim=10240):
#         super().__init__()
#         self.deconv1d = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=3, padding=1, output_padding=1, stride=2)
#         self.deconv1d.weight.data

#     def forward(self, x):
#         x = self.conv1d(x)
#         return x


class Conv1DDecoder(nn.Module):
    # The Conv1DDecoder class is a simple one-layer convolutional neural network that takes an input tensor of shape (batch_size, input_channels, input_dim) and applies a 1D convolution to produce an output tensor of shape (batch_size, output_channels, output_dim).
    # The 1D convolution is defined in the __init__ method of the Conv1DDecoder class using the nn.Conv1d module from PyTorch. This module takes three arguments:
    # input_channels: the number of input channels in the input tensor. In this case, the input tensor has shape (batch_size, 512, 2047), so input_channels is set to 512.
    # output_channels: the number of output channels in the output tensor. In this case, the output tensor should have shape (batch_size, 1, 10240), so output_channels is set to 1.
    # kernel_size: the size of the convolutional kernel. In this case, the kernel size is set to 3, which means that the convolutional operation will consider a window of 3 adjacent values in the input tensor at each position.
    # The forward pass of the Conv1DDecoder class is defined in the forward method. This method takes an input tensor x and applies the 1D convolution defined in the __init__ method to produce an output tensor. The output tensor is then returned.
    def __init__(self, input_channels=512, output_channels=1, output_dim=10240):
        super().__init__()

        # These lines of code define the layers and initialization of the Conv1DDecoder model.
        # The ConvTranspose1d layer is a type of 1D convolutional layer that is used for upsampling the input data. It works by inserting zeros between the elements of the input tensor and then performing a normal 1D convolution operation. The kernel_size, padding, and stride parameters control the shape of the kernel and the spacing between the elements of the input tensor, just like in a regular 1D convolutional layer. The output_padding parameter controls the number of zeros to insert between the elements of the output tensor.
        # The weight and bias parameters of the ConvTranspose1d layer are initialized using the normal_ and zero_ methods of the Tensor class, respectively. The normal_ method initializes the weights with random values drawn from a normal distribution with mean 0 and standard deviation 0.02, while the zero_ method initializes the biases with 0.
        # Finally, the output_dim attribute of the Conv1DDecoder class is initialized with the output_dim parameter, which is the expected size of the output tensor.
        self.deconv1d = nn.ConvTranspose1d(
            input_channels, output_channels,
            kernel_size=3, padding=1, output_padding=1, stride=2)
        self.deconv1d.weight.data.normal_(0, 0.02)
        self.deconv1d.bias.data.zero_()
        self.output_dim = output_dim

    def forward(self, x):
        x = self.deconv1d(x)
        # The F.interpolate function will allow you to resize the output tensor to the desired size of output_dim.
        x = F.interpolate(x, size=self.output_dim,
                          mode='linear', align_corners=False)
        return x


errors = []  # store the errors here


def train():
    decoder = Conv1DDecoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    # load the data
    train_loader, _, _, _ = get_dataloader.get_de_boer_sounds_decoder_data_loaders(
        opt,
        GIM_encoder=encoder_lambda)

    for epoch in range(20):
        epoch_loss = 0
        for step, (org_audio, enc_audio, filename, _, start_idx) in enumerate(train_loader):
            # what comes out is hidden repr batch of audios
            enc_audios = enc_audio.to(device)  # torch.Size([2, 1, 2047, 512])
            enc_audios = enc_audios.squeeze(
                dim=1)  # torch.Size([2, 2047, 512])
            enc_audios = enc_audios.permute(
                0, 2, 1)  # torch.Size([2, 512, 2047])

            org_audio = org_audio.to(device)  # torch.Size([2, 1, 10240])

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = decoder(enc_audios)
            outputs = outputs.squeeze(dim=1)

            org_audio = org_audio.squeeze(dim=1)  # torch.Size([2,10240])
            loss = criterion(outputs, org_audio)

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            # print the loss at each step
            epoch_loss += loss.item()  # sum of errors instead of mean

        print(f"Epoch {epoch}, Loss: {epoch_loss}")
        errors.append(loss.item())  # store the error

    torch.save(decoder.state_dict(), "decoder.pth")

    plt.plot(errors)  # plot the errors
    plt.xlabel("Training steps")
    plt.ylabel("Training error")
    plt.show()

    return decoder


if __name__ == "__main__":
    arg_parser.create_log_path(opt)

    logs = logger.Logger(opt)

    decoder = train()
    torch.cuda.empty_cache()

    # %%

    decoder.eval()
    train_loader, _, _, _ = get_dataloader.get_de_boer_sounds_decoder_data_loaders(
        opt,
        GIM_encoder=encoder_lambda)

    enc_audios = None
    org_audio = None
    prediction = None
    for step, (org_audio, enc_audio, filename, _, start_idx) in enumerate(train_loader):

        enc_audios = enc_audio.to(device)  # torch.Size([2, 1, 2047, 512])
        enc_audios = enc_audios.squeeze(dim=1)  # torch.Size([2, 2047, 512])
        enc_audios = enc_audios.permute(0, 2, 1)  # torch.Size([2, 512, 2047])

        org_audio = org_audio.to(device)  # torch.Size([2, 1, 10240])

        prediction = decoder(enc_audios)
        print(prediction.shape)

        break

    # %%
    """
    Observations:
    First layer decoded still contains the same sound, but with some added noise (could be because decoder hasn't trained very).
    However, the encoded first layer, still contains the exact sound as the original sound. It is however downsampled a lot -> from 16khz to ~3khz
    """
    # ipd.Audio(prediction[0][0].to('cpu').detach().numpy(), rate=16000)
    # ipd.Audio(org_audio[0][0].to('cpu').detach().numpy(), rate=16000)

    # ipd.Audio(enc_audios[0][0].to('cpu').detach().numpy(), rate=3000)
    # ipd.Audio(enc_audios[0][1].to('cpu').detach().numpy(), rate=3000)
    # ipd.Audio(enc_audios[0][50].to('cpu').detach().numpy(), rate=3000)
    
    ipd.Audio(enc_audios[0][100].to('cpu').detach().numpy(), rate=3000)
    
    # ipd.Audio(enc_audios[0][0].to('cpu').detach().numpy(), rate=16000)

    # multiple channels
    
    # %%


    # %%
    plot_spectrogram(enc_audios[0][0].to('cpu').detach().numpy(), "encoded")
    # %%
    plot_spectrogram(prediction[0][0].to('cpu').detach().numpy(), "prediction")
    # %%
    plot_spectrogram(org_audio[0][0].to('cpu').detach().numpy(), "original")

    # %%
    plt.plot(prediction[0][0].to('cpu').detach().numpy())
    plt.show()

    plt.plot(org_audio[0][0].to('cpu').detach().numpy())
    plt.show()

    plt.plot(enc_audios[0][0].to('cpu').detach().numpy())
    plt.show()

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?
