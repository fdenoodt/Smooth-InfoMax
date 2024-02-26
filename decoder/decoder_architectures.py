# %%
import torchaudio.transforms as T
from utils.helper_functions import *
from decoder_architectures import *
import torch.nn.functional as F
import torch.nn as nn
import torch
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


class SimpleV1Decoder(GimDecoder):
    def __init__(self, hidd_channels=32, out_channels=1):
        super().__init__("Simple_v1_DECODER")

        # Encoder architecture (Simple v1)
        kernel_sizes = [10, 10, 3]
        strides = [5, 5, 1]
        padding = [0, 0, 1]
        output_padding = [0, 0, 0]
        max_unpool_k_size = 8
        max_unpool_stride = 4

        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.ReLU(),

            # # Replaces maxpooling
            nn.ConvTranspose1d(hidd_channels, hidd_channels, max_unpool_k_size,
                               stride=max_unpool_stride, padding=0, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[1], stride=strides[1], padding=padding[1]),
            nn.ReLU(),

            # # Replaces maxpooling
            nn.ConvTranspose1d(hidd_channels, hidd_channels, max_unpool_k_size,
                               stride=max_unpool_stride, padding=0, output_padding=3),
            nn.ReLU(),

            nn.ConvTranspose1d(hidd_channels, out_channels,
                               kernel_sizes[0], stride=strides[0], padding=padding[0]),
        )

    def forward(self, x):
        return self.decoder(x)


# x = torch.randn(96, 1, 10240) # = objective
# z = torch.randn(96, 24, 32) # (b, l, c)
# # z = torch.randn(96, 2047, 32)
# # z = torch.randn(96, 510, 32)
# # z = torch.randn(96, 101, 32)
# # z = torch.randn(96, 24, 32)
# z = z.permute(0, 2, 1) # (b, c, l)

# decoder = SimpleV1Decoder()
# y = decoder(z)
# print(y.shape)

# c1 = nn.Conv1d(1, 32, 10, 5, 0)(x) # 10240 -> 2047
# p1 = nn.MaxPool1d(8, 4)(c1) # 2047 -> 510
# c2 = nn.Conv1d(32, 32, 10, 5, 0)(p1) # 510 -> 101
# p2 = nn.MaxPool1d(8, 4)(c2) # 101 -> 24
# p2.shape

class SimpleV2Decoder(GimDecoder):
    def __init__(self, hidd_channels=32, out_channels=1):
        super().__init__("Simple_v2_DECODER")

        # Encoder architecture (Simple v2)
        kernel_sizes = [10, 8, 3]
        strides = [4, 3, 1]
        padding = [2, 2, 1]
        max_unpool_k_size = 8
        max_unpool_stride = 4

        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.ReLU(),

            # Replaces maxpooling
            nn.ConvTranspose1d(hidd_channels, hidd_channels, max_unpool_k_size,
                               stride=max_unpool_stride, padding=0, output_padding=0),
            nn.ReLU(),

            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[1], stride=strides[1], padding=padding[1], output_padding=1),
            nn.ReLU(),

            # Replaces maxpooling
            nn.ConvTranspose1d(hidd_channels, hidd_channels, max_unpool_k_size,
                               stride=max_unpool_stride, padding=0, output_padding=3),
            nn.ReLU(),

            nn.ConvTranspose1d(hidd_channels, out_channels,
                               kernel_sizes[0], stride=strides[0], padding=padding[0], output_padding=2),
        )

    def forward(self, x):
        return self.decoder(x)


class SimpleV2DecoderTwoModules(GimDecoder):
    def __init__(self, hidd_channels=32, out_channels=1):
        super().__init__("Simple_v2_2Module_DECODER")

        kernel_sizes = [8, 8, 3]
        strides = [3, 3, 1]
        padding = [2, 2, 1]

        self.module2 = nn.Sequential(
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.ReLU(),
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[1], stride=strides[1], padding=padding[1], output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[0], stride=strides[0], padding=padding[0], output_padding=0),
        )
        self.module1 = SimpleV2Decoder(hidd_channels, out_channels)

    def forward(self, z):
        z = self.module2(z)
        x = self.module1(z)  # SimpleV2Decoder
        return x


class SimpleV3DecoderTwoModules(GimDecoder):
    def __init__(self, hidd_channels=32, out_channels=1):
        super().__init__("Simple_v3_2Module_DECODER")

        kernel_sizes = [6, 6, 3]
        strides = [2, 2, 1]
        padding = [2, 2, 1]

        self.module2 = nn.Sequential(
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.ReLU(),
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[1], stride=strides[1], padding=padding[1], output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(hidd_channels, hidd_channels,
                               kernel_sizes[0], stride=strides[0], padding=padding[0], output_padding=0),
        )
        self.module1 = SimpleV2Decoder(hidd_channels, out_channels)

    def forward(self, z):
        z = self.module2(z)
        x = self.module1(z)  # SimpleV2Decoder
        return x


# # # # 52 --> 10240
# z = torch.randn(96, 13, 8)  # (b, l, c)
# # z = torch.randn(96, 5, 32)  # (b, l, c)
# # # z = torch.randn(96, 52, 32)  # (b, l, c)


# # # z = torch.randn(96, 2559, 32)  # (b, l, c)
# # # z = torch.randn(96, 638, 32)  # (b, l, c)
# # # z = torch.randn(96, 212, 32)  # (b, l, c)
# # # z = torch.randn(96, 52, 32)  # (b, l, c)
# z = z.permute(0, 2, 1)  # (b, c, l)

# decoder = SimpleV3DecoderTwoModules()
# y = decoder(z)
# print(y.shape)


# # x = torch.randn(96, 1, 10240)  # = objective

# # c1 = nn.Conv1d(1, 32, 10, 4, 2)(x)  # 10240 -> 2559
# # p1 = nn.MaxPool1d(8, 4)(c1)  # --> 638
# # c2 = nn.Conv1d(32, 32, 8, 3, 2)(p1)  # --> 212
# # p2 = nn.MaxPool1d(8, 4)(c2)  # 52
# # p2.shape  # --> 10240 --> 52
# # print(p2.shape)

# x = p2
# # c1 = nn.Conv1d(32, 32, 8, 3, 2)(x)
# # c2 = nn.Conv1d(32, 32, 8, 3, 2)(c1)

# # c1 = nn.Conv1d(32, 32, 6, 2, 2)(x) # arch simple v3
# # c2 = nn.Conv1d(32, 32, 6, 2, 2)(c1)
# # c2.shape # --> 10240 --> 13




if __name__ == "__main__":
    directory = r"C:\GitHub\thesis-fabian-denoodt\GIM\datasets\corpus\train"
    file1 = "bababi_1.wav"
    file2 = "bababu_1.wav"

    signal1, sr = librosa.load(f"{directory}/{file1}", sr=16000)
    signal2, sr = librosa.load(f"{directory}/{file2}", sr=16000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal1 = torch.from_numpy(signal1).unsqueeze(0).unsqueeze(0).to(device)
    signal2 = torch.from_numpy(signal2).unsqueeze(0).unsqueeze(0).to(device)

    print(signal1.shape)

    criterion = MEL_LOSS().to(device)
    loss1 = criterion(signal1, signal2).item()
    loss2 = criterion(signal2, signal1).item()
    loss3 = criterion(signal1, signal1).item()

    print(loss1, loss2, loss3)

    # (batch_size, 1, 10240)
    # rnd = torch.rand((2, 512, 2047)).to(device)  # layer 1

    # %%
    # signal1 = torch.rand((5, 1, 10240)).to(device)
    # signal2 = torch.rand((5, 1, 10240)).to(device)

    # criterion = FFTLoss(10240).to(device)  # must be power
    # loss = criterion(signal1, signal2)

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
