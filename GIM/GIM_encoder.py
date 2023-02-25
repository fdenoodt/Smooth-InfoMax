# %%
from models import full_model
from utils import model_utils
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GIM_Encoder():
    def __init__(self, opt, layer_depth=1, path='./g_drive_model/model_180.ckpt') -> None:
        self.opt = opt

        self.encoder, _ = self.load_model(path)
        self.encoder.eval()
        self.layer_depth = layer_depth

    def __call__(self, xs_batch) -> torch.tensor:
        with torch.no_grad():
            return self.encode(xs_batch)

    def load_model(self, path):
        # Origins comes from: def load_model_and_optimizer()
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]
        padding = [2, 2, 2, 2, 1]
        enc_hidden = 512
        reg_hidden = 256

        calc_accuracy = False
        num_GPU = None

        # Initialize model.
        model = full_model.FullModel(
            self.opt,
            kernel_sizes=kernel_sizes,
            strides=strides,
            padding=padding,
            enc_hidden=enc_hidden,
            reg_hidden=reg_hidden,
            calc_accuracy=calc_accuracy,
        )

        # Run on only one GPU for supervised losses.
        if self.opt["loss"] == 2 or self.opt["loss"] == 1:
            num_GPU = 1

        model, num_GPU = model_utils.distribute_over_GPUs(
            self.opt, model, num_GPU=num_GPU)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.opt['learning_rate'])
        model.load_state_dict(torch.load(path,
                                         map_location=device
                                         ))

        return model, optimizer

    def encode(self, audio_batch):
        # print(audio.shape, "enc")
        # audios = audio.unsqueeze(0)
        # audios = audio
        model_input = audio_batch.to(device)

        for idx, layer in enumerate(self.encoder.module.fullmodel):
            context, z = layer.get_latents(model_input)
            model_input = z.permute(0, 2, 1)

            if(idx == self.layer_depth - 1):
                return z.permute(0, 2, 1) # swap channels and depth

# %%

if __name__=="__main__":

    from options import OPTIONS as opt
    # from data.de_boer_decoder_sounds import DeBoerDecoderDataset
    # import os


    encoder = GIM_Encoder(opt)
    org_batch  = torch.rand((64, 1, 10240))
    enc = encoder.encode(org_batch)



#     print("Using Train+Val / Test Split")
#     train_dataset = DeBoerDecoderDataset(
#         encoder,
#         opt=opt,
#         root=os.path.join(
#             opt["data_input_dir"],
#             "corpus",
#         ),
#         directory="train"
#     )

#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=opt["batch_size_multiGPU"],
#         shuffle=True,
#         drop_last=True,
#         num_workers=1,
#     )


#     d = next(iter(train_loader))
#     # %%
#     d
# # %%
