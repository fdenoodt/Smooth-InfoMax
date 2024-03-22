# %%
from models import full_model
from utils import model_utils
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GIM_Encoder():
    def __init__(self, opt, path='./g_drive_model/model_180.ckpt') -> None:
        self.opt = opt

        self.encoder: full_model.FullModel = self.load_model(path)[0]
        self.encoder.eval()

    def __call__(self, xs_batch) -> torch.tensor:
        with torch.no_grad():
            return self.encode(xs_batch)

    def load_model(self, path):
        # Originates from: def load_model_and_optimizer()

        calc_accuracy = False
        num_GPU = None

        # Initialize model.
        model: full_model.FullModel = full_model.FullModel(
            self.opt,
            calc_accuracy=calc_accuracy,
        )

        model, num_GPU = model_utils.distribute_over_GPUs(
            self.opt, model, num_GPU=num_GPU)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.opt['learning_rate'])
        model.load_state_dict(torch.load(path,
                                         map_location=device
                                         ))

        return model, optimizer

    def encode(self, audio_batch):
        latent_per_module = []

        model_input = audio_batch.to(device)

        for idx, module in enumerate(self.encoder.module.fullmodel):
            latent, _ = module.get_latents(model_input)  # returns a latent representation

            latent_per_module.append(latent)

            model_input = latent.permute(0, 2, 1)

        return latent_per_module  # out: b, l, c


# %%


if __name__ == "__main__":
    from configs import OPTIONS as opt

    # from data.xxxx_decoder_sounds import DeBoerDecoderDataset
    # import os

    encoder = GIM_Encoder(opt)
    org_batch = torch.rand((64, 1, 10240))
    enc = encoder(org_batch)
    # enc = encoder.encode(org_batch)

#     print("Using Train+Val / Test Split")
#     train_dataset = DeBoerDecoderDataset(
#         encoder,
#         opt=opt,
#         root=os.path.join(
#             opt.data_input_dir,
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
