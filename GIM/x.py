# %%
# from models.GIM_encoder import GIM_Encoder
import torch

data = torch.load("C:\\GitHub\\thesis-fabian-denoodt\\GIM\\g_drive_model\\model_180.ckpt", 
                                        map_location=torch.device('cuda')
                                        #  map_location=device
                                         )
data

# %%
data.shape

# x = GIM_Encoder()

