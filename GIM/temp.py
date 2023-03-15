import torch
import torch.nn as nn

from models.independent_module import IndependentModule

device = torch.device("cpu")

ARCHITECTURE = { # given inp: (-1, 1, 8800), out: (-1, 32, 20)
    'max_pool': True,
    'kernel_sizes': [10, 10, 3],
    'strides': [5, 5, 1],
    'padding': [0, 0, 1],
    'cnn_hidden_dim': 32,
    'regressor_hidden_dim': 16,
}

opt = {'auto_regressor_after_module': False, 'negative_samples': 10, 'prediction_step': 12, 'subsample': True, 'device': device, 'batch_size':128, 'architecture': ARCHITECTURE}
7

kernel_sizes = [10, 10, 3]
strides = [5, 5, 1]
padding = [0, 0, 1]
cnn_hidden_dim = 32
regressor_hidden_dim = 16
calc_accuracy = False

model = IndependentModule(
        opt,
        enc_kernel_sizes=kernel_sizes,  # [10, 8, 4, 4, 4]
        enc_strides=strides,  # [5, 4, 2, 2, 2]
        enc_padding=padding,  # [2, 2, 2, 2, 1]
        nb_channels_cnn=cnn_hidden_dim,  # 512
        nb_channels_regress=regressor_hidden_dim,  # 256
        calc_accuracy=calc_accuracy,
    ).to(device)


print(model)

x = torch.randn(128, 1, 8800)
# x = torch.randn(128, 55, 512)
y = model(x)
print(y)