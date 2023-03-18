import torch
import torch.nn as nn

from models.independent_module import IndependentModule
b = 99
device = torch.device("cpu")


kernel_sizes = [10, 8, 3]
strides = [4, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = 8
max_pool_stride = 4

# kernel_sizes = [10, 8, 4, 4, 4]
# strides = [5, 4, 2, 2, 2]
# padding = [2, 2, 2, 2, 1]
# max_pool_k_size = None
# max_pool_stride = None

cnn_hidden_dim = 32
regressor_hidden_dim = 16

ARCHITECTURE = {  # given inp: (-1, 1, 8800), out: (-1, 32, 20)
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'predict_distributions': True
}

opt = {'auto_regressor_after_module': False, 'negative_samples': 10, 'prediction_step': 12, 'subsample': True, 'device': device, 'batch_size': b, 'architecture': ARCHITECTURE,
       'kld_weight': 0}


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

x = torch.randn(b, 1, 8800)  # de boer sounds: single syllable
x = torch.randn(b, 1, 20480)  # librispeech default length
# x = torch.randn(b, 55, 512)
y = model(x)


# current model: 20480 -> 128
# simple v1: 20480 -> 49
# simple v2: 20480 -> 105
