# *****
# INTERPOLATE FIRST MODULE.

import torch
from decoder.decoder_architectures import SimpleV2Decoder

# Pair w/ split = 1, architecture v2
CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_kld_weight=0.0033 !!/model_290.ckpt"
DECODER_MODEL_PATH = r"D:\thesis_logs\logs\GIM_DECODER_simple_v2_experiment\MEL_SPECTR_n_fft=4096 !!\lr_0.0050000\GIM_L1\model_99.pt"



# Simple architecture v2 # 20480 -> 105
kernel_sizes = [10, 8, 3]
strides = [4, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = 8
max_pool_stride = 4

cnn_hidden_dim = 32
regressor_hidden_dim = 16

predict_distributions = True

ARCHITECTURE = {
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'prediction_step': 12,
}

# v2
kernel_sizes = [8, 8, 3]
strides = [3, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = None
max_pool_stride = None

# v3
# kernel_sizes = [6, 6, 3]
# strides = [2, 2, 1]
# padding = [2, 2, 1]
# max_pool_k_size = None
# max_pool_stride = None

ARCHITECTURE2 = {
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'prediction_step': 4,  # TODO
}

AUTO_REGRESSOR_AFTER_MODULE = False

BATCH_SIZE = 171


def get_options():
    options = {
        'device':  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'cpc_model_path': CPC_MODEL_PATH,
        'decoder_model_path': DECODER_MODEL_PATH,
        'predict_distributions': predict_distributions,
        'architecture_module_1': ARCHITECTURE,
        'architecture_module_2': ARCHITECTURE2,
        'decoder':  SimpleV2Decoder(), # TODO SimpleV3DecoderTwoModules(), 
        'train_layer': 1,  # TODO
        'model_splits': 1,  # TODO
        'auto_regressor_after_module': AUTO_REGRESSOR_AFTER_MODULE,


        # useless options required by the GIM decoder, originally meant for training
        'prediction_step': 12,
        'negative_samples': 10,
        'subsample': True,
        'loss': 0,
        'batch_size': BATCH_SIZE,
        'batch_size_multiGPU': BATCH_SIZE,
        'learning_rate': 0.01,
        'data_input_dir': './datasets/',


    }
    return options



# # *****
# # INTERPOLATE SECOND MODULE.

# import torch
# from decoder_architectures import SimpleV2Decoder, SimpleV2DecoderTwoModules, SimpleV3DecoderTwoModules

# # Pair w/ split = 1, architecture v2
# # CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_kld_weight=0.0033 !!/model_290.ckpt"
# # DECODER_MODEL_PATH = r"D:\thesis_logs\logs\GIM_DECODER_simple_v2_experiment\MEL_SPECTR_n_fft=4096 !!\lr_0.0050000\GIM_L1\model_99.pt"


# # Two modules, version 3 architecture, 1.6k epochs
# CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!/model_1599.ckpt"
# DECODER_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!\DECODER\MEL_SPECTR_n_fft=4096\lr_0.0050000\GIM_L1\model_799.pt"

# # idk
# # CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_TWO_MODULES_kld_weight=0.0033_latent_dim=32 !!/model_799.ckpt"
# # DECODER_MODEL_PATH = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\GIM_DECODER_simple_v2_TWO_MODULES_experiment\MEL_SPECTR_n_fft=4096\lr_0.0050000\GIM_L1\model_2.pt"

# # Simple architecture v2 # 20480 -> 105
# kernel_sizes = [10, 8, 3]
# strides = [4, 3, 1]
# padding = [2, 2, 1]
# max_pool_k_size = 8
# max_pool_stride = 4

# cnn_hidden_dim = 32
# regressor_hidden_dim = 16

# predict_distributions = True

# ARCHITECTURE = {
#     'max_pool_k_size': max_pool_k_size,
#     'max_pool_stride': max_pool_stride,
#     'kernel_sizes': kernel_sizes,
#     'strides': strides,
#     'padding': padding,
#     'cnn_hidden_dim': cnn_hidden_dim,
#     'regressor_hidden_dim': regressor_hidden_dim,
#     'prediction_step': 12,
# }

# # v2
# # kernel_sizes = [8, 8, 3]
# # strides = [3, 3, 1]
# # padding = [2, 2, 1]
# # max_pool_k_size = None
# # max_pool_stride = None

# # v3
# kernel_sizes = [6, 6, 3]
# strides = [2, 2, 1]
# padding = [2, 2, 1]
# max_pool_k_size = None
# max_pool_stride = None

# ARCHITECTURE2 = {
#     'max_pool_k_size': max_pool_k_size,
#     'max_pool_stride': max_pool_stride,
#     'kernel_sizes': kernel_sizes,
#     'strides': strides,
#     'padding': padding,
#     'cnn_hidden_dim': cnn_hidden_dim,
#     'regressor_hidden_dim': regressor_hidden_dim,
#     'prediction_step': 12 #4,  # TODO
# }

# AUTO_REGRESSOR_AFTER_MODULE = False

# BATCH_SIZE = 171


# def get_options():
#     options = {
#         'device':  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#         'cpc_model_path': CPC_MODEL_PATH,
#         'decoder_model_path': DECODER_MODEL_PATH,
#         'predict_distributions': predict_distributions,
#         'architecture_module_1': ARCHITECTURE,
#         'architecture_module_2': ARCHITECTURE2,
#         'decoder':  SimpleV3DecoderTwoModules(), # SimpleV2Decoder(), # TODO
#         'train_layer': 2,  # TODO
#         'model_splits': 2,  # TODO
#         'auto_regressor_after_module': AUTO_REGRESSOR_AFTER_MODULE,


#         # useless options required by the GIM decoder, originally meant for training
#         'prediction_step': 12,
#         'negative_samples': 10,
#         'subsample': True,
#         'loss': 0,
#         'batch_size': BATCH_SIZE,
#         'batch_size_multiGPU': BATCH_SIZE,
#         'learning_rate': 0.01,
#         'data_input_dir': './datasets/',


#     }
#     return options