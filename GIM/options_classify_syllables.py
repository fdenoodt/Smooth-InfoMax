import torch

# Pair w/ split = 1, architecture v2

# Two modules, version 3 architecture, 1.6k epochs
CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!/model_1599.ckpt"


# Simple architecture # 20480 -> 105
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
# kernel_sizes = [8, 8, 3]
# strides = [3, 3, 1]
# padding = [2, 2, 1]
# max_pool_k_size = None
# max_pool_stride = None

# v3
kernel_sizes = [6, 6, 3]
strides = [2, 2, 1]
padding = [2, 2, 1]
max_pool_k_size = None
max_pool_stride = None

ARCHITECTURE2 = {
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'prediction_step': 12,  # TODO
}

AUTO_REGRESSOR_AFTER_MODULE = False

BATCH_SIZE = 171  # 8

ROOT_LOGS = r"E:\thesis_logs\logs/"


def get_options():
    options = {
        'device':  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'cpc_model_path': CPC_MODEL_PATH,
        'predict_distributions': predict_distributions,
        'architecture_module_1': ARCHITECTURE,
        'architecture_module_2': ARCHITECTURE2,
        'train_layer': 2,  # TODO
        'model_splits': 2,  # TODO
        'auto_regressor_after_module': AUTO_REGRESSOR_AFTER_MODULE,

        'prediction_step': 12,
        'negative_samples': 10,
        'subsample': True,
        'loss': 0,
        'batch_size_multiGPU': BATCH_SIZE,
        'learning_rate': 0.01,  # 0.005, 50 epochs ging tot 38%
        'data_input_dir': './datasets/',
        'root_logs': ROOT_LOGS,
        'validate': True,
        'start_epoch': 0,
        'num_epochs': 150,

        'batch_size': BATCH_SIZE,  # only used if "all", else overwritten
        'subset': 'all'  # 1, 2, .. 'all'
    }
    return options
