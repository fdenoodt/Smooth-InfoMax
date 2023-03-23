import torch
CPC_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_kld_weight=0.0033 !!/model_290.ckpt"
DECODER_MODEL_PATH = r"D:\thesis_logs\logs\GIM_DECODER_simple_v2_experiment\MEL_SPECTR_n_fft=4096 !!\lr_0.0050000\GIM_L1\model_99.pt"

# Simple architecture v2 # 20480 -> 105
kernel_sizes = [10, 8, 3]
strides = [4, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = 8
max_pool_stride = 4

cnn_hidden_dim = 32  # 512
regressor_hidden_dim = 16  # 256

predict_distributions = True

ARCHITECTURE = {
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'predict_distributions': predict_distributions
}
AUTO_REGRESSOR_AFTER_MODULE = False

BATCH_SIZE = 171


def get_options():
    options = {
        'device':  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'cpc_model_path': CPC_MODEL_PATH,
        'decoder_model_path': DECODER_MODEL_PATH,
        'architecture': ARCHITECTURE,
        'train_layer': 1,
        'model_splits': 1,
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
