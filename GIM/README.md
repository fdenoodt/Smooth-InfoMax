# Variational Greedy InfoMax

Code based on Greedy Infomax GitHub repository.




Example options.py file
```python
import torch

# EXPERIMENT_NAME = 'audio_experiment_test_w_ar'
# EXPERIMENT_NAME = 'audio_experiment_vae_zero'
# EXPERIMENT_NAME = 'de_boer_reshuf_simple_v2_kld_weight=0.00'
# EXPERIMENT_NAME = 'de_boer_reshuf_simple_v2_kld_weight=0.0033 !!'

# WARNING: CURRENT BUG: THIS NAME SHOULD BE THE SAME AS WHERE CPC LOCATION,
# if not: inconsitent results
# (see options_autoencoder.py > `gim_model_path`) 
EXPERIMENT_NAME = 'de_boer_reshuf_simple_v2_TWO_MODULES_kld_weight=0.0033_latent_dim=32 !!'
NUM_EPOCHS = 1
START_EPOCH = 0
AUTO_REGRESSOR_AFTER_MODULE = False
BATCH_SIZE = 171

ROOT_LOGS = r"D:\thesis_logs\logs/"

# Original dimensions given in CPC paper (Oord et al.).
# kernel_sizes = [10, 8, 4, 4, 4] # 20480 -> 128
# strides = [5, 4, 2, 2, 2]
# padding = [2, 2, 2, 2, 1]
# max_pool_stride = None
# max_pool_k_size = None


# simple architecture v1 # 20480 -> 49
# kernel_sizes = [10, 10, 3]
# strides = [5, 5, 1]
# padding = [0, 0, 1]
# max_pool_k_size = 8
# max_pool_stride = 4

# Simple architecture v2 # 20480 -> 105
kernel_sizes = [10, 8, 3]
strides = [4, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = 8
max_pool_stride = 4

cnn_hidden_dim = 32  # TODO: CHANGED FROM 32
regressor_hidden_dim = 16  # TODO: CHANGED FROM 16

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


ARCHITECTURE2 = {  # TODO: changed
    'max_pool_k_size': max_pool_k_size,
    'max_pool_stride': max_pool_stride,
    'kernel_sizes': kernel_sizes,
    'strides': strides,
    'padding': padding,
    'cnn_hidden_dim': cnn_hidden_dim,
    'regressor_hidden_dim': regressor_hidden_dim,
    'prediction_step': 4,  # latents only have length 5 so this is the max
}

# ARCHITECTURE2 = {
#     'max_pool_k_size': max_pool_k_size,
#     'max_pool_stride': max_pool_stride,
#     'kernel_sizes': kernel_sizes,
#     'strides': strides,
#     'padding': padding,
#     'cnn_hidden_dim': cnn_hidden_dim,
#     'regressor_hidden_dim': regressor_hidden_dim,
#     'prediction_step': 4,  # latents only have length 5 so this is the max
# }

LEARNING_RATE = 0.01  # 0.003 # old: 0.0001
DECAY_RATE = 0.99
KLD_WEIGHT = 0.0033  # 0.0025
TRAIN_W_NOISE = False

# de_boer_sounds OR librispeech OR de_boer_sounds_reshuffled
DATA_SET = 'de_boer_sounds_reshuffled'
SPLIT_IN_SYLLABLES = False
PERFORM_ANALYSIS = True


def get_options(experiment_name):
    options = {
        'num_epochs': NUM_EPOCHS,
        'seed': 2,
        'data_input_dir': './datasets/',
        'validate': True,
        'negative_samples': 10,
        'subsample': True,
        'loss': 0,

        'train_layer': 2,
        'model_splits': 2,
        'predict_distributions': predict_distributions,
        'architecture_module_1': ARCHITECTURE,
        'architecture_module_2': ARCHITECTURE2,
        'kld_weight': KLD_WEIGHT,

        'learning_rate': LEARNING_RATE,
        'decay_rate': DECAY_RATE,

        'train_w_noise': TRAIN_W_NOISE,
        'split_in_syllables': SPLIT_IN_SYLLABLES,

        # is for intermediate layers, by default this is set to false, only last layer has auto regressor
        # (this is always the case, regardless of what this param is set to)
        'use_autoregressive': False,
        'remove_BPTT': False,
        'model_num': '',
        'model_type': 0,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

        'experiment': 'audio',
        'save_dir': experiment_name,
        'log_path': f'{ROOT_LOGS}/{experiment_name}',
        'log_path_latent': f'{ROOT_LOGS}/{experiment_name}/latent_space',

        'log_every_x_epochs': 1,

        'model_path': f'{ROOT_LOGS}/{experiment_name}/',
        'start_epoch': START_EPOCH,

        'data_set': DATA_SET,
        'batch_size_multiGPU': BATCH_SIZE,  # 22,
        'batch_size': BATCH_SIZE,
        'auto_regressor_after_module': AUTO_REGRESSOR_AFTER_MODULE,


        'perform_analysis': PERFORM_ANALYSIS,
        # options for analysis
        'ANAL_LOG_PATH': f'{ROOT_LOGS}/{experiment_name}/analyse_hidden_repr/',
        'ANAL_ENCODER_MODEL_DIR': f"{ROOT_LOGS}/{experiment_name}",
        'ANAL_EPOCH_VERSION': START_EPOCH + NUM_EPOCHS - 1,
        'ANAL_AUTO_REGRESSOR_AFTER_MODULE': AUTO_REGRESSOR_AFTER_MODULE,
        'ANAL_ONLY_LAST_PREDICTION_FROM_TIME_WINDOW': False,

        'ANAL_SAVE_ENCODINGS': True,
        'ANAL_VISUALISE_LATENT_ACTIVATIONS': False,
        'ANAL_VISUALISE_TSNE': True,
        'ANAL_VISUALISE_TSNE_ORIGINAL_DATA': False,
        'ANAL_VISUALISE_HISTOGRAMS': True
    }
    return options


# simplified architecture
OPTIONS = get_options(EXPERIMENT_NAME)

if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")

```

