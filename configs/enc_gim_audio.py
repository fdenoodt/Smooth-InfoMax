import torch

from encoder.architecture_config import ArchitectureConfig, ModuleConfig

NUM_EPOCHS = 4
START_EPOCH = 0
AUTO_REGRESSOR_AFTER_MODULE = False
BATCH_SIZE = 8

ROOT_LOGS = r"C:\\sim_logs\\"

# Original dimensions given in CPC paper (Oord et al.).
kernel_sizes = [10, 8, 4, 4, 4] # 20480 -> 128
strides = [5, 4, 2, 2, 2]
padding = [2, 2, 2, 2, 1]
max_pool_stride = None
max_pool_k_size = None
cnn_hidden_dim = 512
predict_distributions = False

# Splits each layer into a separate module
modules = ModuleConfig.get_modules_from_list(kernel_sizes, strides, padding, cnn_hidden_dim, predict_distributions)
ARCHITECTURE = ArchitectureConfig(modules=modules)


LEARNING_RATE = 2e-4  # 0.01  # 0.003 # old: 0.0001
DECAY_RATE = 0.99 # no decay: 1.0
KLD_WEIGHT = 0.0033 # 0.0033  # 0.0025

# de_boer_sounds OR librispeech OR de_boer_sounds_reshuffled
DATA_SET = 'librispeech'
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

        'architecture': ARCHITECTURE,
        'kld_weight': KLD_WEIGHT,

        'learning_rate': LEARNING_RATE,
        'decay_rate': DECAY_RATE,

        'train_w_noise': False,
        'split_in_syllables': SPLIT_IN_SYLLABLES,

        'model_num': '', # for loading a specific model from a specific epoch and continue training
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


        'perform_analysis': PERFORM_ANALYSIS,
        # configs for analysis
        'ANAL_LOG_PATH': f'{ROOT_LOGS}/{experiment_name}/analyse_hidden_repr/',
        'ANAL_ENCODER_MODEL_DIR': f"{ROOT_LOGS}/{experiment_name}",
        'ANAL_EPOCH_VERSION': START_EPOCH + NUM_EPOCHS - 1,
        'ANAL_AUTO_REGRESSOR_AFTER_MODULE': AUTO_REGRESSOR_AFTER_MODULE,
        'ANAL_ONLY_LAST_PREDICTION_FROM_TIME_WINDOW': False,

        'ANAL_SAVE_ENCODINGS': True,
        'ANAL_VISUALISE_LATENT_ACTIVATIONS': False,
        'ANAL_VISUALISE_TSNE': True,
        'ANAL_VISUALISE_TSNE_ORIGINAL_DATA': False,
        'ANAL_VISUALISE_HISTOGRAMS': False # TODO
    }
    return options




if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
