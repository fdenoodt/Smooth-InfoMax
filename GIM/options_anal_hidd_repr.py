# old model that was trained on larger samples:
# DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/

# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_test_w_ar"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise"
ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise_no_autoreg"
LOG_PATH = f"{ENCODER_MODEL_DIR}/analyse_hidden_repr/"
EPOCH_VERSION = 199
AUTO_REGRESSOR_AFTER_MODULE = False

SAVE_ENCODINGS = False
VISUALISE_LATENT_ACTIVATIONS = False
VISUALISE_TSNE = True
