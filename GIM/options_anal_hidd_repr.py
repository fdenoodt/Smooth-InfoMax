# old model that was trained on larger samples:
# 

ENCODER_MODEL_DIR = r"DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_test_w_ar"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise_no_autoreg"
LOG_PATH = f"{ENCODER_MODEL_DIR}/analyse_hidden_repr/"
EPOCH_VERSION =  360 # 199
AUTO_REGRESSOR_AFTER_MODULE = False

# ACTIONS
SAVE_ENCODINGS = False
VISUALISE_LATENT_ACTIVATIONS = False
VISUALISE_TSNE = True

VISUALISE_TSNE_ORIGINAL_DATA = False


# warning for the old gim model, set in options.py:
# 'train_layer': 6,
# 'model_splits': 6,
# 'auto_regressor_after_module': False
