# old model that was trained on larger samples:
# ENCODER_MODEL_DIR = r"DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/"

# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_test_w_ar"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise_no_autoreg" # w/o auto reg
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise"  # w/ auto reg
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise"  # w/ auto reg
ENCODER_MODEL_DIR = r"E:\thesis_logs\logs\temp2"  # w/ auto reg

# ENCODER_MODEL_DIR = r"E:\\thesis_logs\\logs//libri_simple_v2_kld_weight=0.00"  # w/ auto reg
# ENCODER_MODEL_DIR = r"E:\\thesis_logs\\logs//libri_cpc_kld_weight=0.00"  # w/ auto reg
# ENCODER_MODEL_DIR = r"D:\\thesis_logs\\logs//de_boer_reshuf_simple_v2_kld_weight=0.0033 !!"
# ENCODER_MODEL_DIR = r"D:\thesis_logs\logs\\de_boer_reshuf_simple_v2_TWO_MODULES_kld_weight=0.0000_latent_dim=32 !!"
# ENCODER_MODEL_DIR = r"D:\thesis_logs\logs\\de_boer_reshuf_simple_v2_TWO_MODULES_kld_weight=0.0033_latent_dim=32 !!"

# ENCODER_MODEL_DIR = r"D:\thesis_logs\logs\good models\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!"
# ENCODER_MODEL_DIR = r"D:\thesis_logs\logs\good models\de_boer_TWO_MODULE_V3_dim32_kld_weight=0"


LOG_PATH = f"{ENCODER_MODEL_DIR}/analyse_hidden_repr/"
EPOCH_VERSION = 257
AUTO_REGRESSOR_AFTER_MODULE = False

# Actions
SAVE_ENCODINGS = True # TODO
VISUALISE_LATENT_ACTIVATIONS = False
VISUALISE_TSNE = True
VISUALISE_TSNE_ORIGINAL_DATA = False
VISUALISE_HISTOGRAMS = False

# warning: only makes sense for outputs of GRU
ONLY_LAST_PREDICTION_FROM_TIME_WINDOW = False

# warning for the old gim model, set in options.py:
# 'train_layer': 6,
# 'model_splits': 6,
# 'auto_regressor_after_module': False


OPTIONS = {
    'LOG_PATH': LOG_PATH,
    'EPOCH_VERSION': EPOCH_VERSION,
    'ONLY_LAST_PREDICTION_FROM_TIME_WINDOW': ONLY_LAST_PREDICTION_FROM_TIME_WINDOW,
    'SAVE_ENCODINGS': SAVE_ENCODINGS,
    'AUTO_REGRESSOR_AFTER_MODULE': AUTO_REGRESSOR_AFTER_MODULE,
    'ENCODER_MODEL_DIR': ENCODER_MODEL_DIR,
    'VISUALISE_LATENT_ACTIVATIONS': VISUALISE_LATENT_ACTIVATIONS,
    'VISUALISE_TSNE': VISUALISE_TSNE,
    'VISUALISE_TSNE_ORIGINAL_DATA': VISUALISE_TSNE_ORIGINAL_DATA,
    'VISUALISE_HISTOGRAMS': VISUALISE_HISTOGRAMS, # TODO: ADDED
}
