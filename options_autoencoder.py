from decoder.decoder_architectures import MEL_LOSS, SimpleV2DecoderTwoModules

EXPERIMENT_NAME = "GIM_DECODER_simple_v2_TWO_MODULES"


def get_options(experiment_name):
    lr = 0.005
    decay_rate = 0.995
    n_fft = 4096
    num_epochs = 500
    options = {
        'experiment_name': experiment_name,
        'lr': lr,
        'decay_rate': decay_rate,
        'criterion': MEL_LOSS(n_fft=n_fft),
        # "DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt"
        # 'gim_model_path': r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_kld_weight=0.0033 !!/model_290.ckpt",
        'gim_model_path': r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_TWO_MODULES_kld_weight=0.0033_latent_dim=32 !!/model_799.ckpt",
        'decoder': SimpleV2DecoderTwoModules(),
        'num_epochs': num_epochs,
    }
    return options


OPTIONS = get_options(EXPERIMENT_NAME)
