# %%
from decoder_architectures import SimpleV3DecoderTwoModules, MEL_LOSS
from options import get_options
from main import run_configuration as run_cpc_train_configuration
from train_autoencoder import run_configuration as run_autoencoder_train_configuration


def get_autoencoder_options(experiment_name, decoder_name, enc_version):
    lr = 0.005
    decay_rate = 0.995
    n_fft = 4096
    num_epochs = 2
    criterion = MEL_LOSS(n_fft=n_fft)
    decoder = SimpleV3DecoderTwoModules()
    gim_model_path = rf"D:\thesis_logs\logs\{decoder_name}/model_{enc_version}.ckpt"

    options = {
        'experiment_name': experiment_name,
        'lr': lr,
        'decay_rate': decay_rate,
        'criterion': criterion,
        'gim_model_path': gim_model_path,
        'decoder': decoder,
        'num_epochs': num_epochs,
    }
    return options


if __name__ == "__main__":
    EXPERIMENT_NAME = 'de_boer_TWO_MODULE_V3'

    kld_weights = [0.0033, 0.0]
    # kld_weights = [0.0033, 0, 0.0035]

    for kld_weight in kld_weights:
        # try:
        final_name = f"{EXPERIMENT_NAME}_kld_weight={kld_weight}"
        OPTIONS = get_options(final_name)
        OPTIONS['kld_weight'] = kld_weight

        # run_cpc_train_configuration(OPTIONS)

        options_autoencoder = get_autoencoder_options(
            f"{final_name}", final_name, OPTIONS['num_epochs'] - 1)

        experiment_name = options_autoencoder["experiment_name"]
        GIM_MODEL_PATH = options_autoencoder["gim_model_path"]
        lr = options_autoencoder["lr"]
        decay_rate = options_autoencoder["decay_rate"]
        criterion = options_autoencoder["criterion"]
        decoder = options_autoencoder["decoder"]
        num_epochs = options_autoencoder["num_epochs"]

        run_autoencoder_train_configuration(OPTIONS, experiment_name, GIM_MODEL_PATH, lr,
                                            decay_rate, criterion, decoder, num_epochs)

        # except Exception as e:
        #     print("********************************************")
        #     print(f"Error: {e}, {kld_weight}")
        #     print("********************************************")
