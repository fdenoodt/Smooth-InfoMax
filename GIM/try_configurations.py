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
    EXPERIMENT_NAME = 'VGIM_kld_weight=0.0035_diff_k'

    # kld_weights = [0.0033, 0.0]
    # kld_weights = [0.0033, 0, 0.0035]

    decay_rates = [1, 0.995, 0.999, 0.99, 0.9]
    for decay_rate in decay_rates:
        try:
            final_name = f"{EXPERIMENT_NAME}_decay_rate={decay_rate}"
            OPTIONS = get_options(final_name)
            # OPTIONS['kld_weight'] = kld_weight
            OPTIONS['decay_rate'] = decay_rate

            run_cpc_train_configuration(OPTIONS)

            # options_autoencoder = get_autoencoder_options(
            #     f"{final_name}", final_name, OPTIONS['num_epochs'] - 1)

            # experiment_name = options_autoencoder["experiment_name"]
            # GIM_MODEL_PATH = options_autoencoder["gim_model_path"]
            # lr = options_autoencoder["lr"]
            # decay_rate = options_autoencoder["decay_rate"]
            # criterion = options_autoencoder["criterion"]
            # decoder = options_autoencoder["decoder"]
            # num_epochs = options_autoencoder["num_epochs"]

            # run_autoencoder_train_configuration(OPTIONS, experiment_name, GIM_MODEL_PATH, lr,
            #                                     decay_rate, criterion, decoder, num_epochs)

        except Exception as e:
            print("********************************************")
            print(f"Error: {e}, {final_name}")
            print("********************************************")
