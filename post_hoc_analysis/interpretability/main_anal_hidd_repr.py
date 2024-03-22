# %%
"""
This file is used to analyse the hidden representation of the audio signal.
- It stores the hidden representation of the audio signal for each batch in a tensor.
- The tensor is then visualised using a scatter plot.
"""
import random
from sklearn.manifold import TSNE
from utils.helper_functions import *
from options_anal_hidd_repr import OPTIONS as OPT_ANAL
from arg_parser import arg_parser
from data import get_dataloader

random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualise_2d_tensor(tensor, GIM_model_name, target_dir, name):
    nd_arr = tensor.to('cpu').numpy()
    length, nb_channels = nd_arr.shape

    nd_arr_flat = nd_arr.flatten()  # (length * nb_channels)
    s = nd_arr_flat / np.max(nd_arr_flat)
    xs = np.repeat(np.arange(0, length, 1), nb_channels)  # length
    ys = np.tile(np.arange(0, nb_channels, 1), length)  # channels

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(ys, xs, s=100*(s**4), marker="s", c='orange', alpha=0.3)
    ax.set_aspect('equal')

    ax.set_xlabel('Channels')
    ax.set_ylabel('Signal length')
    ax.set_title(
        f'Hidden representation of the audio signal - {GIM_model_name} - {name}')

    # Show the plot
    plt.savefig(f"{target_dir}/{name}.png")
    # plt.show()


def _save_encodings(opt_anal, root_dir, data_type, encoder: GIM_Encoder, data_loader):
    assert data_type in ["train", "test"]

    # audio, filename, pronounced_syllable, full_word
    for idx, (batch_org_audio, filenames, pronounced_syllable, _) in enumerate(iter(data_loader)):
        batch_org_audio = batch_org_audio.to(device)
        batch_enc_audio_per_module = encoder(batch_org_audio)

        for module_idx, batch_enc_audio in enumerate(batch_enc_audio_per_module):
            # eg: batch_enc_audio.shape = (96, 55, 256)

            if opt_anal['ONLY_LAST_PREDICTION_FROM_TIME_WINDOW']:
                batch_enc_audio = batch_enc_audio[:, -1, :]
                # eg: batch_enc_audio.shape = (96, 256)
                # Expand, eg: batch_enc_audio.shape = (96, 1, 256)
                batch_enc_audio = batch_enc_audio.unsqueeze(1)

            # eg: 01GIM_L{layer_depth}/module=1/train/
            target_dir = f"{root_dir}/module={module_idx + 1}/{data_type}/"
            create_log_dir(target_dir)

            print(
                f"Batch {idx} - {batch_enc_audio.shape} - Mean: {torch.mean(batch_enc_audio)} - Std: {torch.std(batch_enc_audio)}")

            torch.save(batch_enc_audio,
                       f"{target_dir}/batch_encodings_{idx}.pt")
            torch.save(filenames, f"{target_dir}/batch_filenames_{idx}.pt")
            torch.save(pronounced_syllable,
                       f"{target_dir}/batch_pronounced_syllable_{idx}.pt")
            torch.save(batch_org_audio,
                       f"{target_dir}/batch_org_audio_{idx}.pt")


def generate_and_save_encodings(opt_enc, opt_anal, encoder_model_path):
    encoder: GIM_Encoder = GIM_Encoder(opt_enc, path=encoder_model_path)
    split = True
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt_enc, dataset="xxxx_sounds_reshuffled", shuffle=False, split_and_pad=split, train_noise=False)

    target_dir = f"{opt_anal['LOG_PATH']}/hidden_repr/{'split' if split else 'full'}"

    _save_encodings(opt_anal, target_dir, "train", encoder, train_loader)
    _save_encodings(opt_anal, target_dir, "test", encoder, test_loader)


def _generate_visualisations(data_dir, GIM_model_name, target_dir):
    # iterate over files in train_dir
    for file in os.listdir(data_dir):  # Generated via copilot
        if file.endswith(".pt") and file.startswith("batch_encodings"):
            # load the file
            batch_encodings = torch.load(f"{data_dir}/{file}")
            batch_filenames = torch.load(
                f"{data_dir}/{file.replace('encodings', 'filenames')}")
            try:
                labels = torch.load(
                    f"{data_dir}/{file.replace('encodings', 'pronounced_syllable')}").numpy()
            except FileNotFoundError:
                # list of Nones, where N is the number of files in the batch
                labels = [
                    None] * len(batch_filenames)

            # iterate over the batch
            for idx, (enc, name, pronounced_syllable_idx) in enumerate(zip(batch_encodings, batch_filenames, labels)):
                name = name.split("_")[0]  # eg: babugu_1 -> babugu
                if pronounced_syllable_idx is not None:  # simple check to deal with split/full audio files
                    pronounced_syllable = translate_number_to_syllable(
                        pronounced_syllable_idx)
                    name = f"{name} - {pronounced_syllable}"

                visualise_2d_tensor(enc, GIM_model_name, target_dir, f"{name}")

                if idx > 2:  # only do 2 visualisations per batch. So if 17 batches, 34 visualisations
                    break









def generate_tsne_visualisations_original_data(train_or_test):
    assert train_or_test in ["train", "test"]
    target_dir = rf"./datasets\corpus\split up graphs\{train_or_test}"

    all_audios = None
    all_syllables = np.array([])  # indicies

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt, dataset="xxxx_sounds", shuffle=False, split_and_pad=True, train_noise=False)

    for idx, (batch_org_audio, filenames, syllable_idxs, _) in enumerate(iter(train_loader if train_or_test == "train" else test_loader)):
        # eg: batch_org_audio.shape = (96, 1, 8800)
        if idx == 0:  # obtain intial shape from first batch.
            # (batch_size, length, nb_channels)
            all_audios = torch.empty(0, batch_org_audio.size(
                1), batch_org_audio.size(2)).cpu()

        all_audios = torch.cat((all_audios, batch_org_audio), dim=0).cpu()
        all_syllables = np.concatenate((all_syllables, syllable_idxs))

    all_audios = all_audios.numpy()
    batch_size = all_audios.shape[0]
    all_audios = np.reshape(all_audios, (batch_size, -1))
    plot_tsne(all_audios, all_syllables, "Original data", target_dir)


def _visualise_latent_space_tsne(data_dir, gim_name, target_dir):

    all_cs = None
    all_labels = np.array([])  # indicies
    all_audio = None

    # iterate over files in train_dir
    for idx, file in enumerate(os.listdir(data_dir)):
        # eg: batch_encodings_0.pt
        if file.endswith(".pt") and file.startswith("batch_encodings"):
            # load the file
            # (batch_size, length, nb_channels)
            batch_encodings = torch.load(f"{data_dir}/{file}").cpu()

            labels = torch.load(
                f"{data_dir}/{file.replace('encodings', 'pronounced_syllable')}").numpy()

            audio = torch.load(
                f"{data_dir}/{file.replace('encodings', 'org_audio')}").cpu()


            if idx == 0:  # obtain intial shape from first batch.
                all_cs = torch.empty(0, batch_encodings.size(
                    1), batch_encodings.size(2)).cpu()
                all_audio = torch.empty(0, audio.size(1), audio.size(2)).cpu()

            # merge the batch to a single tensor
            all_cs = torch.cat((all_cs, batch_encodings), dim=0)
            all_labels = np.concatenate((all_labels, labels))
            all_audio = torch.cat((all_audio, audio), dim=0)

    # (b, l, c)
    b, l, c = all_cs.shape

    assert c == 32 or c == 16 or c == 512 or c == 256
    # assert l == 4 or l == 44

    all_cs = all_cs.permute(0, 2, 1)  # (b, c, l)

    # pool
    # m = nn.MaxPool1d(10, stride=10)
    # all_cs = m(all_cs)  # (b, c, l)
    # all_cs = nn.functional.adaptive_max_pool1d(all_cs, 1)  # (b, c, 1)
    # all_cs = nn.functional.adaptive_avg_pool1d(all_cs, 1)  # (b, c, 1)

    all_cs = all_cs.permute(0, 2, 1).reshape(b, -1)  # (b, l, c) -> (b, c)
    all_cs = all_cs.numpy()

    b, c = all_cs.shape
    assert b == all_labels.shape[0]


    plot_tsne(all_cs,
              all_labels,
              gim_name, target_dir)


def _visualise_latent_space_histograms(data_dir, gim_name, target_dir):

    all_encs = None

    # iterate over files in train_dir
    for idx, file in enumerate(os.listdir(data_dir)):
        if file.endswith(".pt") and file.startswith("batch_encodings"):
            # load the file

            # (b, l, c)
            encs = torch.load(f"{data_dir}/{file}").cpu()

            if idx == 0:  # obtain intial shape from first batch.
                all_encs = torch.empty(0, encs.size(1), encs.size(2)).cpu()

            # merge the batch to a single tensor
            all_encs = torch.cat((all_encs, encs), dim=0)

    all_encs = encs.permute(2, 0, 1)  # (b, c, l) --> (c, b, l)
    all_encs = all_encs.numpy()

    nb_channels = all_encs.shape[0]  # 32
    assert nb_channels == 32 or nb_channels == 8 or nb_channels == 16 or nb_channels == 512 or nb_channels == 256
    all_encs = np.reshape(all_encs, (nb_channels, -1))  # (32, b*l)

    plot_histograms(all_encs, gim_name, target_dir)


def generate_visualisations(opt_anal):
    # eg LOG_PATH = ./GIM\logs\audio_experiment_3_lr_noise\analyse_hidden_repr\
    for split in ['split', 'full']:
        if split == 'full':  # TODO: temporary disabled as full is not yet implemented
            continue

        saved_modules_dir = f"{opt_anal['LOG_PATH']}/hidden_repr/{split}/"
        nb_modules = len(os.listdir(saved_modules_dir))  # module=1, ...


        # only visualise last 3 modules. The earlier latents are too high dimension and cannot be stored in memory
        first_module = max(nb_modules - 3, 1)

        for module_idx in range(first_module, nb_modules + 1):

            saved_files_dir = f"{opt_anal['LOG_PATH']}/hidden_repr/{split}/module={module_idx}/"

            train_dir = f"{saved_files_dir}/train/"
            test_dir = f"{saved_files_dir}/test/"

            target_dir = f"{opt_anal['LOG_PATH']}/hidden_repr_vis/{split}/module={module_idx}/"
            train_vis_dir = f"{target_dir}/train"
            test_vis_dir = f"{target_dir}/test/"
            create_log_dir(train_vis_dir)
            create_log_dir(test_vis_dir)

            if opt_anal['VISUALISE_LATENT_ACTIVATIONS']:
                _generate_visualisations(train_dir, "GIM", train_vis_dir)
                _generate_visualisations(test_dir, "GIM", test_vis_dir)

            if opt_anal['VISUALISE_TSNE'] and split == 'split':
                _visualise_latent_space_tsne(test_dir, "GIM", test_vis_dir)
                _visualise_latent_space_tsne(train_dir, "GIM", train_vis_dir)

            if opt_anal['VISUALISE_HISTOGRAMS']:
                _visualise_latent_space_histograms(
                    test_dir, "GIM", test_vis_dir)
                _visualise_latent_space_histograms(
                    train_dir, "GIM", train_vis_dir)


def run_visualisations(opt, opt_anal):

    arg_parser.create_log_path(opt)
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']
    opt['auto_regressor_after_module'] = opt_anal['AUTO_REGRESSOR_AFTER_MODULE']

    logs = logger.Logger(opt)

    ENCODER_NAME = f"model_{opt_anal['EPOCH_VERSION']}.ckpt"
    ENCODER_MODEL_PATH = f"{opt_anal['ENCODER_MODEL_DIR']}/{ENCODER_NAME}"

    if opt_anal['SAVE_ENCODINGS']:
        generate_and_save_encodings(opt, opt_anal, ENCODER_MODEL_PATH)

    if opt_anal['VISUALISE_LATENT_ACTIVATIONS'] or opt_anal['VISUALISE_TSNE'] or opt_anal['VISUALISE_HISTOGRAMS']:
        generate_visualisations(opt_anal)

    if opt_anal['VISUALISE_TSNE_ORIGINAL_DATA']:
        generate_tsne_visualisations_original_data("train")
        generate_tsne_visualisations_original_data("test")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    assert OPT_ANAL['SAVE_ENCODINGS'] or OPT_ANAL['VISUALISE_LATENT_ACTIVATIONS'] or \
        OPT_ANAL['VISUALISE_TSNE'] or OPT_ANAL['VISUALISE_TSNE_ORIGINAL_DATA'] or \
        OPT_ANAL['VISUALISE_HISTOGRAMS'], "Nothing to do"

    run_visualisations(OPT, OPT_ANAL)

    torch.cuda.empty_cache()
