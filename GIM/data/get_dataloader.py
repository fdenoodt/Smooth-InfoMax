import os
import torch
from data import de_boer_sounds, librispeech

NUM_WORKERS = 1


def _dataloaders(opt, train_specific_dir, test_specific_dir, train_sub_dir, test_sub_dir, split_and_pad, train_noise, shuffle):
    train_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"], f"corpus/{train_specific_dir}"
        ),
        directory=train_sub_dir,

        # ONLY NOISE FOR TRAINING DATASETS!
        background_noise=train_noise, white_guassian_noise=train_noise,
        background_noise_path=os.path.join(
            opt["data_input_dir"],
            "musan"),
        split_into_syllables=split_and_pad
    )

    test_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"], f"corpus/{test_specific_dir}",
        ),
        directory=test_sub_dir,
        background_noise=False, white_guassian_noise=False,
        split_into_syllables=split_and_pad
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def _get_de_boer_sounds_data_loaders(opt, reshuffled=None, split_and_pad=True, train_noise=True, shuffle=True, subset_size=None):
    ''' Retrieve dataloaders where audio signals are split into syllables '''
    print("Loading De Boer Sounds dataset...")

    if split_and_pad:
        if subset_size:
            print(f"Using subset of size {subset_size} and batch size {opt['batch_size']}")
            train_specific_directory = "subsets/"
            train_sub_dir = f"{subset_size}" # eg: subsets/all
            test_specific_directory = "split up data padded reshuffled"
            return _dataloaders(opt, train_specific_directory, test_specific_directory, train_sub_dir, "test", split_and_pad, train_noise, shuffle)
        else:
            print("************************")
            print("************************")
            print("************************")
            print("************************")
            print("************************")
            print("Using full dataset")
            # specific_directory = "split up data padded reshuffled"
            specific_directory = "split up data cropped reshuffled"
    elif reshuffled == "v1":
        specific_directory = "reshuffled"
    elif reshuffled == "v2":
        specific_directory = "reshuffledv2"
    else:
        specific_directory = ""

    return _dataloaders(opt, specific_directory, specific_directory, "train", "test", split_and_pad, train_noise, shuffle)


def _get_libri_dataloaders(opt):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
    """
    num_workers = 1

    print("Using Train+Val / Test Split")
    train_dataset = librispeech.LibriDataset(
        opt,
        os.path.join(
            opt['data_input_dir'],
            "LibriSpeech/train-clean-100",
        ),
        os.path.join(
            opt['data_input_dir'], "LibriSpeech100_labels_split/train_split.txt"
        ),
    )

    test_dataset = librispeech.LibriDataset(
        opt,
        os.path.join(
            opt['data_input_dir'],
            "LibriSpeech/train-clean-100",
        ),
        os.path.join(
            opt['data_input_dir'], "LibriSpeech100_labels_split/test_split.txt"
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt['batch_size_multiGPU'],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt['batch_size_multiGPU'],
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_dataloader(opt, dataset, **kwargs):

    if dataset == "de_boer_sounds":
        return _get_de_boer_sounds_data_loaders(opt, **kwargs)
    elif dataset == "de_boer_sounds_reshuffled":  # used for training CPC
        return _get_de_boer_sounds_data_loaders(opt, reshuffled="v1", **kwargs)
    elif dataset == "de_boer_sounds_reshuffledv2":  # used for training CPC Decoder
        return _get_de_boer_sounds_data_loaders(opt, reshuffled="v2", **kwargs)
    elif dataset == "librispeech":
        return _get_libri_dataloaders(opt)
    else:
        raise ValueError("Unknown dataset")
