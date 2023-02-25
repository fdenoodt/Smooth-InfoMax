import torch
import os
from GIM_encoder import GIM_Encoder

from data import librispeech
from data import de_boer_sounds
from data import de_boer_decoder_sounds

NUM_WORKERS = 1


def get_libri_dataloaders(opt):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt["validate"]
    """
    if opt["validate"]:
        print("Using Train / Val Split")
        train_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt["data_input_dir"],
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt["data_input_dir"], "LibriSpeech100_labels_split/train_val_train.txt"
            ),
        )

        test_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt["data_input_dir"],
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt["data_input_dir"], "LibriSpeech100_labels_split/train_val_val.txt"
            ),
        )

    else:
        print("Using Train+Val / Test Split")
        train_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt["data_input_dir"],
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt["data_input_dir"], "LibriSpeech100_labels_split/train_split.txt"
            ),
        )

        test_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt["data_input_dir"],
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt["data_input_dir"], "LibriSpeech100_labels_split/test_split.txt"
            ),
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_de_boer_sounds_data_loaders(opt):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt["validate"]
    """
    print("Using Train+Val / Test Split")
    train_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"],
            "corpus"),
        directory="train",

        # ONLY NOISE FOR TRAINING DATASETS!
        background_noise=True, white_guassian_noise=True,
        background_noise_path=os.path.join(
            opt["data_input_dir"],
            "musan")
    )

    test_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"],
            "corpus",
        ),
        directory="test",
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_de_boer_sounds_split_data_loaders(opt):
    ''' Retrieve dataloaders where audio signals are split into syllables'''

    train_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"], "corpus/split up data"
        ),
        directory="train",

        # ONLY NOISE FOR TRAINING DATASETS!
        background_noise=True, white_guassian_noise=True,
        background_noise_path=os.path.join(
            opt["data_input_dir"],
            "musan")
    )

    test_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"], "corpus/split up data",
        ),
        directory="test",
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_de_boer_sounds_decoder_data_loaders(opt, shuffle=True):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt["validate"]
    """
    print("Using Train+Val / Test Split")
    train_dataset = de_boer_decoder_sounds.DeBoerDecoderDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"],
            "corpus",
        ),
        directory="train"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_dataset = de_boer_decoder_sounds.DeBoerDecoderDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"],
            "corpus",
        ),
        directory="test"
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt["batch_size_multiGPU"],
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset,
