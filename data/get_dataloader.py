import os
import torch
from torch.utils.data import dataset

from config_code.config_classes import DataSetConfig, Dataset
from data.de_boer import de_boer_sounds
from data.libri import librispeech
from data.radio.radio import _get_radio_data_loaders, _get_noisy_test_data_loader


def _dataloaders(dataset_options: DataSetConfig, train_specific_dir, test_specific_dir, train_sub_dir, test_sub_dir,
                 shuffle) -> (
        torch.utils.data.DataLoader, dataset.Dataset, torch.utils.data.DataLoader, dataset.Dataset):
    data_input_dir = dataset_options.data_input_dir
    train_dataset = de_boer_sounds.DeBoerDataset(
        dataset_options=dataset_options,
        root=os.path.join(
            data_input_dir, f"corpus/{train_specific_dir}"
        ),
        directory=train_sub_dir,
    )

    test_dataset = de_boer_sounds.DeBoerDataset(
        dataset_options=dataset_options,
        root=os.path.join(
            data_input_dir, f"corpus/{test_specific_dir}",
        ),
        directory=test_sub_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=dataset_options.batch_size_multiGPU,
        shuffle=shuffle,
        drop_last=True,
        num_workers=dataset_options.num_workers,
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=dataset_options.batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=dataset_options.num_workers,
        persistent_workers=True
    )

    return train_loader, train_dataset, test_loader, test_dataset


def _get_de_boer_sounds_data_loaders(d_config: DataSetConfig, shuffle=True):
    ''' Retrieve dataloaders where audio signals are split into syllables '''
    print("Loading De Boer Sounds dataset...")

    split: bool = d_config.split_in_syllables
    if split:  # for classification
        # specific_directory = "split up data cropped reshuffled"
        specific_directory = "split up data padded"
    else:
        specific_directory = "reshuffledv2"

    print(f"using {specific_directory} directory")
    return _dataloaders(d_config, specific_directory, specific_directory, "train", "test", shuffle)


def _get_libri_dataloaders(options: DataSetConfig):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
    """
    print("Loading LibriSpeech dataset...")

    print("Using Train+Val / Test Split")

    libri_dir = "LibriSpeech/train-clean-100"
    labels_dir = "LibriSpeech100_labels_split" if options.dataset == Dataset.LIBRISPEECH else "LibriSpeech100_labels_split_subset"

    train_dataset = librispeech.LibriDataset(
        os.path.join(
            options.data_input_dir,
            libri_dir,
        ),
        os.path.join(
            options.data_input_dir, f"{labels_dir}/train_split.txt"
        ),
    )

    test_dataset = librispeech.LibriDataset(
        os.path.join(
            options.data_input_dir,
            libri_dir,
        ),
        os.path.join(
            options.data_input_dir, f"{labels_dir}/test_split.txt"
        ),
    )

    batch_size_multiGPU = options.batch_size_multiGPU
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_multiGPU,
        shuffle=True,
        drop_last=True,
        num_workers=options.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=options.num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_dataloader(config: DataSetConfig,
                   # **kwargs for instance shuffle=True/False, used in DeBoer dataset.
                   # (for instance for training dataset to make fixed in latent space analysis)
                   # But kwargs not used anywhere else I believe.
                   **kwargs) \
        -> (torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader):
    d = config.dataset
    if d == Dataset.DE_BOER:
        raise NotImplementedError("De Boer dataset is not supported")
        return _get_de_boer_sounds_data_loaders(config, **kwargs)
    elif d in [Dataset.LIBRISPEECH, Dataset.LIBRISPEECH_SUBSET]:
        raise NotImplementedError("LibriSpeech dataset is not supported")
        return _get_libri_dataloaders(config)
    elif d == Dataset.RADIO:
        return _get_radio_data_loaders(config)
    else:
        raise ValueError("Unknown dataset")

def get_noisy_test_data_loader(config: DataSetConfig, snr) -> torch.utils.data.DataLoader:
    return _get_noisy_test_data_loader(config, snr)
