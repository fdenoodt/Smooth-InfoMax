import os
import torch
from torch.utils.data import dataset

from data import de_boer_sounds, librispeech
from configs.config_classes import DataSetConfig, Dataset, OptionsConfig

NUM_WORKERS = 1


def _dataloaders(dataset_options: DataSetConfig, train_specific_dir, test_specific_dir, train_sub_dir, test_sub_dir,
                 split_and_pad, train_noise, shuffle):
    data_input_dir = dataset_options.data_input_dir
    train_dataset = de_boer_sounds.DeBoerDataset(
        dataset_options=dataset_options,
        root=os.path.join(
            data_input_dir, f"corpus/{train_specific_dir}"
        ),
        directory=train_sub_dir,

        # ONLY NOISE FOR TRAINING DATASETS!
        background_noise=train_noise, white_guassian_noise=train_noise,
        background_noise_path=os.path.join(data_input_dir, "musan"),
        split_into_syllables=split_and_pad
    )

    test_dataset = de_boer_sounds.DeBoerDataset(
        dataset_options=dataset_options,
        root=os.path.join(
            data_input_dir, f"corpus/{test_specific_dir}",
        ),
        directory=test_sub_dir,
        background_noise=False, white_guassian_noise=False,
        split_into_syllables=split_and_pad
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=dataset_options.batch_size_multiGPU,
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=dataset_options.batch_size_multiGPU,
        shuffle=shuffle,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def _get_de_boer_sounds_data_loaders(opt: OptionsConfig, reshuffled=None, split_and_pad=True, train_noise=True,
                                     shuffle=True,
                                     subset_size=None):
    ''' Retrieve dataloaders where audio signals are split into syllables '''
    print("Loading De Boer Sounds dataset...")

    if split_and_pad:
        if subset_size:
            print(f"Using subset of size {subset_size} and batch size {opt.encoder_config.dataset.batch_size}")
            train_specific_directory = "subsets/"
            train_sub_dir = f"{subset_size}"  # eg: subsets/all
            test_specific_directory = "split up data padded reshuffled"
            dataset_config = opt.encoder_config.dataset
            return _dataloaders(dataset_config, train_specific_directory, test_specific_directory, train_sub_dir,
                                "test", split_and_pad, train_noise, shuffle)
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

    print(f"using {specific_directory} directory")
    dataset_config = opt.encoder_config.dataset
    return _dataloaders(dataset_config, specific_directory, specific_directory, "train", "test", split_and_pad, train_noise,
                        shuffle)


def _get_libri_dataloaders(options: DataSetConfig):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
    """
    print("Loading LibriSpeech dataset...")
    num_workers = 1

    print("Using Train+Val / Test Split")
    train_dataset = librispeech.LibriDataset(
        os.path.join(
            options.data_input_dir,
            "LibriSpeech/train-clean-100",
        ),
        os.path.join(
            options.data_input_dir, "LibriSpeech100_labels_split/train_split.txt"
        ),
    )

    test_dataset = librispeech.LibriDataset(
        os.path.join(
            options.data_input_dir,
            "LibriSpeech/train-clean-100",
        ),
        os.path.join(
            options.data_input_dir, "LibriSpeech100_labels_split/test_split.txt"
        ),
    )

    batch_size_multiGPU = options.batch_size_multiGPU
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_multiGPU,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset


def get_dataloader(opt: OptionsConfig, config: DataSetConfig, **kwargs):
    d = config.dataset
    if d == Dataset.DE_BOER:
        return _get_de_boer_sounds_data_loaders(opt, **kwargs)
    elif d == Dataset.DE_BOER_RESHUFFLED:  # used for training CPC
        return _get_de_boer_sounds_data_loaders(opt, reshuffled="v1", **kwargs)
    elif d == Dataset.DE_BOER_RESHUFFLED_V2:  # used for training CPC Decoder
        return _get_de_boer_sounds_data_loaders(opt, reshuffled="v2", **kwargs)
    elif d == Dataset.LIBRISPEECH:
        return _get_libri_dataloaders(config)
    else:
        raise ValueError("Unknown dataset")
