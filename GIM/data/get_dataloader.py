import os
import torch
from data import de_boer_sounds

NUM_WORKERS = 1


def get_de_boer_sounds_data_loaders(opt, split_and_pad=True, train_noise=True, shuffle=True):
    ''' Retrieve dataloaders where audio signals are split into syllables '''

    specific_directory = "split up data padded" if split_and_pad else ""

    train_dataset = de_boer_sounds.DeBoerDataset(
        opt=opt,
        root=os.path.join(
            opt["data_input_dir"], f"corpus/{specific_directory}"
        ),
        directory="train",

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
            opt["data_input_dir"], f"corpus/{specific_directory}",
        ),
        directory="test",
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
