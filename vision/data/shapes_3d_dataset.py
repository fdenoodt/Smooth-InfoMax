# https://github.com/google-deepmind/3d-shapes

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import config_code.config_classes as config_classes
from config_code.config_classes import DataSetConfig


class Shapes3dDataset(Dataset):
    def __init__(self, config: DataSetConfig, device: torch.device, train=True):
        self.train = train
        data_dir = f"{config.data_input_dir}/3dshapes/3dshapes.h5"

        print(f"Opening file {data_dir}")
        with h5py.File(data_dir, 'r') as data:
            self.images = torch.tensor(np.array(data['images']))
            self.labels = torch.tensor(np.array(data['labels']))
        print(f"Loaded data from {data_dir}")

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.train:  # flip image with 50% probability
            if np.random.rand() > 0.5:
                img = torch.flip(img, [2])

        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    config = DataSetConfig(config_classes.Dataset.SHAPES_3D, batch_size=32, grayscale=False, num_workers=1)
    dataset = Shapes3dDataset(config, torch.device('cuda'))
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][1])

    # label:
    label = dataset[0][1]
    print(label)
    print(label.shape)  # torch.Size([6])
