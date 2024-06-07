# https://github.com/google-deepmind/3d-shapes

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import config_code.config_classes as config_classes
from config_code.config_classes import DataSetConfig


class Shapes3dDataset(Dataset):
    def __init__(self, config: DataSetConfig, images: np.ndarray, labels: np.ndarray, len: int,
                 device: torch.device, train=True):

        self.images = images
        self.labels = labels
        self.len = len

        self.grayscale = config.grayscale
        self.transform = transforms.Compose([transforms.ToPILImage()])

        # append grayscale if needed
        if self.grayscale:
            self.transform.transforms.append(transforms.Grayscale())

        if train:
            self.transform.transforms.append(transforms.RandomHorizontalFlip())

        self.transform.transforms.append(transforms.ToTensor())

    @staticmethod
    def get_data(config: DataSetConfig):
        with h5py.File(f"{config.data_input_dir}/3dshapes/3dshapes.h5", 'r') as data:
            images = data['images']
            labels = data['labels']

            # Subset for local testing. (dataset is too large to fit in memory locally)
            if config.dataset == config_classes.Dataset.SHAPES_3D_SUBSET:
                images = images[:800]
                labels = labels[:800]

            # Save in memory
            images = np.array(images)
            labels = np.array(labels)
            length = len(images)
        return images, labels, length

    def __getitem__(self, index):
        img = self.images[index]
        img = self.transform(img)

        label = self.labels[index]  # (6,): 0: floor_hue, 1: wall_hue, 2: object_hue, 3: scale, 4: shape, 5: orientation
        shape = label[4] # double to int
        shape = torch.tensor(shape, dtype=torch.long)
        return img, shape

    def __len__(self):
        return self.len


if __name__ == '__main__':
    config = DataSetConfig(config_classes.Dataset.SHAPES_3D_SUBSET, batch_size=32, grayscale=False, num_workers=8)

    images, labels, len = Shapes3dDataset.get_data(config)
    dataset = Shapes3dDataset(config, images, labels, len, torch.device('cuda'))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)  # (32, 6); 6 is the number of factors of variation
        print(labels)  # 0.0
        break
