# https://github.com/google-deepmind/3d-shapes

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config_code.config_classes as config_classes
from config_code.config_classes import DataSetConfig


class Shapes3dDataset(Dataset):
    def __init__(self, config: DataSetConfig, device: torch.device, train=True):
        data_dir = f"{config.data_input_dir}/3dshapes/3dshapes.h5"
        self.data = h5py.File(data_dir, 'r')

        self.images = torch.tensor(self.data['images'], device=device)  # Move data to GPU
        self.labels = torch.tensor(self.data['labels'], device=device)  # Move data to GPU

        self.grayscale = config.grayscale
        if self.grayscale:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Grayscale(),
                 transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

        if train:  # random flip
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), self.transform])

    def __getitem__(self, index):
        img = self.images[index]
        if self.grayscale:
            # Apply transform on CPU then move back to GPU
            # because the transform is not implemented for GPU due to ToPilImage (suboptimal)
            img = self.transform(img.cpu()).to(self.images.device)
        else:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    config = DataSetConfig(config_classes.Dataset.SHAPES_3D, batch_size=32, grayscale=False)
    dataset = Shapes3dDataset(config, torch.device('cuda'))
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][1])
