# https://github.com/google-deepmind/3d-shapes

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config_code.config_classes as config_classes
from config_code.config_classes import DataSetConfig


class Shapes3dDataset(Dataset):
    def __init__(self, config: DataSetConfig):
        data_dir = f"{config.data_input_dir}/3dshapes/3dshapes.h5"
        self.data = h5py.File(data_dir, 'r')

        self.images = self.data['images']  # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = self.data['labels']  # array shape [480000,6], float64

        # Shuffle the data
        self.grayscale = config.grayscale
        if self.grayscale:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img = self.images[index]
        if self.grayscale:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)  # Convert image from HWC to CHW format
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    config = DataSetConfig(config_classes.Dataset.SHAPES_3D, batch_size=32, grayscale=False)
    dataset = Shapes3dDataset(config)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][1])
