import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from PIL import Image

from config_code.config_classes import DataSetConfig


class AnimalsWithAttributesDataset(Dataset):
    # https://github.com/dfan/awa2-zero-shot-learning/blob/master/AnimalDataset.py
    def __init__(self, config: DataSetConfig):
        data_dir = f"{config.data_input_dir}/awa2-dataset/AwA2-data/Animals_with_Attributes2_resized/"
        self.transform = None

        class_to_index = dict()
        # Build dictionary of indices to classes
        with open(f'{data_dir}/classes.txt') as f:
            index = 0
            for line in f:
                class_name = line.split('\t')[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        img_names = []
        img_index = []
        with open(f'{data_dir}/classes.txt') as f:
            for line in f:
                class_name = line.split('\t')[1].strip()  # split the line and take the second element
                FOLDER_DIR = os.path.join(f'{data_dir}/JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                files = glob(file_descriptor)  # glob is used to get all the files in the folder

                class_index = class_to_index[class_name]  # use the class name as the key
                for file_name in files:
                    img_names.append(file_name)
                    img_index.append(class_index)
        self.img_names = img_names
        self.img_index = img_index

    def __getitem__(self, index):
        im = Image.open(self.img_names[index])
        if im.getbands()[0] == 'L':
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)

        # if im.shape != (3, 64, 64):
        #     print(f"Image shape is {im.shape} for {self.img_names[index]}")

        im_index = self.img_index[index]

        # im_predicate = self.predicate_binary_mat[im_index, :]
        im_predicate = torch.zeros(85)  # todo
        class_index = self.img_index[index]
        return im, class_index
        # im, #im_predicate, self.img_names[index], im_index

    def __len__(self):
        return len(self.img_names)
