from typing import Tuple

import lightning as L
import torch
from torch import Tensor

from config_code.config_classes import DataSetConfig
from data import get_dataloader


class MyDataModule(L.LightningDataModule):
    def __init__(self, config: DataSetConfig):
        super().__init__()
        self.config = config

        self.train_loader, self.val_loader, self.test_loader = (
            get_dataloader.get_dataloader(config=self.config))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def get_batch(self, device) -> Tuple[Tensor, Tensor]:
        batch = next(iter(self.test_loader))
        # Assuming the batch is a tuple of (inputs, labels)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        return (inputs, labels)

    def get_all_data(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_list = torch.Tensor().to(device)
        labels_list = torch.Tensor().to(device)
        first_batch = True

        for inputs, labels in self.test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if first_batch:
                inputs_list = inputs
                labels_list = labels
                first_batch = False
            else:
                inputs_list = torch.cat((inputs_list, inputs), dim=0)
                labels_list = torch.cat((labels_list, labels), dim=0)

        print("-" * 10)
        print(f"In get_all_data: inputs_list.shape = {inputs_list.shape}, labels_list.shape = {labels_list.shape}")
        print("-" * 10)
        return inputs_list, labels_list
