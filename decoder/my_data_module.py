import lightning as L

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
