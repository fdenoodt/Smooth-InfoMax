from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from config_code.config_classes import DataSetConfig
import torch


def load_data_and_label(path, batch_size, data_type):
    print(f"info: start loading the {data_type} data")
    X_data = torch.load(f"{path}X_{data_type}.pth")  # (b, c, t)
    Y_data = torch.load(f"{path}y_{data_type}.pth")  # (b, 1)

    # Flatten the channel and time dimensions for min and max calculation
    flat_X_data = X_data.view(X_data.shape[0], -1)  # Flattening to (b, c*t)

    # Calculate global min and max across both channels
    min_vals = flat_X_data.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_vals = flat_X_data.max(dim=1, keepdim=True)[0].view(-1, 1, 1)

    # Apply normalization using global min and max
    X_data_scaled = -1 + 2 * (X_data - min_vals) / (max_vals - min_vals)
    X_data_scaled = X_data_scaled.float()
    print(f"info: loaded the {data_type} signals")

    data_dataset = TensorDataset(X_data_scaled, Y_data)
    shuffle = True if data_type == 'train' else False
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return data_loader


def _get_radio_data_loaders(config: DataSetConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    path = f"{config.data_input_dir}/RadioIdentification/"
    train_loader = load_data_and_label(path, config.batch_size_multiGPU, 'train')
    # create a sub_train_loader which is the subset of the train_loader of percentage 10%
    sub_train_loader = torch.utils.data.Subset(train_loader.dataset, torch.arange(0, int(0.7 * len(train_loader.dataset))))
    # print the shape of the sub_train_loader
    print(f"info: sub_train_loader shape: {len(sub_train_loader)}")
    sub_train_loader = DataLoader(sub_train_loader, batch_size=config.batch_size_multiGPU, shuffle=True, drop_last=True)
    val_loader = load_data_and_label(path, config.batch_size_multiGPU, 'val')
    test_loader = load_data_and_label(path, config.batch_size_multiGPU, 'test')
    return sub_train_loader, val_loader, test_loader
