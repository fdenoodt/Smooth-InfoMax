# temp cpc_audio_xxxx

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from data import get_dataloader
from config_code.config_classes import OptionsConfig
from options import get_options


class FullySupervisedModel(nn.Module):
    def __init__(self, cnn_hidden_dim, regressor_hidden_dim, num_classes, freeze: bool):
        super(FullySupervisedModel, self).__init__()
        self.label_num = num_classes
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_hidden_dim, kernel_size=10, stride=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # batch_first=True: input and output tensors are provided as (batch, seq, feature)
        self.regressor = nn.GRU(input_size=cnn_hidden_dim, hidden_size=regressor_hidden_dim, batch_first=True)

        if freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.regressor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(regressor_hidden_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, feature, seq) -> (batch, seq, feature)

        regress_hidden_state = torch.zeros(1, x.size(0), 256, device=x.device)

        self.regressor.flatten_parameters()
        output, regress_hidden_state = self.regressor(x, regress_hidden_state)
        # output: (batch, seq, feature)
        output = output.permute(0, 2, 1)  # (batch, feature, seq)
        pooled_c = nn.functional.adaptive_avg_pool1d(output, 1, )  # shape: (batch_size, hidden_dim, 1)
        pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, 256)  # shape: (batch_size, hidden_dim)

        x = self.classifier(pooled_c)
        return x


def calculate_accuracy(opt: OptionsConfig, model: FullySupervisedModel, test_loader: DataLoader, syllables: bool):
    # Move the testing loop outside of the training loop
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, _, labels, _) in test_loader:
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = correct / total  # Calculate final accuracy
    wandb.log({"Final Accuracy": final_accuracy})  # Log final accuracy
    print(f'Final Accuracy: {final_accuracy}')  # Print final accuracy


def main(syllables: bool):
    wandb.init(project="temp",
               name=f"FROZEN_model_{'vowel' if not syllables else 'syllable'}_classifier_{wandb.util.generate_id()}")

    opt: OptionsConfig = get_options()

    if syllables:
        opt.syllables_classifier_config.dataset.labels = "syllables"
    else:
        opt.syllables_classifier_config.dataset.labels = "vowels"
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.syllables_classifier_config.dataset)

    num_classes = 9 if syllables else 3
    model = FullySupervisedModel(cnn_hidden_dim=512, regressor_hidden_dim=256, num_classes=num_classes, freeze=True)
    model = model.to(opt.device)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 200
    lr = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i, (inputs, _, labels, _) in enumerate(train_loader):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, _, labels, _) in test_loader:
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        wandb.log({"Loss": loss.item(), "Accuracy": correct / total, "Epoch": epoch})
        print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {correct / total}')

    calculate_accuracy(opt, model, test_loader, syllables)


if __name__ == "__main__":
    # main(True)

    # test model
    model = FullySupervisedModel(cnn_hidden_dim=512, regressor_hidden_dim=256, num_classes=9, freeze=True)
    rnd = torch.rand(64, 1, 10240)
    print(model(rnd).shape)

