import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoded_dimension):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False), #bias false since following batchnorm cancels it
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.flatten = nn.Flatten(start_dim=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=encoded_dimension)
        )

    def forward(self, x):
        out = self.encoder_cnn(x)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out

class Encoder_new(nn.Module):
    def __init__(self, encoded_dimension):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=8),


            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False), #bias false since following batchnorm cancels it
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),


            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32))

        self.flatten = nn.Flatten(start_dim=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            #       nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=encoded_dimension)
        )

    def forward(self, x):
        out = self.encoder_cnn(x)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out

class Encoder_small(nn.Module):
    def __init__(self, encoded_dimension):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1, bias=False), #bias false since following batchnorm cancels it
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.flatten = nn.Flatten(start_dim=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=16, out_features=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=2, out_features=encoded_dimension)
        )

    def forward(self, x):
        out = self.encoder_cnn(x)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out

