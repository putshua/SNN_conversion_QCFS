import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, width, height, in_channels,num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=5, stride=1, padding='same', bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same', bias=False),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * int(width / 8) * int(height / 8), 1024, bias=False), # 4, since the size halves at each maxpool w/current config: Hout = (Hin - 2)/2 + 1
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

def cnn(width, height, in_channels, num_classes, **kwargs): 
    """ returns CNN object
    """
    return CNN(width, height, in_channels, num_classes)