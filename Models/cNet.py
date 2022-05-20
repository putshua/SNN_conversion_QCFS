import torch.nn as nn
import torch

class cNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_1 = 16
        self.out_channels_2 = 32
        self.out_channels_3 = 64
        self.out_channels_4 = 10
    
        self.num_classes = num_classes

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels_1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels=self.out_channels_1, out_channels=self.out_channels_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels=self.out_channels_2, out_channels=self.out_channels_3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels_3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Defining another 2D convolution layer
            nn.Conv2d(in_channels=self.out_channels_3, out_channels=self.out_channels_4, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels_4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(self.out_channels_4 * 28 * 28, self.num_classes) # assumes 128x128 input, change if necessary
        )
    
    def forward(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

def cnet(in_channels, num_classes, **kwargs): 
    """ returns cNet object
    """
    return cNet(in_channels, num_classes)


