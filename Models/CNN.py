import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, width, height, in_channels,num_classes):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 8 # intermediate channels
        self.num_classes = num_classes

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(self.out_channels * (width / 4) * (height / 4), self.num_classes) # 4, since the size halves at each maxpool w/current config: Hout = (Hin - 2)/2 + 1
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

def cnn(width, height, in_channels, num_classes, **kwargs): 
    """ returns CNN object
    """
    return CNN(width, height, in_channels, num_classes)