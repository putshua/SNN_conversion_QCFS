import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.in_channels = 3 # I think there are 3 RGB input channels?
        self.out_channels = 5
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
            nn.Linear(self.out_channels * 8 * 8, self.num_classes) # 8 because of the MaxPool: Hout = (Hin - 2)/2 + 1, Hin = Win = 32
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

def cnn(num_classes=10, **kwargs): 
    """ returns CNN object
    """
    return CNN(num_classes=num_classes)