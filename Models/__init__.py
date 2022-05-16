from .ResNet import *
from .VGG import *
from .CNN import *

def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    elif 'dvsgestures' in DATANAME.lower():
        num_classes = 11
    else:
        num_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'cnn':
        if 'cifar' in DATANAME.lower(): 
            return cnn(width=32, height=32, in_channels=3, num_classes=num_classes)
        elif 'dvsgesture' in DATANAME.lower():
            return cnn(width=128, height=128, in_channels=2, num_classes=num_classes)
    else:
        print("still not support this model/dataset combination")
        exit(0)