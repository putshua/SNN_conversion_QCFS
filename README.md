# IMPORTANT
**I have updated a new version in a new repo at https://github.com/putshua/ANN_SNN_QCFS.
The biggest unreproducable bug is fixed.
Compatible with old version.
Please switch to the new version.**

# Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks


## Usage

Please first change the variable "DIR" at File ".Preprocess\getdataloader.py", line 9 to your own dataset directory

Train model with QCFS-Layer 

```bash
python main.py train --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP
```
Test accuracy in ann mode or snn mode

```bash
python main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME
```

One pretrained model at https://drive.google.com/file/d/1HL-ngCcRTqXw6L6XML-1RCL6dgP1GIDZ/view?usp=share_link

The paper in the openreview has a little problem with the derivative of $\lambda$ for the QCFS activation function, we will soon upadate an arxiv version and make a correction. Codes are always correct because of the autograd mechanism in pytorch.
