# Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## Usage

Please first change the variable "DIR" at File ".Preprocess\getdataloader.py", line 9 to your own dataset directory

Train model with QCFS layer 

python .\main.py train --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP

Test accuracy in ann mode or snn mode

python .\main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME