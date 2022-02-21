# Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## Usage
Train ResNet-34 on ImageNet parallel
```python
python train.py --gpus=8 --L=8 --id=MODEL_SAVED_NAME
```

Test ANN
```python
python evaluate.py --mode=ann --gpus=8 --t=8 --id=MODEL_SAVED_NAME
```

Test SNN
```python
python evaluate.py --mode=snn --gpus=8 --t=8 --id=MODEL_SAVED_NAME
```