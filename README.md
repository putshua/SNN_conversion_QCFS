# ai-research - ANN to SNN tuning

In this project we try to reproduce the results of a paper suggesting a new method for optimal ANN to SNN conversion with higher accuracy and ultra-low latency. We are using the code provided by the authors and try it on other datasets e-g. Dynamic Sensor event-based data.

- [Intro to SNN](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full)
- [Original paper](https://openreview.net/forum?id=7B3IJMM1k_M)
- [Authors repo](https://github.com/putshua/SNN_conversion_QCFS)
- [Tonic lib](https://github.com/neuromorphs/tonic)

## Run the models
```
# before running
1) update the `DIR` variable in the `Preprocess\getdataloader.py` (line 9)
2) create a directort at root called `saved_models`
````
You can train a model with QCFS layer with
```
python main.py train --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP
```

You can test accuracy in ANN mode or SNN mode with
```
python main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME
```

## ETH cluster usage
```
# first launch (download dataset)
load module eth-proxy

# create a new job default configs
bsub  -n {NUM_PROCESSORS} -W {WALL_CLOCK}  -o {LOG_PATH} -R {REQUIREMENTS}

# create new job project configs
bsub  -n 4 -W 08:00  -o {LOG_PATH} -R "rusage[mem=2048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]"

# list active jobs
bjobs 

# track active jobs
bpeek JOB_ID 
```

## ETH Euler cluster setup
```
# connect to cluster
ssh creds@euler.ethz.ch

# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh
rm miniconda.sh

# disable base env.
conda config --set auto_activate_base false

# create new env.
conda create --name env_name python=3.8
conda activate env_name
```