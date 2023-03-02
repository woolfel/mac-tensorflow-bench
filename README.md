# mac-tensorflow-bench

Simple set of Tensorflow training scripts to see how well different versions of Apple silicon perform on training

## Install Conda

The benchmark uses conda to create an environment and installs the necessary tensorflow packages. You can download miniconda package and install it.

[https://docs.conda.io/en/latest/miniconda.html#macos-installers] installers

Select ARM 64 pkg and run the installer.

## Setup with Conda

1. git clone https://github.com/woolfel/mac-tensorflow-bench
2. cd mac-tensorflow-bench
3. conda env update environment.yml
4. conda activate tensorflow_bench
5. verify everything is installed correctly by running env_check.py

### Setup / Errors on Windows

To install on windows, rename environment.yml to environment-mac.yml. Rename environment-win.yml to environment.yml and run "conda env update environment.yml" command.

## Cifar 10 benchmark

Cifar10 benchmark will use CIFAR 10 from tensorflow datasets with a simple Keras sequential model. The model has 13 layers. The script takes 2 arguments: checkpoint_path batch_size

python cifar10_train.py checkpoints/batch64 64

## Results

I have some basic results in a CSV file

[https://github.com/woolfel/mac-tensorflow-bench/blob/master/cifar10_results.csv] cifar10_results.csv

## Rock Paper Scissors benchmark

rps_benchmark uses rock_paper_scissors dataset from Tensorflow datasets. It is similar to cifar10 benchmark, but uses more memory. Using batch size 64 on M2Max, it uses 30G of memory. On Windows 10 with RTX 2060 6G, it crashes with out of memory error. This suggests the benchmark won't run on anything less than 20G of video memory.

## Observations

RTX2060 6G runs into memory limitation with batch size 1024. Cifar10 images are small 32x32 pixels. If you were to train with larger images like ImageNet, PascalVoc or COCO, tensorflow would run into memory limitations quicker on memory limited video cards. ImageNet images are 224x224. COCO images vary in dimensions. Cifar10 images are ~1k, whereas COCO are 20-86K. If you use high res images that are 1024x1024 you would probably have to keep batch size below 64.

![memory error](https://github.com/woolfel/mac-tensorflow-bench/blob/master/windows_memory_warning.png)
