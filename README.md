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

## Cifar 10 benchmark

