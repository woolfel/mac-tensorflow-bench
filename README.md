# mac-tensorflow-bench

Simple set of Tensorflow training scripts to see how well different versions of Apple silicon perform on training

## Install Conda

The benchmark uses conda to create an environment and installs the necessary tensorflow packages. You can download miniconda package and install it.

[https://docs.conda.io/en/latest/miniconda.html#macos-installers] installers

Select ARM 64 pkg and run the installer.

## Setup with Conda

1. git clone https://github.com/woolfel/mac-tensorflow-bench
2. cd mac-tensorflow-bench
3. conda env update -f environment.yml
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

rps_train uses rock_paper_scissors dataset from Tensorflow datasets. It's similar to cifar10 benchmark, but uses more memory. There's two versions of rock_paper_scissors training benchmark. The main difference between the two versions is the total number of parameters and layers. On Windows with RTX 2060 6G, both versions get out-of-memory error and won't run.

rps_train.py - 170,305,283

rps2_train.py - 749,091

Example:

python rps_train.py checkpoints/batch64 64


## Observations

RTX2060 6G runs into memory limitation with batch size 1024. Cifar10 images are small 32x32 pixels. If you were to train with larger images like ImageNet, PascalVoc or COCO, tensorflow would run into memory limitations quicker on memory limited video cards. ImageNet images are 224x224. COCO images vary in dimensions. Cifar10 images are ~1k, whereas COCO are 20-86K. If you use high res images that are 1024x1024 you would probably have to keep batch size below 64.

![memory error](https://github.com/woolfel/mac-tensorflow-bench/blob/master/windows_memory_warning.png)

### Rock Paper Scissors

When I run both versions on Windows, tensorflow attempts to allocate video memory and dies on layer 5. The big win for Apple Silicon Macbook is it enables you to run experiments on realistic datasets. Cifar10, Mnist and Fashion Mnist are industry standard benchmarks, but they don't produce usable models. If we look at Google's transfer learning recommendation, they suggest using MobileNet, DenseNet or ResNet trained on ImageNet dataset.

[ImageNet 2012 on Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012)

ImageNet is a good starting point for training real models. The dataset doesn't have a fixed size and the photo dimensions vary. There's 1000 classes in the 2012 version. 

Rock paper scissors photos are 300x300. Training has 2,520 and test has 372 images. Some general observations on the dataset. If you keep parameter count below 1 million, you can training batch size up to 32. Models like faster_rcnn, resnet, mobilenet and densenet will only train with batch size 8 or 16. To calculate out the limit, subtract the memory used by the model from the physical VRAM. The remaining memory is what's left for images.

paramter_count x 4 kilobytes = total memory used by model

phyiscal memory - total memory used by model = memory remaining for training data

If the training data is 300 x 300 pixel, the average file size should be around 270 kilobyte.

If the training data is 600 x 600, the average file size should be around 1 megabyte.
