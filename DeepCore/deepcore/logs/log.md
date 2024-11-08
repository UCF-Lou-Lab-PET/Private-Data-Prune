# Training with data pruning

We provide some logs files as well as visualization.

## Different datasets
We train with $10\%$ of the total data on different datasets. 
On the MNIST dataset, in ```DeepCore/deepcore/logs/tl_mnist.log```.
![mnist](./fig/mnist.jpg "HEPrune")
On the CIFAR-10 dataset, in ```DeepCore/deepcore/logs/tl_cifar10.log```.
![cifar10](./fig/cifar10.jpg "HEPrune")
On the FMD dataset, in ```DeepCore/deepcore/logs/tl_fmd.log```.
![fmd](./fig/fmd.jpg "HEPrune")
On the DermaMNIST dataset, in ```DeepCore/deepcore/logs/tl_derma.log```.
![derma](./fig/derma.jpg "HEPrune")
On the SNIPS dataset, in ```DeepCore/deepcore/logs/tl_snips.log```.
![snips](./fig/snips.jpg "HEPrune")

For detailed log of training, please refer to the log files.

## Different pruninig ratio
Different ratios on the CIFAR-10 dataset.
![snips](./fig/ratiocifar10.jpg "HEPrune")

Different ratios on the MNIST dataset.
![snips](./fig/ratiomnist.jpg "HEPrune")

Log files will generated when executing the scripts. Sample log files in ```DeepCore/deepcore/logs/tl_cifar10_0.01.log```.