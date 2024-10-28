# Fast Confidential Training of Deep Neural Networks With Encrypted Data Pruning


### Introduction
This is the code for Fast Confidential Training of Deep Neural Networks With Encrypted Data Pruning (Neurips 2024). This code is build upon the [DeepCore](https://github.com/PatrickZH/DeepCore) library, which implements a wide range of dataset pruning / coreset methods.

### Datasets
We use 5 datasets from [HETAL](https://github.com/CryptoLabInc/HETAL):
* MNIST
* CIFAR-10
* [DermaMNIST](https://www.nature.com/articles/s41597-022-01721-8)
* [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
* [SNIPS](https://arxiv.org/pdf/1805.10190v3.pdf)

For the transfer learning setting, we use a ViT-base model to extract the features of these data as a vector of dimension 768. More details about the feature extraction can be found [here](https://github.com/CryptoLabInc/HETAL/blob/main/src/hetal/README.md). For the training from scratch setting, we use the MNIST dataset. The pruning ratio, pruning frequency, pruning method etc. can be set accordingly.


### Tests
Simply run the following scripts, ```./tl_run.sh``` and ```./srct_run.sh```, to test the performance of data pruning.
```sh
model="Linear"
selections="AdaEL2NL1"
fraction=0.1
lr=0.5
num_experiments=1
datasets=("TL_CIFAR10" )
data_paths=("./deepcore/data/cifar10")
classes=(10)

for n in 1; do
    for (( i=0; i<num_experiments; i++ )); do
        dataset=${datasets[$i]}
        data_path=${data_paths[$i]}
        num_classes=${classes[$i]}
        for selection in $selections; do
            CUDA_VISIBLE_DEVICES=0,1 python3 -u tl_main.py \
            --fraction "$fraction" \
            --select_every 5 \
            --dataset "$dataset" \
            --data_path "$data_path"\
            --warm_epoch 0 \
            --num_exp 1 \
            --workers 10 \
            --optimizer SGD \
            -se 10 \
            --selection "$selection" \
            --model "$model" \
            --lr "$lr" \
            --save_path ./result \
            --batch 128 \
            --epochs 20 \
            --scheduler CosineAnnealingLR\
            --in_dim 768\
            --num_classes "$num_classes"
        done
    done
done
```

