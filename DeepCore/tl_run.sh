#!/bin/bash
# number of class in each dataset:
# cifar10==10, mnist==10, derma==7, snips==7, fmd==3

# => Table 1. Transfer learning different dataset
#       change the --dataset, --data_path, --num_classes parameter
# for all datasets, refer to: https://github.com/CryptoLabInc/HETAL

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
        # Iterate over selections
        for selection in $selections; do
            # Run your command with the current values of --fraction and --selection
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

# => Figure 4. Different ratio
#       change the --fraction parameter