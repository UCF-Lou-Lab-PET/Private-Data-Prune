# => Table 3. training from scratch
selections="AdaEL2NL1"
fraction=0.1
lr=0.5
num_experiments=1
datasets=("CIFAR10" )
data_paths=("./deepcore/data/cifar10")
classes=(10)

for n in 1; do
    for (( i=0; i<num_experiments; i++ )); do
        for selection in $selections; do
            # Run your command with the current values of --fraction and --selection
            CUDA_VISIBLE_DEVICES=0,1 python3 -u scrt_main.py \
            --fraction "$fraction" \
            --select_every 10 \
            --dataset MNIST \
            --data_path ~/datasets \
            --warm_epoch 0 \
            --num_exp 1 \
            --workers 10 \
            --optimizer SGD \
            -se 10 \
            --selection "$selection" \
            --model MLP \
            --lr "$lr" \
            --save_path ./result \
            --batch 128 \
            --epochs 10 \
            --scheduler CosineAnnealingLR \
            --save_path ./checkpoint
        done
    done
done