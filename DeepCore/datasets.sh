model="Linear"
selections="AdaEL2NL1"
fraction=0.1

num_experiments=5
datasets=("TL_MNIST" "TL_CIFAR10" "TL_FMD" "TL_DERMA" "TL_SNIPS")
data_paths=("../data/mnist" "../data/cifar10" "../data/face-mask-detection" "../data/dermamnist" "../data/snips")
lrs=(0.5 0.5 0.8 0.95 0.5)
wrs=(0.04 0 0.025 0.18 0)
freqs=(3 3 5 5 5)
epochs=(7 9 22 23 14)
classes=(10 10 3 7 7)

for n in 1; do
    for (( i=0; i<num_experiments; i++ )); do
        dataset=${datasets[$i]}
        data_path=${data_paths[$i]}
        lr=${lrs[$i]}
        wr=${wrs[$i]}
        freq=${freqs[$i]}
        epoch=${epochs[$i]}
        num_classes=${classes[$i]}
        for selection in $selections; do
            CUDA_VISIBLE_DEVICES=0,1 python3 -u tl_main.py \
            --fraction "$fraction" \
            --select_every "$freq" \
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
            --epochs "$epoch" \
            --scheduler CosineAnnealingLR\
            --in_dim 768\
            --num_classes "$num_classes"\
            --window_ratio "$wr" 
        done
    done
done