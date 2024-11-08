# for CIFAR-10
model="Linear"
selections="AdaEL2NL1"
num_experiments=10
fractions=(0.01 0.05 0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)
lrs=(1.5 1.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)
wrs=(0.04 0.04 0 0 0 0 0 0 0 0)

for n in 1; do
    for (( i=0; i<num_experiments; i++ )); do
        fraction=${fractions[$i]}
        lr=${lrs[$i]}
        wr=${wrs[$i]}
        for selection in $selections; do
            CUDA_VISIBLE_DEVICES=0,1 python3 -u tl_main.py \
            --fraction "$fraction" \
            --select_every 5 \
            --dataset TL_CIFAR10 \
            --data_path ../data/cifar10\
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
            --epochs 15 \
            --scheduler CosineAnnealingLR\
            --in_dim 768\
            --num_classes 10\
            --window_ratio "$wr" 
        done
    done
done

# for MNIST
# model="Linear"
# selections="AdaEL2NL1"
# num_experiments=10
# fractions=(0.01 0.05 0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)
# lrs=(1.5 1.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)
# wrs=(0.04 0.04 0 0 0 0 0 0 0 0)

# for n in 1; do
#     for (( i=0; i<num_experiments; i++ )); do
#         fraction=${fractions[$i]}
#         lr=${lrs[$i]}
#         wr=${wrs[$i]}
#         for selection in $selections; do
#             CUDA_VISIBLE_DEVICES=0,1 python3 -u tl_main.py \
#             --fraction "$fraction" \
#             --select_every 5 \
#             --dataset TL_MNIST \
#             --data_path ../data/mnist\
#             --warm_epoch 0 \
#             --num_exp 1 \
#             --workers 10 \
#             --optimizer SGD \
#             -se 10 \
#             --selection "$selection" \
#             --model "$model" \
#             --lr "$lr" \
#             --save_path ./result \
#             --batch 128 \
#             --epochs 15 \
#             --scheduler CosineAnnealingLR\
#             --in_dim 768\
#             --num_classes 10\
#             --window_ratio "$wr" 
#         done
#     done
# done