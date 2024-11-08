# => Table 3. training from scratch
selections="AdaEL2NL1"
num_experiments=10
fractions=(0.01 0.05 0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)
lrs=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
wrs=(0.13 0.03 0 0 0 0 0 0 0 0)

for n in 1; do
    for (( i=0; i<num_experiments; i++ )); do
        fraction=${fractions[$i]}
        lr=${lrs[$i]}
        wr=${wrs[$i]}
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
            --epochs 50 \
            --scheduler CosineAnnealingLR \
            --save_path ./checkpoint
        done
    done
done