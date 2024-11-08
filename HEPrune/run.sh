# number of class in each dataset:
# cifar10==10, mnist==10, derma==7, snips==7, fmd==3

### 1. Experiments on different datasets
# cifar10
./bin/hefs cls=10 ctn=342
./bin/cam nsample=43750 p=0.1
./bin/cwp batch=128 ctn=342 p=0.1

# mnist
./bin/hefs cls=10 ctn=411
./bin/cam nsample=52500 p=0.1
./bin/cwp batch=128 ctn=411 p=0.1

# fmd
./bin/hefs cls=3 ctn=23
./bin/cam nsample=2849 p=0.1
./bin/cwp batch=128 ctn=23 p=0.1

# derma
./bin/hefs cls=7 ctn=55
./bin/cam nsample=7007 p=0.1
./bin/cwp batch=128 ctn=55 p=0.1

# snips
./bin/hefs cls=7 ctn=103
./bin/cam nsample=13084 p=0.1
./bin/cwp batch=128 ctn=103 p=0.1

### 2. Experiments with different pruning ratio
# CIFAR-10
# The latency of HEFS is unchanged.
./bin/hefs cls=10 ctn=342
# for cam and cwp, vaty p from 0.01 to 0.1 (here p is the remaining fraction)
./bin/cam nsample=43750 p=0.1
./bin/cwp batch=128 ctn=342 p=0.1

# MNIST
# The latency of HEFS is unchanged.
./bin/hefs cls=10 ctn=411
# for cam and cwp, vaty p from 0.01 to 0.1 (here p is the remaining fraction)
./bin/cam nsample=52500 p=0.1
./bin/cwp batch=128 ctn=411 p=0.1

# 3. Traing from scratch
# The overhead of enctypted data prunning does not directly depend on the model architecture
# The forward pass needs more time, the forward pass should be tested in HETAL
