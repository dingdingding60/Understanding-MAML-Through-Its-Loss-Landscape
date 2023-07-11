# Transfer learning
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=5 --stop_epoch=1000 --save_freq=20
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=6 --stop_epoch=1000 --save_freq=20
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=7 --stop_epoch=1000 --save_freq=20
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=5 --stop_epoch=1000 --save_freq=20
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=6 --stop_epoch=1000 --save_freq=20
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=7 --stop_epoch=1000 --save_freq=20

#FO-MAML
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=5 -train_aug
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=6 -train_aug
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=7 -train_aug
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=5
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=6
CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method maml_approx --seed=7
