# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=5 --stop_epoch=1000 --save_freq=20
# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=6 --stop_epoch=1000 --save_freq=20
# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --train_aug --seed=7 --stop_epoch=1000 --save_freq=20
# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=5 --stop_epoch=1000 --save_freq=20
# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=6 --stop_epoch=1000 --save_freq=20
# CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset CUB --model Conv4 --method baseline --seed=7 --stop_epoch=1000 --save_freq=20

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=200
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split val --save_iter=200



CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=200
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split val --save_iter=200


CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=200
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split val --save_iter=200



CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=0
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=20
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=120
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=140
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=200
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split val --save_iter=200


