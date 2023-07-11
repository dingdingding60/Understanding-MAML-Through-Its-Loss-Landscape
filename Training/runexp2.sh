CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=1
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=1
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed6 --split novel --save_iter=1
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed6 --split novel --save_iter=1
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed7 --split novel --save_iter=1
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed7 --split novel --save_iter=1

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed6 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed6 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed7 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed7 --split novel --save_iter=80

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed5 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed6 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed6 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed7 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_seed7 --split novel --save_iter=3

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split novel --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed5 --split novel --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed6 --split novel --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed6 --split novel --save_iter=180
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed7 --split novel --save_iter=180
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv6_baseline_aug_seed7 --split novel --save_iter=180

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed6 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed6 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed7 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed7 --split novel --save_iter=40

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed6 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed6 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed7 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed7 --split novel --save_iter=100

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed5 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed6 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed6 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed7 --split novel --save_iter=80
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_seed7 --split novel --save_iter=80

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split novel --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed5 --split novel --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed6 --split novel --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed6 --split novel --save_iter=160
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed7 --split novel --save_iter=160
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_baseline_aug_seed7 --split novel --save_iter=160