CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed5 --split novel --save_iter=800
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed5 --split novel --save_iter=800
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed6 --split novel --save_iter=800
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed6 --split novel --save_iter=800
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed7 --split novel --save_iter=800
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_aug_seed7 --split novel --save_iter=800

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed5 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed5 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed6 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed6 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed7 --split novel --save_iter=3
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_seed7 --split novel --save_iter=3

CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed5 --split novel --save_iter=999
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed5 --split novel --save_iter=999
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed6 --split novel --save_iter=999
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed6 --split novel --save_iter=999
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed7 --split novel --save_iter=999
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv6 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv6_baseline_aug_seed7 --split novel --save_iter=999

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed5 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed5 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed6 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed6 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed7 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_seed7 --split novel --save_iter=60

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed5 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed5 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed6 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed6 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed7 --split novel --save_iter=100
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet10 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet10_baseline_aug_seed7 --split novel --save_iter=100

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed5 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed5 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed6 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed6 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed7 --split novel --save_iter=60
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_seed7 --split novel --save_iter=60

CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed5 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed5 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed6 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed6 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python save_features.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed7 --split novel --save_iter=40
CUDA_VISIBLE_DEVICES=1 python test.py --model ResNet18 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/ResNet18_baseline_aug_seed7 --split novel --save_iter=40