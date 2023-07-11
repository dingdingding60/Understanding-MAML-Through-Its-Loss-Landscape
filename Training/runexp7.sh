
CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=0
CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=0
CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split base --save_iter=0
CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split base --save_iter=0


CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=200
CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split novel --save_iter=200
CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split base --save_iter=200
CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split base --save_iter=200

# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=200
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_seed5 --split val --save_iter=200



# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=200
# CUDA_VISIBLE_DEVICES=0 python test.py --model Conv4 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/Conv4_baseline_aug_seed5 --split val --save_iter=200

# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=200
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_seed5 --split val --save_iter=200



# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=0
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=20
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=40
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=60
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=80
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=100
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=120
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=140
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=160
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=180
# CUDA_VISIBLE_DEVICES=0 python save_features.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=200
# CUDA_VISIBLE_DEVICES=0 python test.py --model ResNet10 --method baseline --dataset miniImagenet --ckpt_dir /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_baseline_aug_seed5 --split val --save_iter=200
