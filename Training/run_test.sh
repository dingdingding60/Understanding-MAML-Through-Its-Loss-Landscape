CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2
