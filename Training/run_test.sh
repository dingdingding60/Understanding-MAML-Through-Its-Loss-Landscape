CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed5 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed6 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2
CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method baseline --dataset CUB --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_baseline_seed7 --split novel --save_iter=2


python ./test.py --dataset miniImagenet --model Conv4 --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_aug_seed5 --method maml_approx
python ./test.py --dataset miniImagenet --model Conv4 --ckpt_dir /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_seed5 --method maml_approx
