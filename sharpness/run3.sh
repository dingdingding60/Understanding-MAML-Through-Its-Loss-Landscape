CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_aug_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_aug_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_aug_5way_1shot_seed7/2399.tar --split base --loss lossq --rho=0.05 --dataset CUB --seed=7



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_aug_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_aug_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_aug_5way_1shot_seed7/2399.tar --split novel --loss lossq --rho=0.05 --dataset CUB --seed=7

