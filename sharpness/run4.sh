# CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
# --model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

# CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
# --model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

# CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
# --model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

# CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
# --model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_aug_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_aug_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_aug_5way_1shot_seed6/best_model.tar --split base --loss lossq --rho=0.05 --dataset miniImagenet --seed=6



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6



CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_aug_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_aug_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6

CUDA_VISIBLE_DEVICES=1 python sharpness.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_aug_5way_1shot_seed6/best_model.tar --split novel --loss lossq --rho=0.05 --dataset miniImagenet --seed=6


