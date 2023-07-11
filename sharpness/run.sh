CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_aug_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_aug_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_aug_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_5way_1shot_seed5/best_model.tar --split base 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_aug_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv4_maml_approx_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_aug_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model Conv6 \
--model_file /root/autodl-tmp/checkpoints/CUB/Conv6_maml_approx_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_aug_5way_1shot_seed5/best_model.tar --split novel 

CUDA_VISIBLE_DEVICES=0 python hessian.py --model ResNet18 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet18_maml_approx_5way_1shot_seed5/best_model.tar --split novel 





