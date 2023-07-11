CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2


CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=1

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset miniImagenet --task=2


CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2 --freeze conv

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2 --freeze fc

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=1

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/CUB/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=1 --task_update=0 --n_query=1 --x=-1:1:40 --y=-1:1:40 --dataset CUB --task=2