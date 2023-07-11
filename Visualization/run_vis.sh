#meta loss landscape
CUDA_VISIBLE_DEVICES=1 python plot_surface.py --cuda --plot --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=4 --task_update=10 --n_query=15 --x=-1:1:51 --y=-1:1:51 --dataset miniImagenet

#single task loss landscape
CUDA_VISIBLE_DEVICES=1 python plot_surface.py --cuda --plot --model Conv4 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=1 --task_update=10 --n_query=1 --x=-1:1:51 --y=-1:1:51 --dataset miniImagenet --support_loss
