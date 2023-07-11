python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=4 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=4 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=100 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=100 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=4 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=4 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=novel --test_task=100 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture

python plot_surface.py --cuda --plot --model ResNet10 \
--model_file /root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_aug_5way_1shot_seed5/best_model.tar \
--alg maml_approx --n_way=5 --k_shot=1 --split=base --test_task=100 --task_update=10 --n_query=15 --x=-1:1:50 --y=-1:1:50 --dataset miniImagenet --rnddata=Ture
