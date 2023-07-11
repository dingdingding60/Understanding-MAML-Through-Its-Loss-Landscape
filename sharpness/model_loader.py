import os
# import cifar10.model_loader
import META.backbone as backbone
from META.methods.baselinefinetune import BaselineFinetune
from META.methods.protonet import ProtoNet
from META.methods.maml import MAML
import torch
model_dict = dict(
            Conv4 =backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 
def load(args):
    # dataset=args.dataset
    alg=args.alg
    model_name=args.model
    model_file =args.model_file
    maml_aprox=args.maml_approx
    task_up=args.task_update
    test_n_way = args.n_way
    n_shot=args.k_shot
    # print(dataset,alg,model_name,maml_aprox,task_up,test_n_way,n_shot)
    few_shot_params = dict(n_way = test_n_way , n_support =n_shot, task_update = task_up)
    # if dataset in ['omniglot', 'cross_char']:
    #     assert model_name == 'Conv4' ,'omniglot only support Conv4 without augmentation'
    #     model_name = 'Conv4S'
    #     print("===")
    # if dataset == 'cifar10':
    #     net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    if alg == 'baseline':
        model= BaselineFinetune( model_dict[model_name], **few_shot_params )
    elif alg == 'protonet':
        model= ProtoNet( model_dict[model_name], **few_shot_params )
    elif alg in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[model_name], approx = (maml_aprox) , **few_shot_params )
        # if dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
        #     model.n_task     = 32
        #     model.task_update_num = 1
        #     model.train_lr = 0.1
        #     print("===")
    else:
       raise ValueError('Unknown method')
    # model = model.cuda()
    tmp = torch.load(model_file)
    model.load_state_dict(tmp['state'])
    
    return model
