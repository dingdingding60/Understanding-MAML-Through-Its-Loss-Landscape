"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
from torchmeta.datasets.helpers import cifar_fs
from torchmeta.utils.data import BatchMetaDataLoader
import META.configs as configs
import META.backbone as backbone
import META.data.feature_loader as feat_loader
from META.data.datamgr import SetDataManager,SetDataManager_task
from META.methods.baselinetrain import BaselineTrain
from META.methods.baselinefinetune import BaselineFinetune
from META.methods.protonet import ProtoNet
from META.methods.maml import MAML
from META.io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc

def eval_loss(net,args,rrr):
    n_way=args.n_way
    k_shot=args.k_shot
    dataset=args.dataset
    alg=args.alg
    model=args.model
    test_task=args.test_task
    feature_file =args.feat_file
    split=args.split
    nquery=args.n_query
    print(dataset)
    criterion = nn.CrossEntropyLoss()
    few_shot_params = dict(n_way = n_way , n_support =k_shot)
    net.cuda()
    if alg in ['maml', 'maml_approx']: #maml do not support testing with feature
        if 'Conv' in model:
            if dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84 
        else:
            image_size = 224
            
        if args.rnddata:    
            datamgr = SetDataManager(image_size, n_eposide = test_task, n_query = nquery , **few_shot_params)
            
            loadfile    = configs.data_dir[dataset] + split + '.json'

            novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
            
            net.eval()
            print(1)
            acc_mean, acc_std,lossq_mean,losss_mean = net.test_loop( novel_loader,args, randdata = args.rnddata,return_std = True)
        else:
            print("=====fixed data=====")
            datamgr = SetDataManager_task(image_size, n_eposide = test_task, n_query = nquery , **few_shot_params)
            
            loadfile    = configs.data_dir[dataset] + split + '.json'

            novel_loader     = datamgr.get_data_loader( loadfile,args, aug = False,rrr = rrr)
            
            net.eval()
            acc_mean, acc_std,lossq_mean,losss_mean = net.test_loop( novel_loader,args, randdata = args.rnddata,return_std = True)
    else:
        # novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        novel_file = feature_file
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, net, n_query = nquery, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
    

    return lossq_mean, losss_mean, acc_mean
