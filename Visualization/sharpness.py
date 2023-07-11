import torch.optim as optim
import json
import torch.utils.data.sampler

import glob
import random

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluationmeta as evaluation
from torch.autograd import Variable
import projection as proj
import net_plotter
import model_loader
import scheduler
from torchmeta.utils.data import BatchMetaDataLoader
from META.data.datamgr import SetDataManager,SetDataManager_task
import META.configs as configs
import META.backbone as backbone
import META.data.feature_loader as feat_loader
from META.data.datamgr import SetDataManager,SetDataManager_task
from META.methods.baselinetrain import BaselineTrain
from META.methods.baselinefinetune import BaselineFinetune
from META.methods.protonet import ProtoNet
from META.methods.maml import MAML
def eval_loss(net,args,rrr):
    n_way=args.n_way
    k_shot=args.k_shot
    dataset="miniImagenet"
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
    
    if 'Conv' in model:
        if dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224

    datamgr = SetDataManager_task(image_size, n_eposide = test_task, n_query = nquery , **few_shot_params)
    
    loadfile    = configs.data_dir[dataset] + split + '.json'

    novel_loader     = datamgr.get_data_loader( loadfile,args, aug = False,rrr = rrr)
    net.eval()
    acc_mean, acc_std,lossq_mean,losss_mean = net.test_loop( novel_loader,args, randdata = args.rnddata,return_std = True)
    return lossq_mean, losss_mean, acc_mean

def apply_perturbed_weights(net, perturbed_weights):
    for (name, param), perturbed_weight in zip(net.named_parameters(), perturbed_weights):
        with torch.no_grad():
            param.copy_(perturbed_weight)

def norm(weights):
    
    return torch.sqrt(sum(torch.norm(epsilon)**2 for epsilon in weights))

def grad_sharp(args,net,lr,dir_files):
    rrr = np.random.randint(high=19, low=0, size=5)
    rrr = torch.tensor(rrr)
    lossq_origin_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
    max_sharpness = 0
    lambda_ = torch.tensor(1e-5, requires_grad=True, device='cuda')
    d = net_plotter.load_directions(dir_files[1])
    scaled_data = []
    for dataset in d[0]:
        scaled_data.append(np.array(dataset) * 0.01)
    # epsilons = [torch.tensor(param, requires_grad=True)*0.001 for param in d[0]]
    epsilons = [torch.tensor(param, requires_grad=True, device='cuda') for param in scaled_data]
    # epsilons = [torch.randn_like(param, requires_grad=True) for param in net.parameters()]
    rho=100*norm(epsilons).item()
    # epsilons = [param.clone().detach().mul(0.001).requires_grad_(True).cuda() for param in d[0]]
    optimizer = optim.SGD(epsilons, lr=lr)
    original_weights = [param.clone().cuda() for param in net.parameters()]
    count=0
    old_s =-111
    while count<50:
        optimizer.zero_grad()
        
        perturbed_weights = [weight + epsilon for weight, epsilon in zip(original_weights, epsilons)]
        # Evaluate the model with perturbed weights (using a function to temporarily replace the weights)
        apply_perturbed_weights(net, perturbed_weights)
        lossq_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
        apply_perturbed_weights(net, original_weights)
        # sharpness = lossq_mean - lossq_origin_mean
        sharpness = lossq_mean 
        print(sharpness)
        if sharpness>max_sharpness:
            max_sharpness = sharpness
        if sharpness-old_s<0.01*sharpness:
            count+=1
        else:
            count=0
        old_s =sharpness
        constraint = norm(epsilons)-rho
        lagrangian = sharpness #+ constraint
        print("constraint:",constraint.item(),"rho:",rho)
        lagrangian = -lagrangian 
        lagrangian.backward()
        optimizer.step()
        with torch.no_grad():
            total_norm = norm(epsilons)
            if total_norm > rho:
                for epsilon in epsilons:
                    epsilon.mul_(rho / total_norm)
        
 
    return max_sharpness
        
def sharpness(args,net,dir_files):
    max_diff = 0
    original_weights = [param.clone().cuda() for param in net.parameters()]
    rrr = np.random.randint(high=19, low=0, size=5)
    rrr = torch.tensor(rrr)
    lossq_origin_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
    max_sharpness = 0
    lambda_ = torch.tensor(1e-5, requires_grad=True, device='cuda')
    d = net_plotter.load_directions(dir_files[1])
    scaled_data = []
    for dataset in d[0]:
        scaled_data.append(np.array(dataset) * 0.01)
    # epsilons = [torch.tensor(param, requires_grad=True)*0.001 for param in d[0]]
    epsilons = [torch.tensor(param, requires_grad=True, device='cuda') for param in scaled_data]
    # Initialize the optimizer for the perturbation variables
    optimizer = optim.Adam(epsilons, lr=1e-3)
    rho=100*norm(epsilons).item()
    # Maximize the difference for a fixed number of steps
    num_steps = 100
    for _ in range(num_steps):
        optimizer.zero_grad()

        # Compute perturbed weights
        perturbed_w = [perturbed_weights(w, torch.clamp(epsilon, -rho, rho)) for w, epsilon in zip(original_weights, w_tensors)]

        # Compute the loss for the original and perturbed weights
        original_loss = loss_function(network, original_weights)
        perturbed_loss = loss_function(network, perturbed_w)

        # Compute the difference
        diff = perturbed_loss - original_loss

        # Backpropagate the gradients
        diff.backward()
        optimizer.step()

        # Update the maximum difference
        max_diff = max(max_diff, diff.item())

    return max_diff        
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument("--alg", default="maml_approx",help="baseline/protonet/maml{_approx}" )
    parser.add_argument("--split", default="novel",help="base|val|novel" )
    parser.add_argument("--feat_file",default = "",help="the path to the created feature file" )
    parser.add_argument("--maml_approx",action='store_true', default=False)
    parser.add_argument("--rnddata", default=False,type=bool)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=1, type=int)
    parser.add_argument('--n_query', default=15, type=int)
    parser.add_argument('--task_update', default=10, type=int)
    parser.add_argument('--test_task', default=1, type=int, help='number of test tasks')
    parser.add_argument('--losstype', default="query", help='support or query')
    parser.add_argument('--freeze', default="None", help='conv/fc')
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--task', default=0, type=int)
    parser.add_argument('--model', default='Conv4', help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic= True 
    torch.cuda.set_device(0)
    net = model_loader.load(args)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.model=="Conv4":
        dir_file1 = "/root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_1ttsk_1query.h5"
        dir_file2 = '/root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_4ttsk_15query.h5'
        dir_file3 = "/root/autodl-tmp/checkpoints/miniImagenet/Conv4_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_100ttsk_15query.h5"
    elif args.model=="Conv6":
        dir_file1 = "/root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_1ttsk_1query.h5"
        dir_file2 = "/root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_4ttsk_15query.h5"
        dir_file3 = "/root/autodl-tmp/checkpoints/miniImagenet/Conv6_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_100ttsk_15query.h5"
    elif args.model=="ResNet10":
        dir_file1 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_4ttsk_15query.h5"
        dir_file2 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_100ttsk_15query.h5"
        dir_file3 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet10_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_novel_5way_1shot_1ttsk_1query.h5"
    elif args.model=="ResNet18":
        dir_file1 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_1ttsk_1query.h5"
        dir_file2 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_base_5way_1shot_4ttsk_15query.h5"
        dir_file3 ="/root/autodl-tmp/checkpoints/miniImagenet/ResNet18_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_novel_5way_1shot_1ttsk_1query.h5"
    elif args.model=="ResNet34":
        dir_file1 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet34_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_novel_5way_1shot_1ttsk_1query.h5"
        dir_file2 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet34_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_novel_5way_1shot_4ttsk_15query.h5"
        dir_file3 = "/root/autodl-tmp/checkpoints/miniImagenet/ResNet34_maml_approx_5way_1shot_seed5/best_model.tar_exp0_m_novel_5way_1shot_100ttsk_15query.h5"
    else:
        raise("bacbone don't exist")
    dir_files = [dir_file1, dir_file2, dir_file3]
    d1 = net_plotter.load_directions(dir_file1)
    rho = 50
    num_iterations = 100
    lr = 0.01
    s = grad_sharp(args,net,lr,dir_files)
    print(s)
    
    
    