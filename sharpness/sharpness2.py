import torch.optim as optim
import json
import torch.utils.data.sampler
import scipy.stats as stats
import glob
import random
from torch.autograd import Variable, grad
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

def confident_interval(data):
        confidence_level = 0.95
        
        # Compute the sample mean and standard deviation
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

        # Calculate the margin of error
        sample_size = len(data)
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)  # Calculate the z-score for the given confidence level
        margin_of_error = z_critical * (std_dev / np.sqrt(sample_size))

        # Print the confidence interval in the form of mean ± interval
        return mean, margin_of_error

def _grad_norm(net):
    all_grad = []
    for name, weight in net.named_parameters():
        grad_n = weight.grad.data.norm(p=2)
        all_grad.append(grad_n)
    all_grad = torch.stack(all_grad)
    norm = torch.norm(all_grad,p=2)

    return norm

def _grad_norm2(grad):
    all_grad = []
    # print(grad)
    for g in grad:
        grad_n = g.data.norm(p=2)
        all_grad.append(grad_n)
    all_grad = torch.stack(all_grad)
    norm = torch.norm(all_grad,p=2)

    return norm


def flatten_parameters(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def hessian_vector_product(vector_parts,rrr,args,net):
    # loss = loss_function(net(data), target)
    lossq_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
    grad = torch.autograd.grad(lossq_mean, net.parameters(), create_graph=True, retain_graph=True)
    grad_vector_product = sum((g.reshape(-1) * v.reshape(-1)).sum() for g, v in zip(grad, vector_parts))
    hessian_vector = torch.autograd.grad(grad_vector_product, net.parameters(), retain_graph=True)
    return torch.cat([hv.reshape(-1) for hv in hessian_vector])


def lanczos_algorithm(net, args, k, dtype=torch.float32, device='cuda'):
    if args.split == "base":
        rrr = np.random.choice(range(0, 63), size=5, replace=False)
    elif args.split == "novel":
        rrr = np.random.choice(range(0, 20), size=5, replace=False)
    rrr = torch.tensor(rrr)
    
    weight = flatten_parameters(net)
    n = weight.numel()
    v = torch.zeros(k + 1, n, dtype=dtype, device=device)
    T = torch.zeros(k, k, dtype=dtype, device=device)

    v[0] = torch.randn(n, dtype=dtype, device=device)
    v[0] /= v[0].norm()
    
    vector_parts = [torch.zeros_like(p) for p in net.parameters()]

    for i in range(k):
        start_idx = 0
        for j, p in enumerate(net.parameters()):
            numel = p.numel()
            vector_parts[j] = v[i][start_idx : start_idx + numel].reshape(p.shape)
            start_idx += numel
        Hv = hessian_vector_product(vector_parts,rrr,args,net)
        alpha = torch.dot(v[i], Hv)
        T[i, i] = alpha

        if i + 1 < k:
            if i > 0:
                Hv -= beta * v[i - 1]

            beta = torch.norm(Hv)
            T[i, i + 1] = T[i + 1, i] = beta
            v[i + 1] = Hv / beta

    eigenvalues, eigenvectors = torch.symeig(T, eigenvectors=True)

    return eigenvalues, eigenvectors
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument("--alg", default="maml_approx",help="baseline/protonet/maml{_approx}" )
    parser.add_argument("--split", default="novel",help="base|val|novel" )
    parser.add_argument("--feat_file",default = "",help="the path to the created feature file" )
    parser.add_argument("--maml_approx", default=True)
    parser.add_argument("--rnddata", default=False,type=bool)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=1, type=int)
    parser.add_argument('--n_query', default=15, type=int)
    parser.add_argument('--task_update', default=1, type=int)
    parser.add_argument('--test_task', default=1, type=int, help='number of test tasks')
    parser.add_argument('--losstype', default="query", help='support or query')
    parser.add_argument('--freeze', default="None", help='conv/fc')
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--task', default=0, type=int)
    parser.add_argument('--model', default='Conv4', help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--loss', default="losss", help='losss/lossq')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic= True 
    torch.cuda.set_device(0)
    net = model_loader.load(args)
    original_weights = [param.clone().cuda() for param in net.parameters()]
    if "aug" in args.model_file:
        args.aug="aug"
    else:
        args.aug="no aug"
    sharp1_list = []
    lo_list = []
    lp_list = []
    results_file = "results_-12"+ str(args.rho)+ "_update_"+str(args.task_update)+ "_"+str(args.loss)+ ".txt"
    # Write the strings to a text file
    # with open(results_file, 'a') as f:
    #     f.write(" ")
    for _ in range(1):
        eigenvalues, eigenvectors = lanczos_algorithm(net, args,k=90, dtype=torch.float32, device='cuda')
    print(eigenvalues, eigenvectors)
    # mean, margin_of_error = confident_interval(sharp1_list)    
    # print(np.mean(sharp1_list),np.mean(lo_list),np.mean(lp_list))
    # list1_str = ', '.join(map(str, sharp1_list))
    # list3_str = ', '.join(map(str, lo_list))
    # list4_str = ', '.join(map(str, lp_list))
    
    
    # # Write the strings to a text file
    # with open(results_file, 'a') as f:
    #     f.write(args.model+ "\n")
    #     f.write(args.aug+ "\n")
    #     f.write(args.split+ "\n")
    #     f.write(args.loss+ "\n")
    #     f.write("sharpness: " + "Confidence interval: {:.8f} ± {:.8f}".format(mean, margin_of_error)+"_____lo: "+str(np.mean(lo_list))+"lp: "+str(np.mean(lp_list)) + "\n")
    #     f.write("\n")