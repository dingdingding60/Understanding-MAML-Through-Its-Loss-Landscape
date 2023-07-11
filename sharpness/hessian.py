import torch.optim as optim
import json
import torch.utils.data.sampler
import scipy.stats as stats
import glob
import random
import tqdm
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
def hessian_matmul(net,args,v,rrr):

    net.zero_grad()
    
    lossq_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
    v.requires_grad = False

    grad_result = torch.autograd.grad(
                lossq_mean,
                (p for p in net.parameters() if p.requires_grad),
                create_graph=True
    )
    grad_result = torch.cat(tuple(p.view(1, -1) for p in grad_result), 1)
    grad_result.backward(v.view(1, -1))

    result = torch.cat(
        tuple(p.grad.view(1, -1) for p in net.parameters() if p.requires_grad),
        1
    )

    return result.squeeze(0)



def calculate_hessian(net, args,m):
    # rrr = np.random.randint(high=19, low=0, size=5)
    
    buffer=m
    if args.dataset == "miniImagenet":
        if args.split == "base":
            rrr = np.random.choice(range(0, 63), size=5, replace=False)
        elif args.split == "novel":
            rrr = np.random.choice(range(0, 20), size=5, replace=False)
    elif args.dataset == "CUB":
        if args.split == "base":
            rrr = np.random.choice(range(0, 99), size=5, replace=False)
        elif args.split == "novel":
            rrr = np.random.choice(range(0, 49), size=5, replace=False)
    rrr = torch.tensor(rrr)
    n = sum(p.data.numel() for p in net.parameters() if p.requires_grad)
    assert n >= m
    assert buffer >= 2
    v = torch.ones(n).to("cuda")
    v /= torch.norm(v)
    print("[Batch-mode LANCZOS Algorithm running]")
    w = hessian_matmul(net,args,v,rrr)
    alpha = []
    alpha.append(w.dot(v))
    w -= alpha[0] * v
    V = [v]
    beta = []
    for i in tqdm.tqdm(range(1, m)):
        b = torch.norm(w)
        beta.append(b)
        if b > 0:
            v = w / b
        else:
            done = False
            k = 0
            while not done:
                k += 1
                v = torch.rand(n).to(w.device)

                for v_ in V:
                    v_ = v_.to(v.device)
                    v -= v.dot(v_) * v_

                done = torch.norm(v) > 0
                if k > 2 and not done: # This shouldn't happen even twice
                    raise Exception("Can't find orthogonal vector")

        # Full re-orthogonalization
        for v_ in V:
            v_ = v_.to(v.device)
            v -= v.dot(v_) * v_

        v /= torch.norm(v)
        V.append(v)

        # Saving GPU memory
        if len(V) > buffer:
            V[-buffer - 1] = V[-buffer - 1].cpu()

        w = hessian_matmul(net,args,v,rrr)
        alpha.append(w.dot(v))
        w = w - alpha[-1] * V[-1] - beta[-1] * V[-2]

    T = torch.diag(torch.Tensor(alpha))
    for i in range(m - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]

    V = torch.cat(tuple(v.cpu().unsqueeze(0) for v in V), 0)
    return T, V
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument("--alg", default="maml_approx",help="baseline/protonet/maml{_approx}" )
    parser.add_argument("--split", default="novel",help="base|val|novel" )
    parser.add_argument("--feat_file",default = "",help="the path to the created feature file" )
    parser.add_argument("--dataset",default = "miniImagenet",help="the path to the created feature file" )
    parser.add_argument("--maml_approx", default=True)
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
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--loss', default="lossq", help='losss/lossq')
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

    results_file = "eigen_"+ args.dataset+"_update"+str(args.task_update)+ "_"+str(args.loss)+ ".txt"
    # Write the strings to a text file
    with open(results_file, 'a') as f:
        f.write(" ")
    eigenmax_list = []
    eiginratio_list = []
    for _ in range(100):
        T,v= calculate_hessian(net, args,m=15)
        T_np = T.numpy()  # Convert the T matrix to a numpy array

        # Compute the eigenvalues of the tridiagonal matrix T
        eigenvalues, _ = np.linalg.eig(T_np)
        
        # Sort the eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenratio = eigenvalues[0]/eigenvalues[4]
        eigenmax_list.append(eigenvalues[0])
        eiginratio_list.append(eigenratio)
    # Write the strings to a text file
    meanmax, margin_of_error = confident_interval(eigenmax_list)   
    meanratio, margin_of_erro1 = confident_interval(eiginratio_list)  
    with open(results_file, 'a') as f:
        f.write(args.model+ "\n")
        f.write(args.aug+ "\n")
        f.write(args.split+ "\n")
        # f.write(args.loss+ "\n")
        # f.write(str(np.mean(eigenvalue_list,axis=0)))
        f.write(str(meanmax)+"+_"+str(margin_of_error)+"\n")
        f.write(str(meanratio)+"+_"+str(margin_of_erro1)+"\n")
        f.write("\n")