import torch.optim as optim
import json
import torch.utils.data.sampler
import scipy.stats as stats
import glob
import random
from tqdm import tqdm
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
    if args.test_task ==1:
        datamgr = SetDataManager_task(image_size, n_eposide = test_task, n_query = nquery , **few_shot_params)
        
        loadfile    = configs.data_dir[dataset] + split + '.json'

        novel_loader     = datamgr.get_data_loader( loadfile,args, aug = False,rrr = rrr)
    else:
        datamgr = SetDataManager(image_size, n_eposide = test_task, n_query = nquery , **few_shot_params)
        loadfile    = configs.data_dir[dataset] + split + '.json'
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        args.rnddata = True
        net.eval()
        print(test_task)
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


def calculate_sharpness(net, args, p_norm, q_norm,original_weights):
    # rrr = np.random.randint(high=19, low=0, size=5)
    rho=args.rho
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
    lossq_mean, losss_mean, acc_mean = eval_loss(net,args,rrr)
    print("=======")
    grad = torch.autograd.grad(lossq_mean, net.parameters(), create_graph=True) #build full graph support gradient of gradient
    grad = [ g.detach()  for g in grad ]
    print("=======")
    # print(grad)
    # input()
    epsilons = []
    grad_norm = _grad_norm2(grad)
    scale = rho/(grad_norm+1e-12)
    for (name, weight),g in zip(net.named_parameters(),grad):
        if "weight" in name and not "BN" in name :  # Only consider weight tensors, not biases
            # Extract the gradient for the specified weight
            # print(weight)
            # print(g)
            # Calculate epsilon_hat
            # gradient_sign = torch.sign(weight_gradient)
            # gradient_abs_q_minus_1 = torch.abs(weight_gradient) ** (q_norm - 1)
            # equal = torch.allclose( gradient_sign * gradient_abs_q_minus_1, weight.grad.data)
            
            epsilon_hat = g * scale
            # print(epsilon_hat)
            # Calculate the sharpness for this layer and add to the total sharpness

            # Clear gradients for the current weight tensor
            
        else:
            epsilon_hat = torch.zeros_like(weight)
        if weight.grad is not None:
            weight.grad.detach_()
            weight.grad.zero_()
        epsilons.append(epsilon_hat)
    # Detach variables
    # print(epsilons[-2])
    perturbed_weights = [weight + epsilon for weight, epsilon in zip(original_weights, epsilons)]
    apply_perturbed_weights(net,perturbed_weights)
    lossq, losss, acc = eval_loss(net,args,rrr)
    sharpness = lossq.item()-lossq_mean.item()
    apply_perturbed_weights(net,original_weights)
    return sharpness,lossq_mean.item(),lossq.item()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument("--alg", default="maml_approx",help="baseline/protonet/maml{_approx}" )
    parser.add_argument("--split", default="novel",help="base|val|novel" )
    parser.add_argument("--dataset",default = "CUB",help="the path to the created feature file" )
    parser.add_argument("--feat_file",default = "",help="the path to the created feature file" )
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
    sharp1_list = []
    lo_list = []
    lp_list = []
    if "2399" in args.model_file:
         results_file = "Results2399"+ str(args.rho)+ "_"+args.dataset+"_"+str(args.test_task)+"task_seed"+str(args.seed)+".txt"
    else:
        results_file = "Results"+ str(args.rho)+ "_"+args.dataset+"_"+str(args.test_task)+"task_seed"+str(args.seed)+".txt"
    # Write the strings to a text file
    with open(results_file, 'a') as f:
        f.write(" ")
    if args.test_task==1:
        for _ in tqdm(range(100)):
            s,lo,lp= calculate_sharpness(net, args, 2, 2,original_weights)
            sharp1_list.append(s)
            lo_list.append(lo)
            lp_list.append(lp)
    else:
        for _ in tqdm(range(50)):
            s,lo,lp= calculate_sharpness(net, args, 2, 2,original_weights)
            input()
            sharp1_list.append(s)
            lo_list.append(lo)
            lp_list.append(lp)
    mean, margin_of_error = confident_interval(sharp1_list)    
    print(np.mean(sharp1_list),np.mean(lo_list),np.mean(lp_list))
    list1_str = ', '.join(map(str, sharp1_list))
    list3_str = ', '.join(map(str, lo_list))
    list4_str = ', '.join(map(str, lp_list))
    
    
    # Write the strings to a text file
    with open(results_file, 'a') as f:
        f.write(args.model+ "\n")
        f.write(args.aug+ "\n")
        f.write(args.split+ "\n")
        f.write(args.loss+ "\n")
        if "2399" in args.model_file:
            f.write("2399"+"\n")
        # f.write("sharpness: " + "Confidence interval: {:.8f} ± {:.8f}".format(mean, margin_of_error)+"_____lo: "+str(np.mean(lo_list))+"lp: "+str(np.mean(lp_list)) + "\n")
        # f.write("\n")
        f.write("sharpness: " + "Confidence interval: {:.8f} ± {:.8f}".format(mean, margin_of_error))
        f.write("\n")