"""
    Plot the optimization path in the space spanned by principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader
import net_plotter
from projection import setup_PCA_directions, project_trajectory
import plot_2D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--dataset', default='miniImagenet', help='dataset')
    parser.add_argument('--model', default='Conv4', help='trained models')
    parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='biasbn', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--prefix', default='', help='prefix for the checkpint model')
    parser.add_argument('--suffix', default='.tar', help='prefix for the checkpint model')
    parser.add_argument('--start_epoch', default=0, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=239, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    parser.add_argument("--alg", default="maml_approx",help="baseline/protonet/maml{_approx}" )
    parser.add_argument("--split", default="novel",help="base|val|novel" )
    parser.add_argument("--feat_file",default = "",help="the path to the created feature file" )
    parser.add_argument("--maml_approx",action='store_true', default=False)
    parser.add_argument("--rnddata", default=False,type=bool)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--task_update', default=1, type=int)
    parser.add_argument('--test_task', default=1, type=int, help='number of test tasks')
    
    args = parser.parse_args()
    if "approx" in args.alg:
        args.maml_approx= True
    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    args.model_file = args.model_folder + '/' + args.prefix + str(args.max_epoch) + args.suffix
    # net = model_loader.load(args.dataset, args.model, last_model_file)
    net = model_loader.load(args)
    w = net_plotter.get_weights(net)
    s = net.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files = []
    for epoch in range(args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch):
        model_file = args.model_folder + '/' + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions(args, model_files, w, s)

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = project_trajectory(dir_file, w, s, args.dataset, args.model,
                                model_files, args,args.dir_type, 'cos')
    plot_2D.plot_trajectory(proj_file, dir_file)
