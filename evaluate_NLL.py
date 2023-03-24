from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedMSETest, horiz_eval,maskedNLLTest
from torch.utils.data import DataLoader
import time
import numpy as np
from pathlib import Path
import pandas as pd
from model_args import args
#Ignore the warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = "cuda:0" or "cuda:1" if args["use_cuda"] else "cpu"
#Import the argumnets
from model_args import args
metric = 'nll'
# Evaluation mode
args['train_flag'] = False
pred_horiz = args['pred_horiz']

# Initialize network
# ------------------
net = highwayNet(args)

# load the trained model
net_fname = '/home/trained_models/' + args['pooling']
if args['intention_module']:
    if args['input_dim']==3:
        net_fname = net_fname + '_vel_int.tar'
    else:
        net_fname = net_fname + '_int.tar'
else:
    if args['input_dim'] == 3:
        net_fname = net_fname + '_vel.tar'
    else:
        net_fname = net_fname + '.tar'

pretrain = torch.load(net_fname, map_location='cpu')

new_state_dict = {}
for k,v in pretrain.items():
    new_state_dict[k[7:]] = v

if (args['use_cuda']):
    net.load_state_dict(new_state_dict,strict=True)
    net = net.to(device)
else:
    checkpoint = torch.load(net_fname, map_location='cpu')
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    net.load_state_dict(checkpoint)

# Test 


tsSet = ngsimDataset('/data/TestSet.mat')
tsDataloader = DataLoader(tsSet, batch_size=256, shuffle=True, num_workers=16, persistent_workers = True,prefetch_factor =4,collate_fn=tsSet.collate_fn,pin_memory=True)

if args['intention_module']:
    if args['input_dim']==3:
        eval_fname = '/evaluation/train_model' + args['pooling'] + '_int_vel_nll.csv'
    else:
        eval_fname = '/evaluation/train_model/' + args['pooling'] + '_int_nll.csv'
else:
    if args['input_dim'] == 3:
        eval_fname = '/evaluation/train_model/' + args['pooling'] + '_vel.csv'
    else:
        eval_fname = '/evaluation/train_model/' + args['pooling'] + '.csv'


  
lossVals = torch.zeros(args['out_length'])
counts = torch.zeros(args['out_length'])

if args['use_cuda']:

    lossVals = lossVals.to(device)
    counts = counts.to(device)

    for i, data in enumerate(tsDataloader):

        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, \
        ds_ids, vehicle_ids, frame_ids,\
        all_adjancent_matrix_mean_batch,\
        closeness_list_batch, degree_list_batch, eigenvector_list_batch, all_closeness_mean_batch, all_degree_mean_batch, all_eigenvector_mean_batch = data

        if args['use_cuda']:
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut.to(device)
            op_mask = op_mask.to(device)
            ds_ids = ds_ids.to(device)
            vehicle_ids = vehicle_ids.to(device)
            frame_ids = frame_ids.to(device)
            all_adjancent_matrix_mean = all_adjancent_matrix_mean_batch.to(device)
            closeness_list = closeness_list_batch.to(device)
            degree_list = degree_list_batch.to(device)
            eigenvector_list = eigenvector_list_batch.to(device)
            all_closeness_mean = all_closeness_mean_batch.to(device)
            all_degree_mean = all_degree_mean_batch.to(device)
            all_eigenvector_mean = all_eigenvector_mean_batch.to(device)

        # Forward pass
        if args['intention_module']:
            fut_pred, lat_pred, lon_pred,_ = net(hist, nbrs, mask,  all_adjancent_matrix_mean, 
                                     closeness_list_batch, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean, lat_enc, lon_enc)

            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)

        lossVals += l.detach()
        counts += c.detach()

    
    # Calculate NLL in meters
print('nll test')
nll_result = lossVals / counts
print(lossVals / counts)       
# Saving Results to a csv file
df = pd.DataFrame(nll_result)
test_cases = ['overall']
df.to_csv(eval_fname, header=test_cases,index=False)






