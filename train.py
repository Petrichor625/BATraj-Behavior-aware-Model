from __future__ import print_function
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest
from torch.utils.data import DataLoader
import time
import math
import warnings
import numpy as np
from torch.optim import lr_scheduler

from model_args import args

import warnings
warnings.filterwarnings('ignore')   

# Import the argumnets
from model_args import args
device = "cuda:0" or "cuda:1" if args["use_cuda"] else "cpu"
# Ignore the warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize network
# ------------------
net = highwayNet(args)
if args['use_cuda']:
    net = net.to(device)

net = torch.nn.DataParallel(net, device_ids=None, output_device=None, dim=0)


# Initialize optimizer
# ---------------------
pretrainEpochs = args['pretrainEpochs']
trainEpochs = args['trainEpochs']
optimizer = torch.optim.Adam(net.parameters(),lr=0.001) 
scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=50,T_mult=1,eta_min=0, last_epoch=-1, verbose=True)


batch_size = 256
crossEnt = torch.nn.BCELoss()  

# Initialize data loaders
#valSet = ngsimDataset('data/ValSet.mat')
valSet = ngsimDataset('/data/ValSet.mat')

trSet = ngsimDataset('/data/TrainSet.mat')


trDataloader = DataLoader(trSet, batch_size=batch_size,persistent_workers = True,prefetch_factor =4,
                          shuffle=True, num_workers=16,drop_last=True,collate_fn=trSet.collate_fn,pin_memory=True)


valDataloader = DataLoader(valSet, batch_size=batch_size,persistent_workers = True,prefetch_factor =4,
                           shuffle=True, num_workers=16,drop_last=True,collate_fn=valSet.collate_fn,pin_memory=True)

# Initialize Train and validation loss:
train_loss = []
val_loss = []
prev_val_loss = math.inf


# Main training
# -------------
for epoch_num in range(pretrainEpochs+trainEpochs+2):

    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')
    elif epoch_num == trainEpochs+pretrainEpochs:
        print('Pre-training with MSE loss again')

    # Train:
    # -----------------------------------------------------------------------------
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):

        st_time = time.time()

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



        # If using the intention module
        if args['intention_module']:
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:

                fut_pred, _, _, _ = net(hist, nbrs, mask,  all_adjancent_matrix_mean, 
                                     closeness_list_batch, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean, lat_enc, lon_enc)

                l = maskedMSE(fut_pred, fut, op_mask)

             
                
            elif  pretrainEpochs<= epoch_num  and epoch_num < trainEpochs+pretrainEpochs:
                fut_pred, lat_pred, lon_pred,_ = net(
                    hist, nbrs, mask,  all_adjancent_matrix_mean, 
                                     closeness_list_batch, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean, lat_enc, lon_enc)

                # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) \
                    + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                    lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / \
                    lon_enc.size()[0]

            else:
                fut_pred, _, _, _ = net(hist, nbrs, mask,  all_adjancent_matrix_mean, 
                                     closeness_list_batch, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
        # Without the intention prediction
        else:
            fut_pred , _= net(hist, nbrs, mask, all_adjancent_matrix_mean,
                                     closeness_list_batch, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean)

            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)       
            


        optimizer.zero_grad()
        l.requires_grad_(True)
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        batch_time = time.time() - st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        

        # Printing
        if i % 100 == 99:
            eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)
            print("Epoch no:", epoch_num + 1,
                  "| Epoch progress(%):", format(
                      i / (len(trSet) / batch_size) * 100, '0.2f'),
                  "| Avg train loss:", format(avg_tr_loss / 100, '0.4f'),
                  "| Acc:", format(avg_lat_acc, '0.4f'), format(
                      avg_lon_acc, '0.4f'),
                  "| Validation loss prev epoch", format(
                      prev_val_loss, '0.4f'),
                  "| ETA(s):", int(eta))

            
            train_loss.append(avg_tr_loss / 100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0

    
            scheduler.step()
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Model Saving
# -------------
model_fname = '/home/trained_models'+args['pooling']
if args['intention_module']:

    if args['input_dim'] == 3:
        model_fname = model_fname + '_vel_int.tar'
    else:
        model_fname = model_fname + '_int.tar'

else:
    if args['input_dim'] == 3:
        model_fname = model_fname + '_vel.tar'
    else:
        model_fname = model_fname + '.tar'

torch.save(net.state_dict(), model_fname)
#print('---- Model Saving end ----')