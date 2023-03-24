from __future__ import division
from sympy import arg
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model import *
from model_args import args

device = "cuda:0" or "cuda:1" if args["use_cuda"] else "cpu"

# Main Poolig Function


'positon encoder'
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position = position.to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term = div_term.to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.to(device)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(device)
        return x + self.pe[:x.size(0), :]

def nbrs_pooling(net, soc_enc, masks, nbrs, nbrs_enc,hist_enc_1):#, hist_enc_1
    if net.pooling == 'slstm':
        soc_enc = s_pooling(net, soc_enc)
    elif net.pooling == 'cslstm':
        soc_enc = cs_pooling(net, soc_enc)
    elif net.pooling == 'sgan' or net.pooling == 'polar':
        soc_enc = polar_pooling(net, soc_enc, masks, nbrs, nbrs_enc,hist_enc_1) #, hist_enc_1
    return soc_enc


# SLSTM
def s_pooling(net, soc_enc):

    # Zero padding from bottom
    bottom_pad = net.grid_size[0] % net.kernel_size[0]

    if bottom_pad != 0:
        pad_layer = nn.ZeroPad2d((0, 0, 0, net.kernel_size[0] - bottom_pad))
        soc_enc = pad_layer(soc_enc)

    # Sum pooling
    avg_pool = torch.nn.AvgPool2d((net.kernel_size[0], net.kernel_size[1]))
    soc_enc = net.kernel_size[0] * net.kernel_size[1] * avg_pool(soc_enc)
    soc_enc = soc_enc.view(-1, net.kernel_size[0] * net.encoder_size)
    soc_enc = net.leaky_relu(soc_enc)

    return soc_enc


# CS-LSTM: Apply convolutional social pooling:
def cs_pooling(net, soc_enc):

    soc_enc = net.soc_maxpool(net.leaky_relu(
        net.conv_3x1(net.leaky_relu(net.soc_conv(soc_enc)))))
    soc_enc = soc_enc.view(-1, net.soc_embedding_size)
    return soc_enc

# Pooling operation used in the propoaed Polar-Pooling and SGAN

def polar_pooling(net, soc_enc, masks, nbrs, nbrs_enc,hist_enc_1):
    sum_masks = masks.sum(dim=3)
    soc_enc_1 = soc_enc
    soc_enc = torch.zeros(masks.shape[0], net.bottleneck_dim).float()
    if net.use_cuda:
        soc_enc = soc_enc.cuda()
        nbrs_enc = nbrs_enc.cuda()
        hist_enc_1 = hist_enc_1.cuda()
        soc_enc_1 = soc_enc_1.cuda()
 

    cntr = 0
    for ind in range(masks.shape[0]):
        no_nbrs = sum_masks[ind].nonzero().size()[0]
        if no_nbrs > 0:
            curr_nbr_pos = nbrs[:, cntr:cntr+no_nbrs, :]
            curr_nbr_enc = nbrs_enc[cntr:cntr+no_nbrs, :]
            cntr += no_nbrs

            end_nbr_pos = curr_nbr_pos[-1]
            
            soc_enc_1 = soc_enc_1.contiguous().view(soc_enc_1.shape[0], soc_enc_1.shape[1], -1) 
            hist_enc_1 = hist_enc_1.squeeze()
            hist_enc_1 = hist_enc_1.unsqueeze(2)
            #soc_enc_1  = soc_enc_1.squeeze() 
            
            new_hs = torch.cat((soc_enc_1,hist_enc_1), 2) 
            #position encoding

            pe = PositionalEncoding(d_model=40, max_len=5000)
            new_hs_per = pe(new_hs)

            #Attention
            new_hs_per = new_hs.permute(0, 2, 1)
            weight = net.pre4att(net.tanh(new_hs_per))
            new_hidden_ha, soft_attn_weights_ha = net.attention(weight, new_hs_per)
            new_hidden_ha =torch.cat([new_hidden_ha,new_hidden_ha],dim=1)

            #position-Embedding'
            rel_pos_embedding = net.rel_pos_embedding(end_nbr_pos) 
            mlp_h_input = torch.cat([rel_pos_embedding,curr_nbr_enc], dim=1)
            mlp_h_input = torch.cat([mlp_h_input,new_hidden_ha],dim=0)
            # if only 1 neighbor, BatchNormalization will not work
            # So calling model.eval() before feeding the data will change
            # the behavior of the BatchNorm layer to use the running estimates
            # instead of calculating them
            if mlp_h_input.shape[0] == 1 & net.batch_norm:
                net.mlp_pre_pool.eval()

            curr_pool_h = net.mlp_pre_pool(mlp_h_input)

            curr_pool_h = curr_pool_h.max(0)[0]
            soc_enc[ind] = curr_pool_h
    return soc_enc, soft_attn_weights_ha
