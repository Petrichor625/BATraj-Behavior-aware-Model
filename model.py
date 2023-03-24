from __future__ import division
import torch
import torch.nn as nn
from utils import outputActivation
from pooling_module import nbrs_pooling
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import numba 
from numba import jit

class highwayNet(nn.Module):

    # Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        self.drop_path_prob = 0.0  
        # Unpack arguments
        self.args = args

        # Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        # Input Dimensionality
        self.input_dim = args['input_dim']  
        # Sizes of network layers
        self.encoder_size = args['encoder_size']  # LSTM encoder 64
        self.decoder_size = args['decoder_size']  # LSTM decoder 128
        self.in_length = args['in_length']  # input size
        self.out_length = args['out_length']  # output size
        self.grid_size = args['grid_size']  # 3*13
        

          
        self.dyn_matrix_and_centralit_input = args['dyn_matrix_and_centralit_input']
        self.dyn_matrix_and_centralit_output = args['dyn_matrix_and_centralit_output']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']  # 32
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.device = "cuda:0" or "cuda:1" if args["use_cuda"] else "cpu"
        self.dyn_embedding_size_DGG = args['dyn_matrix_and_centralit_output']*2 
        self.encoder_mean = round(self.encoder_size/8)
        self.dyn_embedding_mean_size = round(self.dyn_embedding_size/4)

        # Pooling Mechanism
        # --------------------
        self.pooling = args['pooling']

        if self.pooling == 'slstm':
            self.kernel_size = args['kernel_size']
            # soc_embedding_size
            pooled_size = self.grid_size[0] // self.kernel_size[0]
            if self.grid_size[0] % self.kernel_size[0] != 0:
                pooled_size += 1
            self.soc_embedding_size = self.encoder_size * pooled_size

        elif self.pooling == 'cslstm':
            self.soc_conv_depth = args['soc_conv_depth']
            self.conv_3x1_depth = args['conv_3x1_depth']
            self.soc_embedding_size = (
                ((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth

            # Convolutional social pooling layer and social embedding layer
            self.soc_conv = torch.nn.Conv2d(
                self.encoder_size, self.soc_conv_depth, 3)
            self.conv_3x1 = torch.nn.Conv2d(
                self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
            self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        elif self.pooling == 'polar':
            self.bottleneck_dim = args['bottleneck_dim']  # 256
            self.mlp_pre_dim = 2 * self.encoder_size #64*2
            self.soc_embedding_size = self.bottleneck_dim
            self.rel_pos_embedding = nn.Linear(self.input_dim, self.encoder_size)

            self.batch_norm = args['sgan_batch_norm'] 
            self.mlp_pre_pool = self.make_mlp(
                self.mlp_pre_dim, self.bottleneck_dim, self.batch_norm) #bottleneck_dim 256

        self.IA_module = args['intention_module']
        if self.IA_module:
            # Decoder LSTM
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size_DGG +self.num_lat_classes+ self.num_lon_classes, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size_DGG, self.decoder_size)

        # Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(
            self.input_dim, self.input_embedding_size)

        # for matrix_and_centralit
        self.ip_emb_1 = torch.nn.Linear(
            self.dyn_matrix_and_centralit_input, self.dyn_matrix_and_centralit_output)


        self.enc_lstm_1 = torch.nn.LSTM(
            self.dyn_matrix_and_centralit_output, self.encoder_mean)

        self.dyn_emb_1 = torch.nn.Linear(
            self.encoder_mean, self.dyn_embedding_mean_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(
            self.input_embedding_size, self.encoder_size, 1)


        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(
            self.encoder_size, self.dyn_embedding_size)

        self.pre4att = nn.Sequential(

            nn.Linear(64, 1),#64 is the final dimension shape of new_hs
        )

        self.tanh = nn.Tanh()

        if self.input_dim == 2:
            op_gauss_dim1 = 5
        elif self.input_dim == 3:
            op_gauss_dim1 = 7  
        
        self.op = torch.nn.Linear(self.decoder_size, op_gauss_dim1)  
        self.op_lat = torch.nn.Linear(
            self.soc_embedding_size + self.dyn_embedding_size_DGG, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(
            self.soc_embedding_size + self.dyn_embedding_size_DGG, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def attention(self, lstm_out_weight, lstm_out):
        # (batch_size, lstm_cell_num, 1), calculate weight along the 1st dimension
        alpha = F.softmax(lstm_out_weight, 1)
        lstm_out = lstm_out.permute(0, 2, 1)
        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)  # new_hidden_state-(batch_size, hidden_dim)
        new_hidden_state = F.relu(new_hidden_state)
        return new_hidden_state, alpha  # , soft_attn_weights_1#, soft_attn_weights_2


    def get_hist_1(self,hist):
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        #hist_enc_1 = hist_enc.squeeze()
        #hist_enc_1 = hist_enc.unsqueeze(2)
        return hist_enc


    # Forward Pass
    def forward(self, hist, nbrs, masks, all_adjancent_matrix_mean_batch,
                closeness_list_batch, degree_list_batch, eigenvector_list_batch, all_closeness_mean_batch, all_degree_mean_batch, all_eigenvector_mean_batch, lat_enc=None, lon_enc=None):


      
        # Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        #hist_enc_1 = hist_enc.squeeze()
        #hist_enc_1 = hist_enc.unsqueeze(2)
  
        hist_enc = self.leaky_relu(self.dyn_emb(
            hist_enc.view(hist_enc.shape[1], hist_enc.shape[2]))) 

        all_adjancent_matrix_mean_batch = all_adjancent_matrix_mean_batch.unsqueeze(0).to(self.device)
        
        all_closeness_mean_batch = all_closeness_mean_batch.unsqueeze(0).to(self.device)

        all_degree_mean_batch = all_degree_mean_batch.unsqueeze(0).to(self.device)

        all_eigenvector_mean_batch = all_eigenvector_mean_batch.unsqueeze(0).to(self.device)



        #_, (all_rates_list, _) = self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(all_rates_list_batch)))
        #all_rates_list = self.leaky_relu(self.dyn_emb(all_rates_list)) 

        '''
        _, (closeness_list, _) = self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(closeness_list_batch)))
        closeness_list = self.leaky_relu(self.dyn_emb(closeness_list.view(closeness_list.shape[1],closeness_list.shape[2]))) 
        
        _, (degree_list, _) = self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(degree_list_batch)))
        degree_list = self.leaky_relu(self.dyn_emb(degree_list.view(degree_list.shape[1],degree_list.shape[2]))) 

        _, (eigenvector_list, _) = self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(eigenvector_list_batch)))
        eigenvector_list = self.leaky_relu(self.dyn_emb(eigenvector_list.view(eigenvector_list.shape[1],eigenvector_list.shape[2]))) 
        '''
        
        _, (all_adjancent_matrix_mean, _)= self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(all_adjancent_matrix_mean_batch)))#这个或许可以不要
        all_adjancent_matrix_mean = self.leaky_relu(self.dyn_emb_1(all_adjancent_matrix_mean.view(all_adjancent_matrix_mean.shape[1],all_adjancent_matrix_mean.shape[2])))

        
        _, (all_closeness_mean, _ )= self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(all_closeness_mean_batch)))
        all_closeness_mean = self.leaky_relu(self.dyn_emb_1(all_closeness_mean.view(all_closeness_mean.shape[1],all_closeness_mean.shape[2]))) 


        _, (all_degree_mean, _) = self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(all_degree_mean_batch)))
        all_degree_mean = self.leaky_relu(self.dyn_emb_1(all_degree_mean.view(all_degree_mean.shape[1],all_degree_mean.shape[2]))) 

        _, (all_eigenvector_mean, _ )= self.enc_lstm_1(self.leaky_relu(self.ip_emb_1(all_eigenvector_mean_batch)))
        all_eigenvector_mean = self.leaky_relu(self.dyn_emb_1(all_eigenvector_mean.view(all_eigenvector_mean.shape[1],all_eigenvector_mean.shape[2]))) 


        # Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        # Social Tensor using Masked scatter
        soc_enc = torch.zeros_like(masks).float()  
        soc_enc = soc_enc.masked_scatter_(masks.bool(), nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)
        soc_enc = soc_enc.contiguous()

        hist_enc_1 = self.get_hist_1(hist)
    

        # Pooling
        # ---------
        soc_enc_pooled, soft_attn_weights_ha = nbrs_pooling(
            self, soc_enc, masks, nbrs, nbrs_enc,hist_enc_1) 
        
        #all_adjancent_matrix_mean = all_adjancent_matrix_mean.squeeze()
        mean_enc = torch.cat((all_adjancent_matrix_mean,all_closeness_mean,all_degree_mean,all_eigenvector_mean), 1)

        #enc = torch.cat((soc_enc_pooled, hist_enc,
                        #all_adjancent_matrix_mean,closeness_list,degree_list,eigenvector_list,all_closeness_mean,all_degree_mean,all_eigenvector_mean), 1)
        
        enc = torch.cat((soc_enc_pooled, hist_enc,mean_enc), 1) 

        if self.IA_module:
            # Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                # Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)


            else:
                fut_pred = []
                # Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))                                                  
            return fut_pred, lat_pred, lon_pred, soft_attn_weights_ha

        else:
            fut_pred = self.decode(enc)
            return fut_pred, soft_attn_weights_ha

    # Decoder Module
    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)

        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        
        return fut_pred

    # MLP
    def make_mlp(self, dim_in, dim_out, batch_norm):
        if batch_norm:
            layers = [nn.Linear(dim_in, dim_out),
                      nn.BatchNorm1d(dim_out), nn.ReLU()]
        else:
            layers = [nn.Linear(dim_in, dim_out), nn.ReLU()]
        return nn.Sequential(*layers)
