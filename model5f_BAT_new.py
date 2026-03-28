import math
import numpy as np
import torch as t
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class GDEncoder(nn.Module):
    def __init__(self, args):
        super(GDEncoder, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden'] #32
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']
        self.behavior_size = args['behavior_size']
        self.traj_linear_hidden_behavior = args['traj_linear_behavior'] #6
        self.lstm_encoder_size_behavior = args['lstm_encoder_size_behavior'] #16
        self.lstm_car_num = args['lstm_encoder_carnum'] #39
        # traj encoder
        self.linear1 = nn.Linear(8, self.traj_linear_hidden)

        self.linear_behavior = nn.Linear(self.behavior_size, self.traj_linear_hidden_behavior)
        self.lstm_behavior = nn.LSTM(self.traj_linear_hidden_behavior * self.lstm_car_num, self.lstm_encoder_size)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size_behavior)
        
        self.pre_glu_proj = nn.Linear(in_features=self.n_head * self.att_out // 2, out_features=self.n_head * self.att_out)

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.activation_behavior = self.leaky_relu = t.nn.LeakyReLU(self.relu_param)
        self.qff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        self.second_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)

        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)

        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)

  
        self.attention = SelfAttention(input_dim=self.lstm_encoder_size_behavior)

    def pool_neighbor_features(self, nbrs_feat, mask):
        T, N_all, H = nbrs_feat.shape
        B, gy, gx, _ = mask.shape

        social_tensor = nbrs_feat.new_zeros(B, gy, gx, T, H)
        social_valid = nbrs_feat.new_zeros(B, gy, gx, 1, 1)

        count = 0
        for b in range(B):
            for y in range(gy):
                for x in range(gx):
                    if mask[b, y, x, 0]:
                        social_tensor[b, y, x, :, :] = nbrs_feat[:, count, :]
                        social_valid[b, y, x, 0, 0] = 1.0
                        count += 1

        summed = social_tensor.sum(dim=(1, 2))              # [B, T, H]
        denom = social_valid.sum(dim=(1, 2)).clamp(min=1.) # [B,1,1]
        pooled = summed / denom
        return pooled

    def forward(self, hist, nbrs, hist_relative, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, nbrs_ref_self, nbrs_ref_nbrs,feature_matrix,BLE_BIE):
        if self.f_length == 11:
            hist = t.cat((hist, hist_relative, cls, va), -1)
            nbrs = t.cat((nbrs,nbrs_ref_self, nbrs_ref_nbrs,nbrscls, nbrsva), -1)

        elif self.f_length == 12:
            hist = t.cat((hist,hist_relative, cls, va, lane), -1)
            nbrs = t.cat((nbrs, nbrs_ref_self,nbrscls, nbrsva, nbrslane), -1) 
 
        hist_enc = self.activation(self.linear1(hist))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        hist_hidden_enc = hist_hidden_enc.permute(1, 0, 2)
    

        time_steps, batch_size, cars_num, features = self.linear_behavior(BLE_BIE).size()
        BLE_BIE_enc_reshaped = BLE_BIE.permute(1, 0, 2, 3).contiguous() 
        BLE_BIE_enc = self.activation_behavior(self.linear_behavior(BLE_BIE_enc_reshaped))
        batch_size, time_steps, cars_num, features = BLE_BIE_enc.shape
        BLE_BIE_enc = BLE_BIE_enc.reshape(batch_size, time_steps, cars_num * features)


        BLE_BIE_enc, (_, _) = self.lstm_behavior(BLE_BIE_enc)
        BLE_BIE_enc = BLE_BIE_enc.permute(1, 0, 2)     



        nbrs_enc = self.activation(self.linear1(nbrs))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc) #[16, 224, 16]
        attended_nbrs_hidden_enc = self.attention(nbrs_hidden_enc)#[16, 257, 16]

        BLE_BIE_enc = BLE_BIE_enc.permute(1, 0, 2)
        attended_nbrs_hidden_enc = self.pool_neighbor_features(attended_nbrs_hidden_enc, mask)  # [B, T, H]
      
        output_values = t.cat((hist_hidden_enc,attended_nbrs_hidden_enc,BLE_BIE_enc),-1)
       
        output_values = self.pre_glu_proj(output_values) 
        output_values, _ = self.first_glu(output_values)
   
        return output_values


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = t.exp(sigX)
    sigY = t.exp(sigY)
    rho = t.tanh(rho)
    out = t.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size) 

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)#
        else:
            x = t.add(x1, x2)
        return self.normalize(x)#


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.relu_param = args['relu']
        self.use_elu = args['use_elu']
        self.use_maneuvers = args['use_maneuvers']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.device = args['device']
        self.cat_pred = args['cat_pred']
        self.use_mse = args['use_mse']
        self.lon_length = args['lon_length']
        self.lat_length = args['lat_length']
        if self.use_maneuvers or self.cat_pred:
            self.mu_f = 16
        else:
            self.mu_f = 0
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.lstm = t.nn.LSTM(self.encoder_size, self.encoder_size)
        if self.use_mse:
            self.linear1 = nn.Linear(self.encoder_size, 2)
        else:
            self.linear1 = nn.Linear(self.encoder_size, 5)
        self.lat_linear = nn.Linear(self.lat_length, 8)
        self.lon_linear = nn.Linear(self.lon_length, 8)

        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):

        if self.use_maneuvers or self.cat_pred:
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = t.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)
      
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)
        if self.use_mse:
            return fut_pred
        else:
            return outputActivation(fut_pred)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.train_flag = args['train_flag']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.lat_length = args['lat_length']
        self.lon_length = args['lon_length']
        self.use_elu = args['use_elu']
        self.use_true_man = args['use_true_man']
        self.Decoder = Decoder(args=args)
        self.mu_fc1 = t.nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.mu_fc = t.nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.op_lat = t.nn.Linear(self.lstm_encoder_size, self.lat_length)
        self.op_lon = t.nn.Linear(self.lstm_encoder_size, self.lon_length)

        

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.normalize = nn.LayerNorm(self.lstm_encoder_size)
   

        self.mapping = t.nn.Parameter(t.Tensor(self.in_length, self.out_length, self.lat_length + self.lon_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Glorot init
        self.manmapping = t.nn.Parameter(t.Tensor(self.in_length, 1))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Glorot init

    def forward(self, values, lat_enc, lon_enc):
        maneuver_state = values[:, -1, :]
        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        lat_pred = F.softmax(self.op_lat(maneuver_state), dim=-1)
        lon_pred = F.softmax(self.op_lon(maneuver_state), dim=-1)
        if self.train_flag:
            if self.use_true_man:
                lat_man = t.argmax(lat_enc, dim=-1).detach()
                lon_man = t.argmax(lon_enc, dim=-1).detach()
            else:
                lat_man = t.argmax(lat_pred, dim=-1).detach().unsqueeze(1)
                lon_man = t.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = t.zeros_like(lat_pred)
                lon_enc_tmp = t.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)
            index = t.cat((lat_man, lon_man), dim=-1).permute(-1, 0)
            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)
            if self.use_maneuvers:
                fut_pred = self.Decoder(dec, lat_enc, lon_enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = self.Decoder(dec, lat_pred, lon_pred)
                return fut_pred, lat_pred, lon_pred
        else:
            out = []
            for k in range(self.lon_length):
                for l in range(self.lat_length):
                    lat_enc_tmp = t.zeros_like(lat_enc)
                    lon_enc_tmp = t.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    index = t.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)
                    mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                    dec = t.matmul(mapping, values).permute(1, 0, 2)
                    fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)
                    out.append(fut_pred)
            return out, lat_pred, lon_pred

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.linear_query = nn.Linear(input_dim, input_dim)
        self.linear_key = nn.Linear(input_dim, input_dim)
        self.linear_value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [T, N, H]
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)

        scores = t.matmul(query, key.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attention_weights = self.softmax(scores)
        attended_value = t.matmul(attention_weights, value)

        return attended_value



# gate
class GLU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x)) 
        return t.mul(activation, gated), gated 
