args = {}
import random
import numpy as np
import torch as t
import model.model5f_BAT_new as model_new
args['path'] = 'Location of model training parameters (.tar)'
# ------------------------------------------------------------------------- 
# Settings
seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" or "cuda:1" if t.cuda.is_available() else "cpu")

learning_rate = 0.001
dataset = "ngsim"  

args['num_worker'] = 16
args['device'] = device
args['lstm_encoder_size'] = 64
args['lstm_encoder_carnum'] = 39
args['lstm_encoder_size_behavior'] = 16 #8
args['n_head'] = 4
args['att_out'] = 48
args['in_length'] = 16
args['out_length'] = 25
args['f_length'] = 12
args['traj_linear_hidden'] = 32
args['behavior_size'] = 6
args['traj_linear_behavior'] = 4 #4
args['batch_size'] = 32
args['use_elu'] = True
args['dropout'] = 0
args['relu'] = 0.1
args['lat_length'] = 3
args['lon_length'] = 3  # 2
args['use_true_man'] = False
args['epoch'] = 15
args['use_spatial'] = False


args['use_cuda'] = True
args['gpu'] = 0 or 1
args['obs_len'] = 16
args['pred_len'] = 25
args['in_channels'] = 4
args['h_dim1'] = 128
args['h_dim2'] = 64
args['M'] = 4
args['z_dim'] = 64
args['gmm_emb_dim'] = 128

assert args['h_dim2'] == args['z_dim']


args['use_maneuvers'] = True
args['cat_pred'] = True
args['use_mse'] = False
args['pre_epoch'] = 7
args['val_use_mse'] = True


# -------------------------------------------------------------------------
