
# -------------------
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64 #64
args['decoder_size'] = 128 #128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13, 3)
args['dyn_embedding_size'] = 32
args['dyn_matrix_and_centralit_input']=39
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 3
args['train_flag'] = True
args['dyn_matrix_and_centralit_output'] = 32

# Dimensionality of the input:
# 2D (X and Y or R and Theta)
# 3D (adding velocity as a 3d dimension)
args['input_dim'] = 3

# Using Intention module?
args['intention_module'] = True

# Choose the pooling mechanism
# -----------------------------
args['pooling'] = 'polar'

if args['pooling'] == 'slstm':
    args['kernel_size'] = (4, 3)

elif args['pooling'] == 'cslstm':
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16

elif args['pooling'] == 'sgan' or args['pooling'] == 'polar':
    args['bottleneck_dim'] = 256
    args['sgan_batch_norm'] = False
    

# ngsimDataset Class in utils.py and HighdDataset Class in utils_HighD.py
args['t_hist'] = 30
args['t_fut'] = 50
args['skip_factor'] = 2  # d_s

args['pretrainEpochs'] = 6
args['trainEpochs'] = 5

# Prediction horizon used in evaluation
args['pred_horiz'] = 5
