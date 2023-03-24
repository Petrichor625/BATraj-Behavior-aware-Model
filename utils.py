from __future__ import print_function, division
from turtle import shape
from matplotlib.pyplot import hist
from sqlalchemy import table
from sympy import re
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from model_args import args
import networkx as nx
import time
import math
from scipy.spatial.distance import pdist, squareform
# ___________________________________________________________________________________________________________________________

# Dataset class for the NGSIM dataset

class ngsimDataset(Dataset):
    def __init__(self, mat_file, t_h=args['t_hist'], t_f=args['t_fut'], d_s=args['skip_factor'],
                 enc_size=args['encoder_size'], grid_size=args['grid_size'], n_lat=args['num_lat_classes'],
                 n_lon=args['num_lon_classes'], input_dim=args['input_dim'], polar=args['pooling'] == 'polar'):
 
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']

        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  
        self.n_lat = n_lat  # num_lat_classes
        self.n_lon = n_lon  # num_lon_classes
        self.polar = polar  # pooling
        self.input_dim = input_dim-1

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)  
        vehId = self.D[idx, 1].astype(int) 
        t = self.D[idx, 2]  
        grid = self.D[idx, 8:]  
        neighbors = [] 
        radius = 32.8  
        hist = self.getHistory(vehId, t, vehId, dsId,
                               nbr_flag=False)  
      
        fut = self.getFuture(vehId, t, dsId, nbr_flag=False)
        
        for i in grid:  
            neighbors.append(self.getHistory(
                i.astype(int), t, vehId, dsId, nbr_flag=True))

        frame_ID_adj_mat_list, closeness_list, degree_list, eigenvector_list = self.get_all_adjancent_matrix_and_centrality(vehId, t, dsId, grid, radius)
     
        all_adjancent_matrix_mean, all_closeness_mean, all_degree_mean, all_eigenvector_mean = self.rate(
             frame_ID_adj_mat_list, closeness_list, degree_list, eigenvector_list)
      


        # Maneuvers 'lon_enc' = one-hot ve
        # ctor, 'lat_enc = one-hot vector
        lon_enc = np.zeros([self.n_lon])
        lon_enc[int(self.D[idx, 7] - 1)] = 1

        lat_enc = np.zeros([self.n_lat])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist, fut, neighbors, lat_enc, lon_enc, dsId, vehId, t,\
            torch.Tensor(all_adjancent_matrix_mean),  torch.Tensor(closeness_list),\
                torch.Tensor(degree_list), torch.Tensor(eigenvector_list), torch.Tensor(all_closeness_mean), torch.Tensor(all_degree_mean), torch.Tensor(all_eigenvector_mean)


    
    def get_coordinates_1(self, vehId, t, dsId): 
        if vehId == 0:
            nothing=[np.nan,np.nan]
            return nothing
        
            #else:
        if self.T.shape[1] <= vehId - 1:  
            nothing=[np.nan,np.nan]
            return nothing
       
        else:
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                nothing=[np.nan,np.nan]
                return nothing
            else:
                x = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1]
                y = vehTrack[np.where(vehTrack[:, 0] == t)][0, 2]
            return [x, y]


    def get_coordinates(self, vehId, t, dsId):  
        if vehId == 0:
            nothing=[np.nan,np.nan]
           
            return nothing
        else:
            if self.T.shape[1] <= vehId - 1:  
                nothing=[np.nan,np.nan]
                return nothing
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                nothing=[np.nan,np.nan]
                return nothing
            else:
                x = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1]
                y = vehTrack[np.where(vehTrack[:, 0] == t)][0, 2]
            return [x, y]
    

    def set_distance(self, a,radius):
        if a > radius:
            return float(0)
        else:
            return a

    def create_adjancent_matrix_1(self, vehId, t, dsId, grid, radius):

        lar = 39  # 3*13
        vehId_ind = round(lar/2)-1
        frame_ID_adj_mat_dict = {}
        grid[vehId_ind] = vehId.astype(int)
        grid_1 = [0,0]
        for i in grid:
            A=np.array(self.get_coordinates_1(i.astype(int),t,dsId))
            grid_1 = np.array(np.vstack((grid_1,A)))
        grid_1 = grid_1[1:]
        distance=np.array(pdist(grid_1, 'euclidean'))
        distance=np.array(np.nan_to_num(distance))
        adj_matrix_1 = np.array(squareform(distance))

        frame_ID_adj_mat_dict['frame_ID'] = t
        frame_ID_adj_mat_dict['adj_matrix'] = np.array(adj_matrix_1)  
        return frame_ID_adj_mat_dict
    
    def create_centrality(self, frame_ID_adj_mat_dict, t):
        closeness_1 = []
        degree_1 = []
        eigenvector_1 = []

        if frame_ID_adj_mat_dict['frame_ID'] == t:
            G = nx.from_numpy_array(np.array(frame_ID_adj_mat_dict['adj_matrix']))
            closeness = nx.closeness_centrality(G)
            degree = nx.degree_centrality(G)
            eigenvector = nx.eigenvector_centrality(G, max_iter=100000)
            for dic_1 in closeness:
                closeness_1.append(closeness[dic_1])
            for dic_2 in degree:
                degree_1.append(degree[dic_2])
         
            for dic_3 in eigenvector:
                eigenvector_1.append(eigenvector[dic_3])
            
            return np.array(closeness_1), np.array(degree_1), np.array(eigenvector_1)
        else:
            return np.empty([0,39,3])
   
    
    def get_global_adjancent_matrix(self, vehId, t, dsId, grid, radius):
        frame_ID_global_adjancent_list = []
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1: 
                return np.empty([0, 2])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            stpt = np.maximum(0, np.argwhere(
                vehTrack[:, 0] == t).item() - self.t_h)
            enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
            for item1 in range(stpt, enpt,2):
                t1 = vehTrack[item1, 0]
                frame_ID_adj_mat_dict = self.create_adjancent_matrix_1(
                    vehId, t1, dsId, grid, radius)
                frame_ID_global_adjancent_list.append(frame_ID_adj_mat_dict)
 
        return frame_ID_global_adjancent_list
    
    def get_all_adjancent_matrix_and_centrality(self, vehId, t, dsId, grid, radius):
        frame_ID_adj_mat_list = []
        closeness_list = []
        degree_list = []
        eigenvector_list = []
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:  
                return np.empty([0, 2])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            stpt = np.maximum(0, np.argwhere(
                vehTrack[:, 0] == t).item() - self.t_h)
            enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
            for item1 in range(stpt, enpt,2):
                t1 = vehTrack[item1, 0]
                frame_ID_adj_mat_dict = self.create_adjancent_matrix_1(
                    vehId, t1, dsId, grid, radius)
                frame_ID_adj_mat_list.append(frame_ID_adj_mat_dict)
                closeness, degree, eigenvector = self.create_centrality(
                    frame_ID_adj_mat_dict, t1)

                closeness_list.append(closeness)
                degree_list.append(degree)
                eigenvector_list.append(eigenvector)

            closeness_list = np.array(closeness_list)
            degree_list = np.array(degree_list)
            eigenvector_list = np.array(eigenvector_list)

        return frame_ID_adj_mat_list, closeness_list, degree_list, eigenvector_list


    def get_torch_Tensor_list(self,Mat_list):
        Numpy_array_list = []
        for item in Mat_list:
            matrix_list = item['adj_matrix']
            Numpy_array_list.append(np.array(matrix_list))

        return Numpy_array_list

    def rate(self, frame_ID_adj_mat_list, closeness_list, degree_list, eigenvector_list):
        num_of_agents = 39
        all_rates_list = []
        count = 1
        diags_list = []

        for list1 in frame_ID_adj_mat_list:
            adj = list1['adj_matrix']
            d_vals = []
            for item in adj:
                row_sum = sum(item)
                d_vals.append(row_sum)
            diag_array = np.diag(d_vals)
            laplacian = diag_array - adj
            L_diag = np.diag(laplacian)
            diags_list.append(np.asarray(L_diag))
        all_rates_arr = np.zeros_like(np.zeros([num_of_agents,1]))
        prev_ = diags_list[0]

        for items in range(1, len(diags_list)):
            next_ = diags_list[items]
            rate = next_ - prev_
            all_rates_arr = np.column_stack((all_rates_arr, rate))
            prev_ = next_
        all_rates_arr = np.delete(all_rates_arr, 0, 1)
        all_rates_list.append(all_rates_arr)  
    
        all_adjancent_matrix_mean = []
        all_rates_arr_1 = all_rates_list[0]
        for item in range(0, num_of_agents):
            avg=np.mean(all_rates_arr_1[item])
            all_adjancent_matrix_mean.append(avg)
        
        all_adjancent_matrix_mean = np.array(all_adjancent_matrix_mean)
        all_adjancent_matrix_mean=np.array(all_adjancent_matrix_mean)
        all_adjancent_matrix_mean =torch.Tensor(all_adjancent_matrix_mean)
        all_adjancent_matrix_mean = all_adjancent_matrix_mean.reshape(num_of_agents, 1)
        all_rates_list = np.array(all_rates_list)

        'closness mean'
        prev_ = closeness_list[0]
        all_rates_closeness_list = []
        for list2 in range(1, len(closeness_list)):
            next_ = closeness_list[list2]
            rate = [next_[i]-prev_[i] for i in range(0, len(prev_))]
            all_rates_closeness_list.append(rate)
            prev_ = next_

        all_rates_closeness_list = np.array(all_rates_closeness_list)
        all_rates_closeness_list = all_rates_closeness_list.reshape(num_of_agents,-1)
        all_closeness_mean = []

        for item1 in range(0, len(all_rates_closeness_list)):
            all_closeness_mean.append(np.mean(all_rates_closeness_list[item1]))

        all_closeness_mean = np.array(all_closeness_mean)
        all_closeness_mean = torch.Tensor(all_closeness_mean)
        all_closeness_mean = all_closeness_mean.reshape(num_of_agents, 1)


        'degree mean'
        prev_ = degree_list[0]
        all_rates_degree_list = []

        for list2 in range(1, len(degree_list)):
            next_ = degree_list[list2]
            rate = [next_[i]-prev_[i] for i in range(0, len(prev_))]
            all_rates_degree_list.append(rate)
            prev_ = next_

        all_degree_mean = []
        all_rates_degree_list = np.array(all_rates_degree_list)
        all_rates_degree_list = all_rates_degree_list.reshape(num_of_agents,-1)

        for item2 in range(0, len(all_rates_degree_list)):
            all_degree_mean.append(np.mean(all_rates_degree_list[item2]))

        all_degree_mean = np.array(all_degree_mean)
        all_degree_mean = torch.Tensor(all_degree_mean)
        all_degree_mean = all_degree_mean.reshape(num_of_agents, 1)

        'eigenvector mean'
        prev_ = eigenvector_list[0]
        all_rates_eigenvector_list = []

        for list3 in range(1, len(eigenvector_list)):
            next_ = eigenvector_list[list3]
            rate = [next_[i]-prev_[i] for i in range(0, len(prev_))]
            all_rates_eigenvector_list.append(rate)
            prev_ = next_

        all_eigenvector_mean = []
        all_rates_eigenvector_list = np.array(all_rates_eigenvector_list)
        all_rates_eigenvector_list = all_rates_eigenvector_list.reshape(num_of_agents,-1)

        for item3 in range(0, len(all_rates_eigenvector_list)):
            all_eigenvector_mean.append(
                np.mean(all_rates_eigenvector_list[item3]))
        all_eigenvector_mean = np.array(all_eigenvector_mean)
        all_eigenvector_mean = torch.Tensor(all_eigenvector_mean)
        all_eigenvector_mean = all_eigenvector_mean.reshape(num_of_agents, 1)

        return all_adjancent_matrix_mean, \
            all_closeness_mean, all_degree_mean, all_eigenvector_mean 

    def getHistory(self, vehId, t, refVehId, dsId, nbr_flag):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            inp_size = self.input_dim + 1
            if self.T.shape[1] <= vehId - 1:  
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:inp_size]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
    
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
              
                hist = vehTrack[stpt:enpt:self.d_s, 1:inp_size] - refPos
                polar = self.polar
                if polar:
                    hist = self.cart2polar(hist, nbr_flag)  

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist  

    def create_degree_num(self, neighbors):
      
        neighbors_num = 0
        for i in range(neighbors[0]):
            if len(neighbors[i]) == 0:
                neighbors_num += 0
            else:
                neighbors_num += 1
        return neighbors_num

    # Helper function to get track future

    def getFuture(self, vehId, t, dsId, nbr_flag):
        inp_size = self.input_dim + 1
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:inp_size]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(
            vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:inp_size] - refPos
        polar = self.polar
        if polar:
            fut = self.cart2polar(fut, nbr_flag)

        return fut

    def cart2polar(self, car_traj, nbr_flag):  
        # np.seterr(divide='ignore', invalid='ignore')
        r_traj = np.sqrt(np.square(car_traj[:, 0]) + np.square(car_traj[:, 1]))

      
        phi_traj = np.arctan2(car_traj[:, 1], car_traj[:, 0])

        # fill the output polar_traj with r and phi
        polar_traj = np.zeros_like(car_traj)
        polar_traj[:, 0] = r_traj  
        polar_traj[:, 1] = phi_traj  

        if nbr_flag:
            # This is Hist of a Nbr
            # Trajectory Orientation w.r.t the initial position
            car_traj_rel = car_traj - car_traj[0, :]

            traj_orient = np.arctan2(car_traj_rel[:, 1], car_traj_rel[:, 0])
            polar_traj[:, 2] = car_traj[:, 2] * \
               np.cos(traj_orient - phi_traj)  
         
        else:
            This is Hist or Fut of the ego vehicle, then Vr = V
            polar_traj[:, 2] = car_traj[:, 2]  
        
        return polar_traj
        

    def traj_orientation(self, car_traj):
        trj_len = car_traj.shape[0]
        mid_pnt = int(trj_len/2)  
        mid_traj = car_traj[mid_pnt, :]-car_traj[0, :]  
     
        mid_orient = np.arctan2(mid_traj[1], mid_traj[0])
        end_traj = car_traj[-1, :]-car_traj[mid_pnt, :]  
        end_orient = np.arctan2(end_traj[1], end_traj[0])  
        trj_orient = np.zeros_like(car_traj[:, 0])
        trj_orient[0:mid_pnt] = mid_orient
        trj_orient[mid_pnt:trj_len] = end_orient
        return trj_orient
  

    # Collate function for dataloader
    def collate_fn(self, samples):
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) !=
                                  0 for i in range(len(nbrs))])
        maxlen = self.t_h // self.d_s + 1  
        nbrs_batch = torch.zeros(
            maxlen, nbr_batch_size, self.input_dim) 

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(
            len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)
        mask_batch = mask_batch.byte()

      
        hist_batch = torch.zeros(maxlen, len(samples), self.input_dim)
        fut_batch = torch.zeros(self.t_f // self.d_s,
                                len(samples), self.input_dim)  
        op_mask_batch = torch.zeros(
            self.t_f // self.d_s, len(samples), self.input_dim)
        lat_enc_batch = torch.zeros(len(samples), self.n_lat)
        lon_enc_batch = torch.zeros(len(samples), self.n_lon)
        ds_ids_batch = torch.zeros(len(samples), 1)
        vehicle_ids_batch = torch.zeros(len(samples), 1)
        frame_ids_batch = torch.zeros(len(samples), 1)

       
        num_of_agents = 39

        all_adjancent_matrix_mean_batch = torch.zeros(
            len(samples), num_of_agents)
        all_adjancent_matrix_mean_batch = torch.Tensor(
            all_adjancent_matrix_mean_batch)

        '''
        frame_ID_global_adjancent_list_batch = torch.zeros(
            maxlen, len(samples), num_of_agents, num_of_agents)
        frame_ID_global_adjancent_list_batch = torch.Tensor(
            frame_ID_global_adjancent_list_batch)
        '''
        
        '''
        frame_ID_adj_mat_list_batch = torch.zeros(
            maxlen, len(samples), num_of_agents, num_of_agents)
        frame_ID_adj_mat_list_batch = torch.Tensor(frame_ID_adj_mat_list_batch)
        '''
        closeness_list_batch = torch.zeros(
            maxlen, len(samples), num_of_agents)
        closeness_list_batch = torch.Tensor(closeness_list_batch)

        degree_list_batch = torch.zeros(maxlen, len(samples), num_of_agents)
        degree_list_batch = torch.Tensor(degree_list_batch)

        eigenvector_list_batch = torch.zeros(
            maxlen, len(samples), num_of_agents)
        eigenvector_list_batch = torch.Tensor(eigenvector_list_batch)

        # all_closeness_mean,all_degree_mean,all_eigenvector_mean
        all_closeness_mean_batch = torch.zeros(len(samples), num_of_agents)
        all_closeness_mean_batch = torch.Tensor(all_closeness_mean_batch)

        all_degree_mean_batch = torch.zeros(len(samples), num_of_agents)
        all_degree_mean_batch = torch.Tensor(all_degree_mean_batch)

        all_eigenvector_mean_batch = torch.zeros(
            len(samples), num_of_agents)
        all_eigenvector_mean_batch = torch.Tensor(all_eigenvector_mean_batch)

        count = 0
     
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, ds_ids, vehicle_ids, frame_ids,
                    all_adjancent_matrix_mean, 
                       closeness_list, degree_list, eigenvector_list, all_closeness_mean, all_degree_mean, all_eigenvector_mean) in enumerate(samples):
 
            
            
            for k in range(self.input_dim):
                hist_batch[0:len(hist), sampleId, k] = torch.from_numpy(hist[:, k])
                fut_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut[:, k])

            '''
            for i in range(num_of_agents):
                for j in range(num_of_agents):
                    frame_ID_global_adjancent_list_batch[0:maxlen,sampleId,i,j]=frame_ID_global_adjancent_list[0:maxlen,i,j]
            '''
            '''
            for i in range(num_of_agents):
                for j in range(num_of_agents):
                    frame_ID_adj_mat_list_batch[0:maxlen,sampleId,i,j]=frame_ID_adj_mat_list[0:maxlen,i,j]
            '''

            for i in range(num_of_agents): 
                closeness_list_batch[0:maxlen,sampleId,i]=closeness_list[0:maxlen,i]
            
            for i in range(num_of_agents):
                degree_list_batch[0:maxlen,sampleId,i]=degree_list[0:maxlen,i]
            
            for i in range(num_of_agents):
                eigenvector_list_batch[0:maxlen,sampleId,i]=eigenvector_list[0:maxlen,i]


            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            ds_ids_batch[sampleId, :] = torch.tensor(ds_ids.astype(np.float64))
            vehicle_ids_batch[sampleId, :] = torch.tensor(
                vehicle_ids.astype(np.float64))
            frame_ids_batch[sampleId, :] = torch.tensor(
                frame_ids.astype(int).astype(np.float64))

            
            for i in range(num_of_agents):
                all_adjancent_matrix_mean_batch[sampleId,i] = all_adjancent_matrix_mean[i]
            
            for i in range(num_of_agents):
                all_closeness_mean_batch[sampleId,i] = all_closeness_mean[i]
            
            for i in range(num_of_agents):
                all_eigenvector_mean_batch[sampleId,i] = all_eigenvector_mean[i]
            
            for i in range(num_of_agents):
                all_degree_mean_batch[sampleId,i] = all_degree_mean[i]


            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    for k in range(self.input_dim):
                        nbrs_batch[0:len(nbr), count, k] = torch.from_numpy(
                            nbr[:, k])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0],
                               :] = torch.ones(self.enc_size).byte()
                    count += 1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, \
            op_mask_batch, ds_ids_batch, vehicle_ids_batch, frame_ids_batch,\
                all_adjancent_matrix_mean_batch,\
                closeness_list_batch, degree_list_batch, eigenvector_list_batch, all_closeness_mean_batch, all_degree_mean_batch, all_eigenvector_mean_batch


# ________________________________________________________________________________________________________________________________________

# Custom activation for output layer (Graves, 2015)


def outputActivation(x):
    if x.shape[2] == 5:
        muX = x[:,:,0:1]
        muY = x[:,:,1:2]
        sigX = x[:,:,2:3]
        sigY = x[:,:,3:4]
        rho = x[:,:,4:5]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        rho = torch.tanh(rho)
        out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)

    elif x.shape[2] == 7:
        muX = x[:, :, 0:1]
        muY = x[:, :, 1:2]
        muTh = x[:, :, 2:3]
        sigX = x[:, :, 3:4]
        sigY = x[:, :, 4:5]
        sigTh = x[:, :, 5:6]
        rho = x[:, :, 6:7]
        sigX = torch.exp(sigX)
        sigY = torch.exp(sigY)
        sigTh = torch.exp(sigTh)
        rho = 0.6*torch.tanh(rho) # sclaing to avoid NaN when computing the loss
        out = torch.cat([muX, muY, muTh, sigX, sigY, sigTh, rho], dim=2)

    return out


# Compute the NLL using the formula of Multivariate Gaussian distribution
#In matrix form
def compute_nll_mat_red(y_pred, y_gt):
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    muTh = y_pred[:, :, 2]
    sigX = y_pred[:, :, 3]
    sigY = y_pred[:, :, 4]
    sigTh = y_pred[:, :, 5]
    rho = y_pred[:, :, 6]

    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    th = y_gt[:, :, 2]

    # XU = ([x - muX, y - muY, th - muTh])
    # XU = torch.cat((x - muX, y - muY, th - muTh),0)
    XU = torch.zeros(x.shape[0], x.shape[1], 3, 1)
    XU[:, :, 0, 0] = x - muX
    XU[:, :, 1, 0] = y - muY
    XU[:, :, 2, 0] = th - muTh

    #sigma
    sigma_mat = torch.zeros(x.shape[0], x.shape[1], 3, 3)
    sigma_mat[:, :, 0, 0] = torch.pow(sigX, 2)
    sigma_mat[:, :, 1, 0] = rho * sigX * sigY
    sigma_mat[:, :, 2, 0] = rho * sigX * sigTh

    sigma_mat[:, :, 0, 1] = rho * sigX * sigY
    sigma_mat[:, :, 1, 1] = torch.pow(sigY, 2)
    sigma_mat[:, :, 2, 1] = rho * sigY * sigTh

    sigma_mat[:, :, 0, 2] = rho * sigX * sigTh
    sigma_mat[:, :, 1, 2] = rho * sigY * sigTh
    sigma_mat[:, :, 2, 2] = torch.pow(sigTh, 2)

    loss_1 = 0.5 * torch.matmul(torch.matmul(XU.transpose(2, 3), sigma_mat.inverse()), XU)
    loss_1 = loss_1.view(x.shape[0], x.shape[1])



    nll_loss = loss_1 + 2.7568 + 0.5*torch.log(sigma_mat.det())

    return nll_loss


## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    input_dim = y_pred.shape[2]
    if input_dim == 5:
        acc = torch.zeros_like(mask)
        muX = y_pred[:,:,0]
        muY = y_pred[:,:,1]
        sigX = y_pred[:,:,2]
        sigY = y_pred[:,:,3]
        rho = y_pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5)
        x = y_gt[:,:, 0]
        y = y_gt[:,:, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:,:,0] = out
        acc[:,:,1] = out
        acc = acc*mask
        lossVal = torch.sum(acc)/torch.sum(mask)

    elif input_dim == 7:
        # FInd the NLL
        nll = compute_nll_mat_red(y_pred, y_gt)

        # nll_loss tensor filled with the loss value
        nll_loss = torch.zeros_like(mask)
        nll_loss[:, :, 0] = nll
        nll_loss[:, :, 1] = nll
        nll_loss[:, :, 2] = nll

        # mask the loss and find the mean value
        nll_loss = nll_loss * mask
        lossVal = torch.sum(nll_loss) / torch.sum(mask)

    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 3,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        # acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes)

        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = (lat_pred[:,l]*lon_pred[:,k]).cuda()
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut

                output_dim = y_pred.shape[2]

                if output_dim==5:
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                elif output_dim == 7:
                    out = compute_nll_mat_red(y_pred, y_gt)

                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] = (out.cuda() + torch.log(wts).cuda()).cuda() #.cpu()
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = (acc * op_mask[:,:,0]).cuda()
        if avg_along_time:
            lossVal = (torch.sum(acc) / torch.sum(op_mask[:, :, 0].cuda())).cuda()
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred.to('cuda:0')
        y_gt = fut.to('cuda:0')
        output_dim = y_pred.shape[2]
        if output_dim == 5:
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
            # If we represent likelihood in m^(-1):
            # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        elif output_dim == 7:
            out = compute_nll_mat_red(y_pred, y_gt)

        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    ip_dim = y_gt.shape[2]
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)

    if ip_dim==3:
        muVel = y_pred[:,:,2]
        Vel = y_gt[:,:, 2]
        out = out + torch.pow(Vel-muVel, 2)

    for k in range(ip_dim):
        acc[:, :, k] = out

    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all//n_horiz
    for i in range(n_horiz):
        if i==0:
            st_id = 0
        else:
            st_id = n_frames*i

        if i == n_horiz-1:
            en_id = n_all-1
        else:
            en_id = n_frames*i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id+1])

    return avg_res
