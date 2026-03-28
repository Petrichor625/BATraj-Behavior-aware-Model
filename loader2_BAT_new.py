from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import scipy.io as scp
import numpy as np
import torch
import h5py
from config_new import args
import time
from scipy import spatial 
import math
import networkx as nx
import concurrent.futures

class NgsimDataset(Dataset):
	def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
		self.D = scp.loadmat(mat_file)['traj']
		self.T = scp.loadmat(mat_file)['tracks']
		self.t_h = t_h  # length of track history
		self.t_f = t_f  # length of predicted trajectory
		self.d_s = d_s  # skip
		self.enc_size = enc_size  # size of encoder LSTM 
		self.grid_size = grid_size  # size of social context grid 
		self.alltime = 0 
		self.count = 0
		self.polar = True

	def __len__(self):
		return len(self.D)

	def __getitem__(self, idx):
		dsId = self.D[idx, 0].astype(int)  # dataset id 
		vehId = self.D[idx, 1].astype(int) # agent id
		t = self.D[idx, 2]  # frame
		grid = self.D[idx, 11:]  #  grid id
		neighbors = [] #List of nearby vehicles, containing the coordinates of these nearby vehicles
		neighborsva = []##Speed information
		neighborslane = []# Road information
		neighborsclass = []#Vehicle Types
		neighborsdistance = []#Distance
		neigborsrelative_self=[]
		neigborsrelative_nbr=[]
		traj_hist_toal = []

		# Get track history 'hist' = ndarray, and future track 'fut' = ndarray
		hist = self.getHistory(vehId, t, vehId, dsId)#Get the position of the center vehicle at time t
		hist_relative = self.getHistory_relative_self(vehId, t, vehId, dsId)
		refdistance = np.zeros_like(hist[:, 0])
		refdistance = refdistance.reshape(len(refdistance), 1)
		fut = self.getFuture(vehId, t, dsId)
		va = self.getVA(vehId, t, vehId, dsId)
		lane = self.getLane(vehId, t, vehId, dsId)
		cclass = self.getClass(vehId, t, vehId, dsId)



		# Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
		for i in grid:
			nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)#Positions of surrounding vehicles at time t
			neighborsxxx = self.getHistory_relative_self(i.astype(int), t, vehId, dsId)
			if nbrsdis.shape != (0,2):
				nbrsdis_total = nbrsdis
			else:
				nbrsdis_total = np.zeros((16,2))

			if nbrsdis.shape != (0, 2):
				uu = np.power(hist - nbrsdis, 2)
				distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
				distancexxx = distancexxx.reshape(len(distancexxx), 1)
			else:
				distancexxx = np.empty([0, 1])

			if nbrsdis.shape != (0, 2):
				getHistory_relative_nbr =  self.getHistory_relative_nbr(i.astype(int), t, vehId, dsId)
			else:
				getHistory_relative_nbr = np.empty([0, 2])

			neighbors.append(nbrsdis)
			neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
			neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
			neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
			neighborsdistance.append(distancexxx)
			neigborsrelative_self.append(neighborsxxx)
			neigborsrelative_nbr.append(getHistory_relative_nbr)
			traj_hist_toal.append(nbrsdis_total)

		traj_hist_toal = np.array(traj_hist_toal)
		ego_ind = 20
		neighbor_distance=1000
		traj_hist_toal[ego_ind,:,:] = hist[:,:]
		
		feature_matrix = self.create_adjancent_matrix(traj_hist_toal,neighbor_distance) #(time step，39，39) =（16，39，39）
		degree, closeness, eigenvector= self.create_centrality(feature_matrix)#,pagerank,katz,betweenness 
		BLE_BIE = self.BLE_BIE(feature_matrix, degree, closeness, eigenvector)#shape：(Number of vehicles, time steps, 8 features)= (39,16,8)

		lon_enc = np.zeros([args['lon_length']])
		lon_enc[int(self.D[idx, 10] - 1)] = 1
		lat_enc = np.zeros([args['lat_length']])
		lat_enc[int(self.D[idx, 9] - 1)] = 1


		return hist, fut, hist_relative, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass,neigborsrelative_self,neigborsrelative_nbr,feature_matrix,BLE_BIE

	def getLane(self, vehId, t, refVehId, dsId):
		if vehId == 0:
			return np.empty([0, 1])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 1])
			refTrack = self.T[dsId - 1][refVehId - 1].transpose()
			vehTrack = self.T[dsId - 1][vehId - 1].transpose()
			refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5] 

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
				return np.empty([0, 1])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 5]

			if len(hist) < self.t_h // self.d_s + 1:
				return np.empty([0, 1])
		
			return hist 

	def getClass(self, vehId, t, refVehId, dsId):
		if vehId == 0:
			return np.empty([0, 1])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 1])
			refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
			vehTrack = self.T[dsId - 1][vehId - 1].transpose() 
		

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
				return np.empty([0, 1])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 6]

			if len(hist) < self.t_h // self.d_s + 1:
				return np.empty([0, 1])
		
			return hist 

	def getVA(self, vehId, t, refVehId, dsId):
		if vehId == 0:
			return np.empty([0, 2])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 2])
			refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
			vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
			refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5] 

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
				return np.empty([0, 2])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 3:5] 

			if len(hist) < self.t_h // self.d_s + 1:
				return np.empty([0, 2])
			return hist 

	## Helper function to get track history
	def getHistory(self, vehId, t, refVehId, dsId): 
		if vehId == 0:
			return np.empty([0, 2])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 2])
			refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
			vehTrack = self.T[dsId - 1][vehId - 1].transpose() 
			x = np.where(refTrack[:, 0] == t)
			refPos = refTrack[x][0, 1:3]

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
				return np.empty([0, 2])

			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) 
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
				polar = self.polar
				if polar:
					hist = self.cart2polar(hist)  

			if len(hist) < self.t_h // self.d_s + 1:
				return np.empty([0, 2])
		
			return hist 

	def getHistory_relative_self(self, vehId, t, refVehId, dsId): 
		if vehId == 0:
			return np.empty([0, 2])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 2])
			refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
			vehTrack = self.T[dsId - 1][vehId - 1].transpose() 
			x = np.where(refTrack[:, 0] == t)
			refPos = refTrack[x][0, 1:3] 

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
				return np.empty([0, 2])

			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) 
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  
			if len(hist) < self.t_h // self.d_s + 1:  
				return np.empty([0, 2])
	
			diff = np.zeros_like(hist) #print('hist',np.shape(hist)) (16,2)
			diff[1:,:] = hist[1:,:]-hist[:-1,:]

			return diff 
	
	def getHistory_relative_nbr(self, vehId, t, refVehId, dsId):
		if vehId == 0:
			return np.empty([0, 2])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 2]) 
			refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
			vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
			refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0: 
				return np.empty([0, 2])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 1:3] 
	
				hist_ref = np.zeros_like(hist)
				hist_reTrack = np.zeros_like(hist)
				for i in range(len(hist_reTrack)):
					if np.shape(refTrack[stpt:enpt:self.d_s, 1:3]) == (0, 2):
						a = [0,0]
					else:
						a = refTrack[i, 1:3]
					hist_reTrack[i,:2] = a

				hist_ref[:,:] = hist_reTrack[:,:] 
				hist_relative = hist - hist_ref 
	
			if len(hist_relative) < self.t_h // self.d_s + 1:
				return np.empty([0, 2])
			return hist_relative 

	def getdistance(self, vehId, t, refVehId, dsId):
		if vehId == 0:
			return np.empty([0, 1])
		else:
			if self.T.shape[1] <= vehId - 1:
				return np.empty([0, 1]) 
			refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
			vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
			refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

			if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0: 
				return np.empty([0, 1])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos 
				hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos 
				uu = np.power(hist - hist_ref, 2)
				distance = np.sqrt(uu[:, 0] + uu[:, 1])
				distance = distance.reshape(len(distance), 1)

			if len(hist) < self.t_h // self.d_s + 1:
				return np.empty([0, 1])
			return distance 

	def create_adjancent_matrix(self, past_trajs, neighbor_distance):
		xy = past_trajs[:, :, 0:2].astype(float)
		xy = np.around(xy, decimals=2)  
		num_vehicles,num_timesteps, _ = xy.shape


		distance_matrix = np.zeros((num_timesteps, num_vehicles, num_vehicles))
		xy = xy.reshape(xy.shape[1],xy.shape[0],-1)
		for i in range(len(xy)):
			distance_matrix[i] = spatial.distance.cdist(xy[i], xy[i])
	

		dist_xy = np.where(distance_matrix < neighbor_distance, -np.exp(-distance_matrix), 0)
	
		return dist_xy#(16,39,39)

	def create_centrality(self, dist_xy):
		degree_1 = []
		closeness_1 = []
		eigenvector_1 = []


		with concurrent.futures.ThreadPoolExecutor() as executor:
			futures = []
			for i in range(dist_xy.shape[0]):
				future = executor.submit(self.calculate_centrality, dist_xy[i])
				futures.append(future)

			for future in concurrent.futures.as_completed(futures):
				degree_centrality, closeness_centrality, eigenvector_centrality = future.result() # betweenness_centrality
				degree_1.append(degree_centrality)
				closeness_1.append(closeness_centrality)
				eigenvector_1.append(eigenvector_centrality)

		return np.array(degree_1), np.array(closeness_1), np.array(eigenvector_1) #, np.array(betweenness_1)


	def calculate_centrality(self, dist_matrix):
		G = nx.from_numpy_array(dist_matrix.round(2))

		degree = nx.degree_centrality(G)
		degree_centrality = np.array([val for (node, val) in degree.items()])

		closeness = nx.closeness_centrality(G)
		closeness_centrality = np.array([val for (node, val) in closeness.items()])

		eigenvector = nx.eigenvector_centrality_numpy(G)
		eigenvector_centrality = np.array([val for (node, val) in eigenvector.items()])


		return degree_centrality, closeness_centrality, eigenvector_centrality


	def compute_mean_diff(self,array):
		return np.mean(array, axis=0)

	def round_to_decimal(self,array):
		return np.round(array, decimals=2)

	def BLE_BIE(self, dist_xy, degree_1, closeness_1, eigenvector_1):#, betweenness_1


		BLE_degree = self.round_to_decimal(closeness_1.reshape(degree_1.shape[0], -1, 1))
		BLE_closeness = self.round_to_decimal(closeness_1.reshape(closeness_1.shape[0], -1, 1))
		BLE_eigenvector = self.round_to_decimal(closeness_1.reshape(eigenvector_1.shape[0], -1, 1))

		BLE_total = np.concatenate([BLE_degree, BLE_closeness, BLE_eigenvector], axis=2)#（39，16，3）#, BLE_betweenness
		
		closeness_diff = closeness_1[:, 1:] - closeness_1[:, :-1]#(39,15)
		zeros_column = np.zeros((closeness_diff.shape[0], 1))
		closeness_diff_shape = np.append(closeness_diff, zeros_column, axis=1)#(39,16)
		BIE_closeness = self.round_to_decimal(closeness_diff_shape.T.reshape(closeness_diff_shape.shape[1], -1, 1))
		
		degree_diff = degree_1[:, 1:] - degree_1[:, 1:]
		zeros_column = np.zeros((degree_diff.shape[0], 1))
		degree_diff_shape = np.append(degree_diff, zeros_column, axis=1)#(39,16)
		BIE_degree = self.round_to_decimal(degree_diff_shape.T.reshape(degree_diff_shape.shape[1], -1, 1))

		eigenvector_diff = eigenvector_1[:, 1:] - eigenvector_1[:, 1:]
		zeros_column = np.zeros((eigenvector_diff.shape[0], 1))
		eigenvector_diff_shape = np.append(eigenvector_diff, zeros_column, axis=1)#(39,16)
		BIE_eigenvector = self.round_to_decimal(eigenvector_diff_shape.T.reshape(eigenvector_diff_shape.shape[1], -1, 1))

		BIE_total = np.concatenate([BIE_degree, BIE_closeness, BIE_eigenvector], axis=2)#(16, 39, 3)#

		BIE_total = BIE_total.reshape(BIE_total.shape[1],BIE_total.shape[0],-1)
		BIE_BLE_total = np.concatenate([BIE_total, BLE_total], axis=2)# (39, 16, 6)

		return BIE_BLE_total# (39, 16, 6)

	## Helper function to get track future
	def getFuture(self, vehId, t, dsId):
		vehTrack = self.T[dsId - 1][vehId - 1].transpose()
		refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
		stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
		enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
		fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
		polar = self.polar
		if polar:
			fut = self.cart2polar(fut)
		return fut


	def cart2polar(self, car_traj):  
		r_traj = np.sqrt(np.square(car_traj[:, 0]) + np.square(car_traj[:, 1]))
		phi_traj = np.arctan2(car_traj[:, 1], car_traj[:, 0])

		polar_traj = np.zeros_like(car_traj)
		polar_traj[:, 0] = r_traj 
		polar_traj[:, 1] = phi_traj 

		return polar_traj
	


	## Collate function for dataloader
	def collate_fn(self, samples):
		ttt = time.time()
		nbr_batch_size = 0
		for _, _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _, _,_,_ in samples:
			temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
			nbr_batch_size += temp
		max_vehicles = 39
		index_num = 6
		maxlen = self.t_h // self.d_s + 1
		nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2) #[16, XXX, 2]
		nbrs_relative_self_batch = torch.zeros(maxlen, nbr_batch_size, 2)
		nbrs_relative_nbr_batch = torch.zeros(maxlen, nbr_batch_size, 2)
		nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
		nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
		nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
		nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

		
		pos = [0, 0]
		mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
		map_position = torch.zeros(0, 2)
		mask_batch = mask_batch.bool()

		# Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
		hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
		hist_relative_batch = torch.zeros(maxlen, len(samples), 2)
		distance_batch = torch.zeros(maxlen, len(samples), 1)
		fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
		op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
		lat_enc_batch = torch.zeros(len(samples), args['lat_length'])  # (batch,3)
		lon_enc_batch = torch.zeros(len(samples), args['lon_length'])  # (batch,2)
		va_batch = torch.zeros(maxlen, len(samples), 2)
		lane_batch = torch.zeros(maxlen, len(samples), 1)
		class_batch = torch.zeros(maxlen, len(samples), 1)

		#behavior-aware index
		feature_matrix_batch = torch.zeros(maxlen, len(samples),max_vehicles, max_vehicles) #(time step，batch_size, Maximum number of vehicles ，Maximum number of vehicles)
		BLE_BIE_batch =  torch.zeros(maxlen, len(samples),max_vehicles, index_num)#(时刻，batch_size,Maximum number of vehicles，Number of behavioral indicators)
		#print('len(samples)',len(samples))
		count = 0
		count1 = 0
		count2 = 0
		count3 = 0
		count4 = 0
		count_self=0
		count_nbr=0

		for sampleId, (hist, fut, hist_relative, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
					   neighborsdistance, cclass, neighborsclass, neigborsrelative_self, neigborsrelative_nbr,feature_matrix, BLE_BIE) in enumerate(samples):


			hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
			hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
			BLE_BIE = BLE_BIE.reshape(maxlen,max_vehicles,index_num)
			feature_matrix_batch[0:feature_matrix.shape[0],sampleId,:max_vehicles,:max_vehicles] = torch.from_numpy(feature_matrix[0:feature_matrix.shape[0],:,:])
			BLE_BIE_batch[0:BLE_BIE.shape[0],sampleId,:max_vehicles,:index_num] = torch.from_numpy(BLE_BIE[0:BLE_BIE.shape[0], :, :])

			hist_relative_batch[0:len(hist_relative), sampleId, 0] = torch.from_numpy(hist_relative[:, 0])
			hist_relative_batch[0:len(hist_relative), sampleId, 1] = torch.from_numpy(hist_relative[:, 1])

			distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
			fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
			fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
			op_mask_batch[0:len(fut), sampleId, :] = 1
			lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
			lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
			va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
			va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
			lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
			class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)


			# Set up neighbor, neighbor sequence length, and mask batches:
			for id, nbr in enumerate(nbrs):
				if len(nbr) != 0:
					nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
					nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])


					pos[0] = id % self.grid_size[0]
					pos[1] = id // self.grid_size[0]
					mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
					map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
					count += 1

			for id, nbr_self in enumerate(neigborsrelative_self):
				if len(nbr_self) != 0:
					nbrs_relative_self_batch[0:len(neigborsrelative_self), count_self, 0] = torch.from_numpy(nbr_self[:, 0])
					nbrs_relative_self_batch[0:len(neigborsrelative_self), count_self, 1] = torch.from_numpy(nbr_self[:, 1])
					count_self += 1

			for id, nbr_nbr in enumerate(neigborsrelative_nbr):
				if len(nbr_nbr) != 0:
					nbrs_relative_nbr_batch[0:len(neigborsrelative_nbr), count_nbr, 0] = torch.from_numpy(nbr_nbr[:, 0])
					nbrs_relative_nbr_batch[0:len(neigborsrelative_nbr), count_nbr, 1] = torch.from_numpy(nbr_nbr[:, 1])
					count_nbr += 1 

			for id, nbrva in enumerate(neighborsva):
				if len(nbrva) != 0:
					nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
					nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
					count1 += 1

			for id, nbrlane in enumerate(neighborslane):
				if len(nbrlane) != 0:
					nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
					count2 += 1

			for id, nbrdis in enumerate(neighborsdistance):
				if len(nbrdis) != 0:
					nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
					count3 += 1

			for id, nbrclass in enumerate(neighborsclass):
				if len(nbrclass) != 0:
					nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
					count4 += 1

		#  mask_batch 
		self.alltime += (time.time() - ttt)
		self.count += args['num_worker']
		return hist_batch, nbrs_batch, hist_relative_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position, nbrs_relative_self_batch, nbrs_relative_nbr_batch, feature_matrix_batch,BLE_BIE_batch #


