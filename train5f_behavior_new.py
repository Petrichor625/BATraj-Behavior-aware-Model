from torch.utils.data import DataLoader
import load.loader2_BAT_new as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from tqdm import tqdm
import os
from evaluate5f_behavior_new import Evaluate
from config_new import *
import time
import torch.multiprocessing as multiprocessing

train_loss = []

def maskedNLL(y_pred, y_gt, mask):
	# mask = t.cat((mask[:15, :, :], t.zeros_like(mask[15:, :, :])), dim=0)
	acc = t.zeros_like(mask)
	muX = y_pred[:, :, 0]
	muY = y_pred[:, :, 1]
	sigX = y_pred[:, :, 2]
	sigY = y_pred[:, :, 3]
	rho = y_pred[:, :, 4]
	ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # (1-rhp^2)^0.5
	x = y_gt[:, :, 0]
	y = y_gt[:, :, 1]
	# If we represent likelihood in feet^(-1)
	out = 0.5 * t.pow(ohr, 2) * (
			t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY, 2) - 2 * rho * t.pow(sigX,
																									  1) * t.pow(
		sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
	# If we represent likelihood in m^(-1):meter out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX,
	# 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX)
	# * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
	acc[:, :, 0] = out
	acc[:, :, 1] = out
	acc = acc * mask
	lossVal = t.sum(acc) / t.sum(mask)
	return lossVal




def MSELoss2(g_out, fut, mask):
	acc = t.zeros_like(mask)
	muX = g_out[:, :, 0]
	muY = g_out[:, :, 1]
	x = fut[:, :, 0]
	y = fut[:, :, 1]
	out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
	acc[:, :, 0] = out
	acc[:, :, 1] = out
	acc = acc * mask
	lossVal = t.sum(acc) / t.sum(mask)
	return lossVal


def CELoss(pred, target):
	value = t.log(t.sum(pred * target, dim=-1))
	return -t.sum(value) / value.shape[0]


def main():
	args['train_flag'] = True
	evaluate = Evaluate()
	multiprocessing.set_start_method("spawn")

	gdEncoder = model.GDEncoder(args)
	generator = model.Generator(args)
	generator.load_state_dict(t.load('../epoch8_g.tar'))
	gdEncoder.load_state_dict(t.load('../epoch8_gd.tar'))
	gdEncoder = gdEncoder.to(device)
	generator = generator.to(device)
	gdEncoder.train()
	generator.train()
	if dataset == "ngsim":
		if args['lon_length'] == 3:
			t1 = lo.NgsimDataset('../NGSIM/TrainSet.mat')
		else:
			t1 = lo.NgsimDataset('../NGSIM/TrainSet.mat')
		
		trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
									 collate_fn=t1.collate_fn,drop_last=True,pin_memory=True,persistent_workers = True,prefetch_factor =8)

	optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=learning_rate)
	optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
	scheduler_gd = CosineAnnealingWarmRestarts(optimizer_gd,T_0=4,T_mult=1,eta_min=0.00001, last_epoch=-1, verbose=True)
	scheduler_g = CosineAnnealingWarmRestarts(optimizer_g,T_0=4,T_mult=1,eta_min=0.00001, last_epoch=-1, verbose=True)

	for epoch in range(args['epoch']):
		print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])
		loss_gi1 = 0
		loss_gix = 0
		loss_gx_2i = 0
		loss_gx_3i = 0
		avg_tr_loss = 0
		avg_tr_time = 0
		avg_lat_acc = 0
		avg_lon_acc = 0
		for idx, data in enumerate(tqdm(trainDataloader)):
			st_time = time.time()

			hist, nbrs, hist_relative, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions, nbrs_ref_self, nbrs_ref_nbrs,feature_matrix,BMI_BTI_BCI = data #,Safe_index_list,Data_list,Graph_list
			
			hist = hist.to(device)
			nbrs = nbrs.to(device)
			hist_relative = hist_relative.to(device)
			mask = mask.to(device)
			lat_enc = lat_enc.to(device)
			lon_enc = lon_enc.to(device)
			fut = fut[:args['out_length'], :, :]
			fut = fut.to(device)
			op_mask = op_mask[:args['out_length'], :, :]
			op_mask = op_mask.to(device)
			va = va.to(device)
			nbrsva = nbrsva.to(device)
			lane = lane.to(device)
			nbrslane = nbrslane.to(device)
			dis = dis.to(device)
			nbrsdis = nbrsdis.to(device)
			map_positions = map_positions.to(device)
			cls = cls.to(device)
			nbrscls = nbrscls.to(device)
			nbrs_ref_self = nbrs_ref_self.to(device)
			nbrs_ref_nbrs = nbrs_ref_nbrs.to(device)
			feature_matrix = feature_matrix.to(device)
			BMI_BTI_BCI = BMI_BTI_BCI.to(device)
		

			values = gdEncoder(hist, nbrs, hist_relative, mask, va, nbrsva, lane, nbrslane, cls, nbrscls,nbrs_ref_self,nbrs_ref_nbrs,feature_matrix,BMI_BTI_BCI)#,Safe_index_list,Data_list,Graph_list
			
			g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
			if args['use_mse']:
				loss_g1 = MSELoss2(g_out, fut, op_mask)
			else:
				if epoch < args['pre_epoch']:
					loss_g1 = MSELoss2(g_out, fut, op_mask)
				else:
					loss_g1 = maskedNLL(g_out, fut, op_mask)
			loss_gx_3 = CELoss(lat_pred, lat_enc)
			loss_gx_2 = CELoss(lon_pred, lon_enc)
			loss_gx = loss_gx_3 + loss_gx_2
			loss_g = loss_g1 + 1 * loss_gx
			optimizer_g.zero_grad()
			optimizer_gd.zero_grad()
			loss_g.backward()
			a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
			a = t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
			optimizer_g.step()
			optimizer_gd.step()

			loss_gi1 += loss_g1.item()
			loss_gx_2i += loss_gx_2.item()
			loss_gx_3i += loss_gx_3.item()
			loss_gix += loss_gx.item()

			
			batch_time = time.time() - st_time
			avg_tr_loss += loss_g.item()
			avg_tr_time += batch_time

			if idx % 100 == 99:
				eta = avg_tr_time / 100 * (len(t1) / args['batch_size']  - idx)
				print("Epoch no:", epoch + 1,
				  "| Epoch progress(%):", format(
					  idx / (len(t1) / args['batch_size'] ) * 100, '0.2f'),
				  "| Avg train loss:", format(avg_tr_loss / 100, '0.4f'))
			#print('----==========scheduler start===========----')
		   #time_start = time.perf_counter()
			
				train_loss.append(avg_tr_loss / 100)
				avg_tr_loss = 0
				avg_lat_acc = 0
				avg_lon_acc = 0
				avg_tr_time = 0


			if idx % 10000 == 9999:
				print('mse:', loss_gi1 / 10000, '|loss_gx_2:', loss_gx_2i / 10000, '|loss_gx_3', loss_gx_3i / 10000)
				loss_gi1 = 0
				loss_gix = 0
				loss_gx_2i = 0
				loss_gx_3i = 0
	

		save_model(name=str(epoch + 1), gdEncoder=gdEncoder,
				   generator=generator, path = args['path'])
	
		scheduler_gd.step()
		scheduler_g.step()


def save_model(name, gdEncoder, generator, path):
	l_path = args['path']
	if not os.path.exists(l_path):
		os.makedirs(l_path)
	t.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.tar')
	t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.tar')


if __name__ == '__main__':
	main()
