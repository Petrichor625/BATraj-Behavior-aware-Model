from __future__ import print_function

import load.loader2_BAT_new as lo
from torch.utils.data import DataLoader
import pandas as pd
from config_new import *
import matplotlib.pyplot as plt
import os
import time

import torch

import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    ## Helper function for log sum exp calculation:
    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def main(self, name, val):
        model_step = 1
        # args['train_flag'] = not args['use_maneuvers']
        args['train_flag'] = True
        l_path = args['path']
        generator = model.Generator(args=args)
        gdEncoder = model.GDEncoder(args=args)
        device = torch.device("cuda:0" or 'cuda:1' if torch.cuda.is_available() else "cpu")
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location=device))
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location=device))
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()
        if val:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('/NGSIM/ValSet.mat')
                else:
                    t2 = lo.NgsimDataset('/NGSIM/ValSet.mat')
            else:
                t2 = lo.HighdDataset('Val')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn,drop_last=True,pin_memory=True,persistent_workers = True,prefetch_factor =8)  # 6716batch
        else:
  
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('/NGSIM/TestSet.mat')
                else:
                    t2 = lo.NgsimDataset('/NGSIM/TestSet.mat')
            else:
                t2 = lo.HighdDataset('Test')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn,drop_last=True,pin_memory=True,persistent_workers = True,prefetch_factor =8)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0

        val_batch_count = len(valDataloader)
        print("begin.................................", name)
        with(t.no_grad()):
            for idx, data in enumerate(valDataloader):
                hist, nbrs,hist_relative, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions,nbrs_ref_self, nbrs_ref_nbrs,feature_matrix,BLE_BIE = data
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
                feature_matrix = feature_matrix.to(device)
                BLE_BIE = BLE_BIE.to(device)
                map_positions = map_positions.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                nbrs_ref_self = nbrs_ref_self.to(device)
                nbrs_ref_nbrs = nbrs_ref_nbrs.to(device)
                map_positions = map_positions.to(device)
                te = time.time()

                values = gdEncoder(hist, nbrs, hist_relative, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, nbrs_ref_self, nbrs_ref_nbrs,feature_matrix,BLE_BIE)
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
                all_time += time.time() - te
                #nbrsss += 1
                #if nbrsss > args['time']*args['num_worker']:
                #    print(all_time / nbrsss,"ref time")
                if not args['train_flag']:
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):  # 128
                            lat_man = t.argmax(lat_enc[k, :]).detach()
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 3 + lat_man
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if self.drawImg:
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist, fut, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask,
                                  indices)
                else:
                    if args['val_use_mse']:
                        l, c, loss = self.maskedMSETest(fut_pred, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if self.drawImg:
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist, fut, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask, None)

                lossVals += l.detach()
                counts += c.detach()
                avg_val_loss += loss.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            # tqdm.write('valmse:', avg_val_loss / val_batch_count)
            if args['val_use_mse']:
                print('valmse:', avg_val_loss / val_batch_count)
                print(t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
            else:
                print('valnll:', avg_val_loss / val_batch_count)
                print(lossVals / counts)
            # print(lossVals/counts*0.3048)

    def add_car(self, plt, x, y, alp):
        plt.gca().add_patch(plt.Rectangle(
            (x - 5, y - 2.5),  
            10,  
            5, 
            color='maroon',
            alpha=alp
        ))

    def draw(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices):

        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        for i in range(hist.size(1)):
            lon_man_i = lon_man[i].item()
            lat_man_i = lat_man[i].item()
            plt.axis('on')
            plt.ylim(-18 * self.scale, 18 * self.scale)
            plt.xlim(-180 * self.scale * self.prop, 180 * self.scale * self.prop)
            plt.figure(dpi=300)
   
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / 64).item())
            for ii in range(IPL_i):
                plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, ':',
                         color='blue',
                         linewidth=0.5)
          
            IPL = IPL + IPL_i
            plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, ':', color='red',
                     linewidth=0.5)
          
            plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-', color='black',
                     linewidth=0.5)
            if train_flag:
                fut_pred = fut_pred.detach().cpu()
  
                plt.plot(fut_pred[:, i, 1] * self.scale * self.prop, fut_pred[:, i, 0] * self.scale, color='green',
                         linewidth=0.2)
                muX = fut_pred[:, i, 0]
                muY = fut_pred[:, i, 1]
                x = fut[:, i, 0]
                y = fut[:, i, 1]
                max_y = y[-1] - y[0]
                out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                acc = out * op_mask[:, i, 0]
                loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
            else:
                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    if j == indices[i].item():
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='red', linewidth=0.2)
                        muX = fut_pred_i[:, i, 0]
                        muY = fut_pred_i[:, i, 1]
                        x = fut[:, i, 0]
                        y = fut[:, i, 1]
                        max_y = y[-1] - y[0]
                        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                        acc = out * op_mask[:, i, 0]
                        loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
                    else:
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='green', linewidth=0.2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig('./pic/' + str(lon_man_i + 1) + '_' + str(lat_man_i + 1) + '/' + str(self.op) + '.png')
            self.op += 1
            # fig.clf()
            plt.close()

            from xlrd import open_workbook
            from xlutils.copy import copy
            r_xls = open_workbook("test.xls")  
            row = r_xls.sheets()[lon_man_i * 3 + lat_man_i].nrows  
            excel = copy(r_xls) 
            table = excel.get_sheet(lon_man_i * 3 + lat_man_i)  
            table.write(row, 0, str(self.op - 1))
            table.write(row, 1, max_y.item())  
            table.write(row, 2, loss.item())  
            table.write(row, 3, str(acc))
            excel.save("./test.xls")  


if __name__ == '__main__':

    names = ['8'] #Enter the corresponding number of epochs to run the test
    evaluate = Evaluate() 
    for epoch in names:
        evaluate.main(name=epoch, val=False)
