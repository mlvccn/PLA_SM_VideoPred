from os import makedirs
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from models import SimVP_Model_Pretrain
from .base_method import Base_method
#from timm.utils import AverageMeter

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SimVP_pretrain(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        if torch.cuda.device_count() >1:
            self.model = torch.nn.DataParallel(self.model)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model_Pretrain(**config).to(self.device)

    def _predict(self, batch_x,maskratio):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x,maskratio)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x,maskratio)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq,maskratio)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq,maskratio)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
        
    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        train_pbar = tqdm(train_loader)
        for batch_x in train_pbar:
            self.model_optim.zero_grad()
            batch_x = batch_x.to(self.device)
            # if epoch <= 50:
            #     maskratio = 0.6 + epoch * 0.006
            # elif epoch<=80:
            #     maskratio = 2.4 - epoch*0.03
            maskratio = 0.95
            re_x = self._predict(batch_x,maskratio)
            loss = self.criterion(re_x, batch_x)
            loss.backward()
            self.model_optim.step()
            self.scheduler.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))

            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        print('mask_ratio',maskratio)
        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x) in enumerate(vali_pbar):
            batch_x = batch_x.to(self.device)
            re_x = self._predict(batch_x,maskratio=0.95)
            loss = self.criterion(re_x, batch_x)    

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [re_x, batch_x], [preds_lst, trues_lst]))

            if i * batch_x.shape[0] > 2000:
                break
            
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
        
        total_loss = np.average(total_loss)
        
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        return preds, trues, total_loss
    
    def test_one_epoch(self, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)
        for batch_x in test_pbar:
            #batch_x = (batch_x * (1 - mask).float())
            batch_x = batch_x.to(self.device)
            re_x = self._predict(batch_x,maskratio=0.95)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_x, re_x], [inputs_lst, trues_lst, preds_lst]))
            
        inputs, trues, preds = map(lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds