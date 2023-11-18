from tqdm import tqdm
import torch
import torch.nn as nn

from models import PredRNNv2_Model
from .predrnn import PredRNN
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

from utils import *

class PredRNNv2(PredRNN):
    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNv2_Model(num_layers, num_hidden, args).to(self.device)
    
    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, eta, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        
        train_pbar = tqdm(train_loader)
        for batch_x, batch_y, mask in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y, mask = batch_x.to(self.device), batch_y.to(self.device), mask.to(self.device)
            batch_x = (batch_x * (1 - mask).float())
            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(eta, num_updates, ims.shape[0], self.args)

            img_gen, loss = self.model(ims, real_input_flag)
            loss.backward()
            self.model_optim.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))
            self.scheduler.step()            
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        return num_updates, loss_mean, eta