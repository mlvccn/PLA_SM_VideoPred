from .optim_scheduler import get_optim_scheduler
from typing import Dict, List, Union
import torch
class Base_method(object):
    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.device = device
        self.config = args.__dict__
        self.criterion = None

    def _build_model(self, **kwargs):
        raise NotImplementedError

    # def _init_optimizer(self, **kwargs):
    #     raise NotImplementedError
    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(self.args, self.args.epoch, self.model, steps_per_epoch)
    
    def train_one_epoch(self, train_loader, **kwargs): 
        '''
        Train the model with train_loader.
        Input params:
            train_loader: dataloader of train.
        '''
        raise NotImplementedError
    
    def vali_one_epoch(self, vali_loader, **kwargs):
        '''
        Evaluate the model with val_loader.
        Input params:
            val_loader: dataloader of validation.
        '''
        raise NotImplementedError

    def test_one_epoch(self, test_loader, **kwargs):
        raise NotImplementedError
    
    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr