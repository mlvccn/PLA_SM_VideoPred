#import nni
import logging
import pickle
import json
import torch
import numpy as np
import os.path as osp
from parser import create_parser
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import warnings
warnings.filterwarnings('ignore')

from API import metric, Recorder

from constants import method_maps
from utils import *

class Exp:
    def __init__(self, args):
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        
        self._preparation()
        print_log(output_namespace(self.args))
        
        T, C, H, W = self.args.in_shape
        if self.args.method == 'SimVP':
            _tmp_input = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            flops = FlopCountAnalysis(self.method.model, _tmp_input )
        elif self.args.method == 'FFT':
            _tmp_input = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            flops = FlopCountAnalysis(self.method.model, _tmp_input)
        elif self.args.method == 'CrevNet':
            _tmp_input = torch.ones(self.args.batch_size, 20, C, H, W).to(self.device)  # crevnet must use the batchsize rather than 1
            flops = FlopCountAnalysis(self.method.model, _tmp_input)
        elif self.args.method == 'PhyDNet':
            _tmp_input1 = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            _tmp_input2 = torch.ones(1, self.args.aft_seq_length, C, H, W).to(self.device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(self.device)
            flops = FlopCountAnalysis(self.method.model, (_tmp_input1, _tmp_input2, _tmp_constraints))
        elif self.args.method in ['ConvLSTM', 'PredRNNpp', 'PredRNN', 'MIM', 'E3DLSTM', 'MAU']:
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            flops = FlopCountAnalysis(self.method.model, (_tmp_input, _tmp_flag))
        elif self.args.method == 'PredRNNv2':
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.total_length - 2, Hp, Wp, Cp).to(self.device)
            flops = FlopCountAnalysis(self.method.model, (_tmp_input, _tmp_flag))
        print_log(self.method.model)
        print_log(flop_count_table(flops))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            #device = torch.device('cuda:0')
            device = torch.device('cuda:{}'.format(0))
            #print('Use GPU:',device)
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)
        
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        prefix = 'train' if not self.args.test else 'test'
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)
        

    def _get_data(self):
        self.train_loader, self.vali_loader, self.test_loader = get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

    # def _save(self, name=''):
    #     torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
    #     fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
    #     state = self.method.scheduler.state_dict()
    #     pickle.dump(state, fw)
    def _save(self, name=''):
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))    

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        # pretrain_model_path = '/data16t/cjj/Documents/zch_Documents/ConvNextVP/results/human_pretrain/checkpoint.pth'
        # print('load pretrain_model:'+pretrain_model_path)
        # self.method.model.load_state_dict(torch.load(pretrain_model_path),strict=False)
        # for name, p in self.method.model.named_parameters():
        #     if name.startswith('enc'):
        #         p.requires_grad = False
        recorder = Recorder(verbose=True)
        num_updates = 0
        # constants for other methods:
        eta = 1.0  # PredRNN
        for epoch in range(self.config['epoch']):
            loss_mean = 0.0
            
            if self.args.method in ['SimVP', 'CrevNet', 'PhyDNet','FFT','SimVP_finetune']:
                num_updates, loss_mean = self.method.train_one_epoch(self.train_loader, epoch, num_updates, loss_mean)
            elif self.args.method in ['ConvLSTM', 'PredRNNpp', 'PredRNN', 'PredRNNv2', 'MIM', 'E3DLSTM', 'MAU']:
                num_updates, loss_mean, eta = self.method.train_one_epoch(self.train_loader, epoch, num_updates, loss_mean, eta)
            
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                
                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                    epoch + 1, len(self.train_loader), cur_lr, loss_mean, vali_loss))
                recorder(vali_loss, self.method.model, self.path)
                #self._save(name='latest')
            # if epoch % 10 == 0:
            #     torch.save(self.method.model.state_dict(), self.path+'/checkpoints/'+str(epoch)+'.pth')
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def vali(self, vali_loader):
        preds, trues, val_loss = self.method.vali_one_epoch(self.vali_loader)
        metric_list, spatial_norm = ['mse', 'mae'], False
        # mae, mse, ssim, psnr, lpips = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, return_ssim_psnr=True)
        # print_log('vali\t mse:{}, mae:{}, ssim:{}, psnr:{}, lpips:{}'.format(mse, mae, ssim, psnr, lpips))
        eval_res, eval_log = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        print_log('val\t '+eval_log)
        #mae, mse, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, return_ssim_psnr=True)
        #print_log('val\t mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}' .format(mse, mae, ssim, psnr))
        #nni.report_intermediate_result(mse)

        return val_loss

    def test(self):
        #best_model_path = '/data16t/cjj/Documents/zch_Documents/ConvNextVP/results/kth_pretrain_10_mrla/checkpoints/90.pth'
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))
        metric_list, spatial_norm = ['mse', 'mae', 'ssim', 'psnr'], False
        inputs, trues, preds = self.method.test_one_epoch(self.test_loader)
        #mae, mse, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, return_ssim_psnr=True)
        eval_res, eval_log = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        metrics = np.array([eval_res['mae'], eval_res['mse']])
        print_log(eval_log)
        #metrics = np.array([mae, mse])
        #print_log('mse:{}, mae:{}, ssim:{}, psnr:{}'.format(mse, mae, ssim, psnr))
        folder_path = osp.join(self.path, 'saved')
        check_dir(folder_path)

        for np_data in ['metrics', 'inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return eval_res['mse']


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    #tuner_params = nni.get_next_parameter()
    #config.update(tuner_params)
    
    default_params = load_config(osp.join('./configs', args.method + '.py') if args.config_file is None else args.config_file)
    config.update(default_params)
    
    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test()
    #nni.report_final_result(mse)