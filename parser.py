import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='human_sla', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--seed', default=2, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='/data16t/')
    parser.add_argument('--dataname', default='human',
                        choices=['mmnist','kth20','kitticaltech','taxibj','kth','human','human_pretrain'])

    
    # method parameters
    parser.add_argument('--method', default='SimVP', choices=[
                        'SimVP', 'SimVP_pretrain','ConvLSTM', 'PredRNNpp', 'PredRNN', 'PredRNNv2', 'MIM', 'E3DLSTM', 'MAU', 'CrevNet', 'PhyDNet','FFT','ConvNext'])
    parser.add_argument('--config_file', '-c', default='./configs/ConvNext.py', type=str)
    
    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--sched', default='onecycle', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    return parser