from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_moving_mnist_pretrain import load_data as load_mmnist_pretrain
from .dataloader_kth import load_data as load_kth
from .dataloader_kitti import load_data as load_kitticaltech
from .dataloader_taxibj import load_data as load_taxibj
from .datalodaer_human import load_data as load_human
from .datalodaer_human_pretrain import load_data as load_human_pretrain
from .dataloader_kth_pretrain import load_data as load_kth_pretrain
def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    if dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, num_workers, data_root)
    elif dataname == 'mmnist_pretrain':
        return load_mmnist_pretrain(batch_size, val_batch_size, num_workers, data_root)
    elif dataname == 'kth':
        return load_kth(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kth_pretrain':
        return load_kth_pretrain(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kth20':
        return load_kth(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'kitticaltech':
        return load_kitticaltech(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'human':
        return load_human(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'human_pretrain':
        return load_human_pretrain(batch_size, val_batch_size, data_root, num_workers)