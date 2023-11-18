import torch
import numpy as np
from torch.utils.data import Dataset
from API.utils import create_random_shape_with_random_motion

class TrafficDataset(Dataset):
    def __init__(self, X, Y,train):
        super(TrafficDataset, self).__init__()
        self.X = (X + 1) / 2
        self.Y = (Y + 1) / 2
        self.mean = 0
        self.std = 1
        #self.mask = np.load('/home/cjj/Documents/SimVP-master/data/taxibj/taxi_mask.npy')
        self.train = train

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        # if self.train:
        #     all_masks = create_random_shape_with_random_motion(
        #     4, imageHeight=32, imageWidth=32)
        #     all_masks = np.stack([np.expand_dims(x, 2) for x in all_masks], axis=0)/ 255.0  
        # else:
        #     all_masks = self.mask[index]
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels#, torch.tensor(all_masks.transpose(0,3,1,2),dtype=torch.float)

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):
    
    dataset = np.load(data_root+'/taxibj/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']

    train_set = TrafficDataset(X=X_train, Y=Y_train,train=1)
    test_set = TrafficDataset(X=X_test, Y=Y_test,train=0)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_test, dataloader_test