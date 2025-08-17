import os 
import sklearn
import numpy as np
import pandas as pd 
from scipy.io import loadmat
from einops import rearrange

import torch 
from torch.utils.data import Dataset, DataLoader

def create_samples(data_root, csv_path, random_state, num_data_point, select_data_idx):
    # Load channel data
    channel_ad_clip = loadmat(os.path.join(data_root, 'channel_ad_clip.mat'))['all_channel_ad_clip']

    # Load data index
    if select_data_idx is None:
        data_idx = pd.read_csv(os.path.join(data_root, csv_path))["data_idx"].to_numpy()
    else:
        data_idx = select_data_idx
    channel_ad_clip = channel_ad_clip[data_idx, ...]

    # Shuffle data
    channel_ad_clip, data_idx = sklearn.utils.shuffle(channel_ad_clip, data_idx, random_state=random_state)
    channel_ad_clip = np.squeeze(channel_ad_clip)

    # Select given number of data
    if num_data_point:
        channel_ad_clip = channel_ad_clip[:num_data_point, ...]
        data_idx = data_idx[:num_data_point, ...]

    # Normalization
    channel_ad_clip /= np.linalg.norm(channel_ad_clip, ord='fro', axis=(-1,-2), keepdims=True)
    channel_ad_clip = channel_ad_clip / np.expand_dims(channel_ad_clip[:, 0, 0] / np.abs(channel_ad_clip[:, 0, 0]), (1,2))

    return channel_ad_clip, data_idx

class DataFeed(Dataset):
    def __init__(self, data_root, csv_path, random_state=0, num_data_point=None, select_data_idx=None):
        self.data_root = data_root
        self.channel_ad_clip, self.data_idx = create_samples(
            data_root, csv_path, random_state, num_data_point, select_data_idx
        )

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx, ...]
        channel_ad_clip = self.channel_ad_clip[idx, ...]

        data_idx = torch.tensor(data_idx, requires_grad=False)

        channel_ad_clip = torch.tensor(channel_ad_clip, requires_grad=False)
        channel_ad_clip = torch.view_as_real(channel_ad_clip)
        channel_ad_clip = rearrange(channel_ad_clip, 'Nt Nc RealImag -> RealImag Nc Nt')

        return channel_ad_clip.float(), data_idx.long()

if __name__ == "__main__":
    data_root = "DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_notree_2"
    train_csv = "train_data_idx.csv"
    val_csv = "test_data_idx.csv"

    batch_size = 4

    train_loader = DataLoader(DataFeed(data_root, train_csv), batch_size=batch_size)
    channel_ad_clip, data_idx = next(iter(train_loader))

    print(channel_ad_clip.size())