import os 
import h5py
import sklearn
import numpy as np
import pandas as pd 
from scipy.io import loadmat
from einops import rearrange

import torch 
from torch.utils.data import Dataset, DataLoader

def create_samples(data_root, csv_path, random_state, num_data_point):
    # Load channel data
    #channel_ad_clip = loadmat(os.path.join(data_root, 'perfect_channel_ad_clip.mat'))['all_channel_ad_clip']
    channel_ad_clip = loadmat(data_root+'/est_channel_ad_clip.mat')['all_channel_ad_clip']

    channel = h5py.File(data_root+'/channel.mat')['all_channel']
    channel = np.transpose(channel, axes=range(channel.ndim)[::-1])
    real_channel = channel['real']
    imag_channel = channel['imag']
    channel = real_channel + 1j * imag_channel

    # Load data index
    data_idx = pd.read_csv(os.path.join(data_root, csv_path))["data_idx"].to_numpy()
    channel = channel[data_idx, ...]
    channel_ad_clip = channel_ad_clip[data_idx, ...]

    channel = np.squeeze(channel)
    channel_ad_clip = np.squeeze(channel_ad_clip)

    # Select given number of data
    if num_data_point:
        channel_ad_clip = channel_ad_clip[:num_data_point, ...]
        data_idx = data_idx[:num_data_point, ...]
        channel = channel[:num_data_point, ...]

    # Normalization
    amplitude = np.linalg.norm(channel_ad_clip, ord='fro', axis=(-1,-2), keepdims=True)
    channel_ad_clip /= amplitude

    phase = np.expand_dims(channel_ad_clip[:, 0, 0] / np.abs(channel_ad_clip[:, 0, 0]), (1,2))
    channel_ad_clip = channel_ad_clip / phase

    return channel_ad_clip, data_idx, channel, amplitude, phase

class DataFeed_Test(Dataset):
    def __init__(self, data_root, csv_path, random_state=0, num_data_point=None):
        self.data_root = data_root
        self.channel_ad_clip, self.data_idx, self.channel, self.amplitude, self.phase = create_samples(
            data_root, csv_path, random_state, num_data_point
        )

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx, ...]
        channel_ad_clip = self.channel_ad_clip[idx, ...]
        channel = self.channel[idx, ...]
        amplitude = self.amplitude[idx, ...]
        phase = self.phase[idx, ...]

        data_idx = torch.tensor(data_idx, requires_grad=False)

        channel = torch.tensor(channel, requires_grad=False)

        channel_ad_clip = torch.tensor(channel_ad_clip, requires_grad=False)
        channel_ad_clip = torch.view_as_real(channel_ad_clip)
        channel_ad_clip = rearrange(channel_ad_clip, 'Nt Nc RealImag -> RealImag Nc Nt')

        amplitude = torch.tensor(amplitude, requires_grad=False)
        phase = torch.tensor(phase, requires_grad=False)

        return channel_ad_clip.float(), data_idx.long(), channel.cfloat(), amplitude.float(), phase.cfloat()

if __name__ == "__main__":
    data_root = "../DeepMIMO_dataset/Boston5G_3p5"
    train_csv = "train_data_idx.csv"
    val_csv = "test_data_idx.csv"

    batch_size = 4

    train_loader = DataLoader(DataFeed_Test(data_root, train_csv), batch_size=batch_size)
    channel_ad_clip, data_idx, channel, amplitude, phase = next(iter(train_loader))

    print(channel_ad_clip.size())