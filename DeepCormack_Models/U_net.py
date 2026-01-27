# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from typing import Optional
import torch
import torch.nn as nn
from LION.models import LIONmodel
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np

"""
class PairedACARDataLoader(Dataset):
    def __init__(self, measurement_folder, target_folder, file_pattern, N, transform=None, max_samples=None):
        self.measurement_files = sorted(glob.glob(os.path.join(measurement_folder, file_pattern)))
        self.target_files = sorted(glob.glob(os.path.join(target_folder, file_pattern)))
        assert len(self.measurement_files) == len(self.target_files), "Mismatch in number of files!"
        if max_samples is not None:
            self.measurement_files = self.measurement_files[:max_samples]
            self.target_files = self.target_files[:max_samples]
        self.N = N
        self.transform = transform

    def __len__(self):
        return len(self.measurement_files)

    def __getitem__(self, idx):
        meas = np.loadtxt(self.measurement_files[idx]).reshape((self.N, self.N))
        targ = np.loadtxt(self.target_files[idx]).reshape((self.N, self.N))
        max_val = np.max(meas)
        if max_val != 0:
            meas = meas / max_val
            targ = targ / max_val
        if self.transform:
            meas = self.transform(meas)
            targ = self.transform(targ)
        return meas, targ
"""

class DeepCormack_dataloader(Dataset):
    """Args:
        measurement_folder (str): Path to the folder containing measurement files.
        target_folder (str): Path to the folder containing target files.
        file_pattern (str): Glob pattern for file matching (e.g., '*.txt').
        N (int): Base dimension for reshaping.
        transform (callable, optional): Optional transform to be applied on a sample.
        max_samples (int, optional): Maximum number of samples to load.
    """
    def __init__(self, measurement_folder, target_folder, file_pattern, N, transform=None, max_samples=None):
        self.measurement_files = sorted(glob.glob(os.path.join(measurement_folder, file_pattern)))
        self.target_files = sorted(glob.glob(os.path.join(target_folder, file_pattern)))

        #print("Measurement files:", len(self.measurement_files), self.measurement_files[:5])
        #print("Target files:", len(self.target_files), self.target_files[:5])
        
        if max_samples is None:
            assert len(self.measurement_files) == len(self.target_files), "Mismatch in number of files!"
        else:
            self.measurement_files = self.measurement_files[:max_samples]
            self.target_files = self.target_files[:max_samples]
        self.N = N
        self.transform = transform
    
    def __len__(self):
        return len(self.measurement_files)
    
    def __getitem__(self, idx):
        # Load flattened data
        meas_flat = np.loadtxt(self.measurement_files[idx])
        targ_flat = np.loadtxt(self.target_files[idx])
    
        # Reshape to (2*N, 5, 2*N)
        meas = meas_flat.reshape((self.N, 5, self.N)).transpose(1, 0, 2)
        targ = targ_flat.reshape((self.N, 5, self.N)).transpose(1, 0, 2)
    
        # Normalize by max value in measurement
        max_val = np.max(meas)
        if max_val != 0:
            meas = meas / max_val
            targ = targ / max_val
    
        # Apply optional transform
        if self.transform:
            meas = self.transform(meas)
            targ = self.transform(targ)
    
        return meas, targ



# This is for importing the test_set used in training
class FermiTestSet(Dataset):
    """
    Dataset for Fermi test set compatible with DeepCormack_dataloader structure.

    Args:
        file_list_path (str): Path to the file containing measurement file paths.
        N (int): Base dimension for reshaping.
        folder_name (str): Name of the target folder.
        transform (callable, optional): Optional transform to be applied on a sample.
        root_dir (str, optional): Root directory for file paths.
    """
    def __init__(self, file_list_path, N, folder_name, transform=None, root_dir=''):
        self.N = N
        self.transform = transform
        self.root_dir = root_dir
        self.folder_name = folder_name

        # Read all file paths from the file list
        with open(file_list_path, 'r') as f:
            self.file_paths = [os.path.join(root_dir, line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        meas_path = self.file_paths[idx]

        # Extract the index from the measurement file name
        file_name = os.path.basename(meas_path)
        identifier = file_name.split('_')[-1].replace('.txt', '')

        # Construct the corresponding target path
        target_path = os.path.join(
            self.root_dir,
            f'/store/LION/gfbl2/DeepCormack_Data/{self.folder_name}/TPMD_Measurement',
            f'tpmd_meas_{identifier}.txt'
        )

        # Load measurement and target data (flattened)
        meas_flat = np.loadtxt(meas_path)
        target_flat = np.loadtxt(target_path)

        # Reshape to (N, 5, N) or (2*N, 5, 2*N) as needed
        meas_data = meas_flat.reshape((self.N, 5, self.N))
        target_data = target_flat.reshape((self.N, 5, self.N))

        # meas_data = meas_flat.reshape((2*self.N, 5, 2*self.N))
        # target_data = target_flat.reshape((2*self.N, 5, 2*self.N))

        # Normalize by max value in measurement
        max_val = np.max(meas_data)
        if max_val != 0:
            meas_data = meas_data / max_val
            target_data = target_data / max_val

        # Apply optional transform
        if self.transform:
            meas_data = self.transform(meas_data)
            target_data = self.transform(target_data)

        return meas_data, target_data
    

        
# Implementation of:

# Jin, Kyong Hwan, et al.
# "Deep convolutional neural network for inverse problems in imaging."
# IEEE Transactions on Image Processing 26.9 (2017): 4509-4522.
# DOI: 10.1109/TIP.2017.2713099


class ConvBlock(nn.Module):
    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3, dropout=0.0):
        super().__init__()
        # input parsing:

        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            layer_list.append(
                nn.Conv2d(
                    channels[ii], channels[ii + 1], kernel_size, padding=1, bias=False
                )
            )
            layer_list.append(nn.BatchNorm2d(channels[ii + 1]))
            if ii < layers - 1 or relu_last:
                if relu_type == "ReLU":
                    layer_list.append(torch.nn.ReLU())
                elif relu_type == "LeakyReLU":
                    layer_list.append(torch.nn.LeakyReLU())
                elif relu_type != "None":
                    raise ValueError("Wrong ReLu type " + relu_type)
                if dropout > 0.0: 
                    layer_list.append(nn.Dropout2d(p=dropout))       # Adding dropout layer
            self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Upscaling with transpose conv"""

    def __init__(self, channels, stride=2, relu_type="ReLU"):
        super().__init__()
        kernel_size = 3
        layer_list = []
        layer_list.append(
            nn.ConvTranspose2d(
                channels[0],
                channels[1],
                kernel_size,
                padding=1,
                output_padding=1,
                stride=stride,
                bias=False,
            )
        )
        layer_list.append(nn.BatchNorm2d(channels[1]))
        if relu_type == "ReLU":
            layer_list.append(nn.ReLU())
        elif relu_type == "LeakyReLU":
            layer_list.append(nn.LeakyReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)

# Change FBP to Cormack
class FBPConvNetParams(LIONmodel.ModelParams):
    def __init__(
        self,
        down_1_channels: list[int],
        down_2_channels: list[int],
        down_3_channels: list[int],
        #down_4_channels: list[int],
        latent_channels: list[int],
        up_1_channels: list[int],
        up_2_channels: list[int],
        up_3_channels: list[int],
        #up_4_channels: list[int],
        last_block: list[int],
        activation: str,
    ):
        super().__init__(LIONmodel.ModelInputType.IMAGE)
        self.down_1_channels = down_1_channels
        self.down_2_channels = down_2_channels
        self.down_3_channels = down_3_channels
        #self.down_4_channels = down_4_channels

        self.latent_channels = latent_channels

        self.up_1_channels = up_1_channels
        self.up_2_channels = up_2_channels
        self.up_3_channels = up_3_channels
        #self.up_4_channels = up_4_channels

        self.last_block = last_block

        self.activation = activation


import torch
import torch.nn as nn

# Change FBP to Cormack
class FBPConvNet(nn.Module):
    def __init__(self, model_parameters):
        super().__init__()
        self.model_parameters = model_parameters
        # Down blocks
        self.block_1_down = ConvBlock(
            self.model_parameters.down_1_channels,
            relu_type=self.model_parameters.activation
        )
        self.down_1 = Down()
        self.block_2_down = ConvBlock(
            self.model_parameters.down_2_channels,
            relu_type=self.model_parameters.activation,
            #dropout=0.1
        )
        self.down_2 = Down()
        self.block_3_down = ConvBlock(
            self.model_parameters.down_3_channels,
            relu_type=self.model_parameters.activation,
            #dropout=0.1
        )
        self.down_3 = Down()
        
        """self.block_4_down = ConvBlock(
            self.model_parameters.down_4_channels,
            relu_type=self.model_parameters.activation,
            dropout=0.2
        )
        self.down_4 = Down()"""
        
        # Latent space
        self.block_bottom = ConvBlock(
            self.model_parameters.latent_channels,
            relu_type=self.model_parameters.activation,
            dropout=0.4
        )
        # Up blocks
        self.up_1 = Up([
            self.model_parameters.latent_channels[-1],
            self.model_parameters.up_1_channels[0] // 2,
        ], relu_type=self.model_parameters.activation)
        self.block_1_up = ConvBlock(
            self.model_parameters.up_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_2 = Up([
            self.model_parameters.up_1_channels[-1],
            self.model_parameters.up_2_channels[0] // 2,
        ], relu_type=self.model_parameters.activation)
        self.block_2_up = ConvBlock(
            self.model_parameters.up_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_3 = Up([
            self.model_parameters.up_2_channels[-1],
            self.model_parameters.up_3_channels[0] // 2,
        ], relu_type=self.model_parameters.activation)
        self.block_3_up = ConvBlock(
            self.model_parameters.up_3_channels,
            relu_type=self.model_parameters.activation,
        )
        """self.up_4 = Up([
            self.model_parameters.up_3_channels[-1],
            self.model_parameters.up_4_channels[0] // 2,
        ], relu_type=self.model_parameters.activation)
        self.block_4_up = ConvBlock(
            self.model_parameters.up_4_channels,
            relu_type=self.model_parameters.activation,
        )"""
        self.block_last = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.last_block[0],
                self.model_parameters.last_block[1],
                self.model_parameters.last_block[2],
                padding=0,
            )
        )

    @staticmethod
    def default_parameters():
        class Params:
            # For 256x256, 2 input channels (measurement + slice_idx), 3 down/up blocks, HALF channels for 1 quadrant
            down_1_channels = [2, 32, 32, 32]
            down_2_channels = [32, 64, 64]
            down_3_channels = [64, 128, 128]
            latent_channels = [128, 256, 256]
            up_1_channels = [256, 128, 128]
            up_2_channels = [128, 64, 64]
            up_3_channels = [64, 32, 32]
            last_block = [32, 1, 1]
            activation = "ReLU"
        return Params()

    def forward(self, x):
        """
        Forward pass for the UNet model.
        Args:
            x (Tensor): Input tensor of shape (N, 2, 256, 256) or (N, 2, 512, 512).
                Channel 0: measurement, Channel 1: slice index.
        Returns:
            Tensor: Output tensor of shape (N, 1, 256, 256) or (N, 1, 512, 512).
        """
        # Ensure input is (N, 2, H, W): N = batch_size * 5
        assert x.dim() == 4, f"Input must be 4D (N, 2, H, W), got {x.shape}"
        assert x.shape[1] == 2, f"Input channel dimension must be 2 (measurement + slice_idx), got {x.shape[1]}"
        assert x.shape[2] == x.shape[3], f"Spatial dimensions must be square, got {x.shape[2:]}"
        
        block_1_res = self.block_1_down(x)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        res = self.block_bottom(self.down_3(block_3_res))
        res = self.block_1_up(torch.cat((block_3_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_2_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_1_res, self.up_3(res)), dim=1))
        res = self.block_last(res)
        return res  # Output shape: (N, 1, H, W)

