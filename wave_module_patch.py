"""
This file should be saved as wave_module_patch.py
It contains patched versions of both wave pooling and reconstruction modules
to fix memory sharing issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Patched wave pooling modules to avoid memory sharing issues
class WavePool2D(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(WavePool2D, self).__init__()
        
        # Create independent weight tensors for each filter to avoid memory sharing
        self.LL_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.LH_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.HL_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.HH_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        
        # Initialize with Haar wavelet coefficients
        with torch.no_grad():
            # LL filter (low-pass)
            self.LL_weight.data[:, 0, 0, 0] = 0.5
            self.LL_weight.data[:, 0, 0, 1] = 0.5
            self.LL_weight.data[:, 0, 1, 0] = 0.5
            self.LL_weight.data[:, 0, 1, 1] = 0.5
            
            # LH filter (vertical)
            self.LH_weight.data[:, 0, 0, 0] = 0.5
            self.LH_weight.data[:, 0, 0, 1] = 0.5
            self.LH_weight.data[:, 0, 1, 0] = -0.5
            self.LH_weight.data[:, 0, 1, 1] = -0.5
            
            # HL filter (horizontal)
            self.HL_weight.data[:, 0, 0, 0] = 0.5
            self.HL_weight.data[:, 0, 0, 1] = -0.5
            self.HL_weight.data[:, 0, 1, 0] = 0.5
            self.HL_weight.data[:, 0, 1, 1] = -0.5
            
            # HH filter (diagonal)
            self.HH_weight.data[:, 0, 0, 0] = 0.5
            self.HH_weight.data[:, 0, 0, 1] = -0.5
            self.HH_weight.data[:, 0, 1, 0] = -0.5
            self.HH_weight.data[:, 0, 1, 1] = 0.5
    
    def forward(self, x):
        # Apply each filter independently
        batch_size, channels, height, width = x.shape
        
        # Pad input for valid convolution
        pad = nn.ReflectionPad2d((0, 1, 0, 1))
        x_pad = pad(x)
        
        # Apply each filter
        LL = F.conv2d(x_pad, self.LL_weight, stride=2, groups=channels)
        LH = F.conv2d(x_pad, self.LH_weight, stride=2, groups=channels) 
        HL = F.conv2d(x_pad, self.HL_weight, stride=2, groups=channels)
        HH = F.conv2d(x_pad, self.HH_weight, stride=2, groups=channels)
        
        return LL, LH, HL, HH

# Patched wave reconstruction module
class WaveUnpool2D(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5', *args, **kwargs):
        super(WaveUnpool2D, self).__init__()
        self.option_unpool = option_unpool
        
        # Create independent weight tensors for each filter to avoid memory sharing
        self.LL_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.LH_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.HL_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        self.HH_weight = nn.Parameter(torch.zeros(in_channels, 1, 2, 2))
        
        # Initialize with inverse Haar wavelet coefficients
        with torch.no_grad():
            # LL filter (low-pass)
            self.LL_weight.data[:, 0, 0, 0] = 0.5
            self.LL_weight.data[:, 0, 0, 1] = 0.5
            self.LL_weight.data[:, 0, 1, 0] = 0.5
            self.LL_weight.data[:, 0, 1, 1] = 0.5
            
            # LH filter (vertical)
            self.LH_weight.data[:, 0, 0, 0] = 0.5
            self.LH_weight.data[:, 0, 0, 1] = 0.5
            self.LH_weight.data[:, 0, 1, 0] = -0.5
            self.LH_weight.data[:, 0, 1, 1] = -0.5
            
            # HL filter (horizontal)
            self.HL_weight.data[:, 0, 0, 0] = 0.5
            self.HL_weight.data[:, 0, 0, 1] = -0.5
            self.HL_weight.data[:, 0, 1, 0] = 0.5
            self.HL_weight.data[:, 0, 1, 1] = -0.5
            
            # HH filter (diagonal)
            self.HH_weight.data[:, 0, 0, 0] = 0.5
            self.HH_weight.data[:, 0, 0, 1] = -0.5
            self.HH_weight.data[:, 0, 1, 0] = -0.5
            self.HH_weight.data[:, 0, 1, 1] = 0.5
    
    def forward(self, LL, LH=None, HL=None, HH=None, skip_type=None):
        # Apply transposed convolution to each component
        batch_size, channels, height, width = LL.shape
        
        # Create reconstruction if needed
        if LH is None and HL is None and HH is None:
            # Simply upsample LL
            return F.interpolate(LL, scale_factor=2, mode='nearest')
        
        # Otherwise, reconstruct using all components
        output_shape = (batch_size, channels, height*2, width*2)
        
        # Transposed convolution for each component
        rec_LL = F.conv_transpose2d(LL, self.LL_weight, stride=2, groups=channels)
        rec_LH = F.conv_transpose2d(LH, self.LH_weight, stride=2, groups=channels) if LH is not None else 0
        rec_HL = F.conv_transpose2d(HL, self.HL_weight, stride=2, groups=channels) if HL is not None else 0
        rec_HH = F.conv_transpose2d(HH, self.HH_weight, stride=2, groups=channels) if HH is not None else 0
        
        # Sum all components for final reconstruction
        reconstructed = rec_LL + rec_LH + rec_HL + rec_HH
        
        return reconstructed

# Functions to patch encoder and decoder models
def patch_wave_encoder(encoder):
    """Replace problematic wave pooling layers with our fixed version"""
    # Create patched pooling layers
    if hasattr(encoder, 'pool1'):
        new_pool1 = WavePool2D(64)
        encoder.pool1 = new_pool1
    
    if hasattr(encoder, 'pool2'):
        new_pool2 = WavePool2D(128)
        encoder.pool2 = new_pool2
    
    if hasattr(encoder, 'pool3'):
        new_pool3 = WavePool2D(256)
        encoder.pool3 = new_pool3
    
    return encoder

def patch_wave_decoder(decoder):
    """Replace problematic wave reconstruction layers with our fixed version"""
    # Create patched reconstruction layers
    option_unpool = getattr(decoder, 'option_unpool', 'cat5')
    
    if hasattr(decoder, 'recon_block1'):
        new_recon1 = WaveUnpool2D(64, option_unpool)
        decoder.recon_block1 = new_recon1
    
    if hasattr(decoder, 'recon_block2'):
        new_recon2 = WaveUnpool2D(128, option_unpool)
        decoder.recon_block2 = new_recon2
    
    if hasattr(decoder, 'recon_block3'):
        new_recon3 = WaveUnpool2D(256, option_unpool)
        decoder.recon_block3 = new_recon3
    
    return decoder

# Function to safely load weights into our patched models
def load_wave_encoder_weights(encoder, state_dict_path):
    """Load weights from state dict into patched encoder model"""
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Create a new state dict for the patched model
    new_state_dict = {}
    
    # Copy non-pool weights directly
    for k, v in state_dict.items():
        if not any(x in k for x in ['pool1', 'pool2', 'pool3']):
            new_state_dict[k] = v
    
    # For pool weights, we need to handle them specially
    pool_weights_map = {
        'pool1.LL.weight': 'pool1.LL_weight',
        'pool1.LH.weight': 'pool1.LH_weight',
        'pool1.HL.weight': 'pool1.HL_weight',
        'pool1.HH.weight': 'pool1.HH_weight',
        'pool2.LL.weight': 'pool2.LL_weight',
        'pool2.LH.weight': 'pool2.LH_weight',
        'pool2.HL.weight': 'pool2.HL_weight',
        'pool2.HH.weight': 'pool2.HH_weight',
        'pool3.LL.weight': 'pool3.LL_weight',
        'pool3.LH.weight': 'pool3.LH_weight',
        'pool3.HL.weight': 'pool3.HL_weight',
        'pool3.HH.weight': 'pool3.HH_weight',
    }
    
    # For each pool weight, create a fresh tensor and copy data
    for old_name, new_name in pool_weights_map.items():
        if old_name in state_dict:
            # Create fresh tensor with numpy to break any shared memory
            tensor_data = state_dict[old_name].cpu().numpy()
            new_tensor = torch.tensor(tensor_data, dtype=state_dict[old_name].dtype)
            new_state_dict[new_name] = new_tensor
    
    # Load weights using strict=False to allow for missing/extra keys
    missing_keys, unexpected_keys = encoder.load_state_dict(new_state_dict, strict=False)
    
    return encoder

def load_wave_decoder_weights(decoder, state_dict_path):
    """Load weights from state dict into patched decoder model"""
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Create a new state dict for the patched model
    new_state_dict = {}
    
    # Copy non-recon weights directly
    for k, v in state_dict.items():
        if not any(x in k for x in ['recon_block1', 'recon_block2', 'recon_block3']):
            new_state_dict[k] = v
    
    # For recon weights, we need to handle them specially
    recon_weights_map = {
        'recon_block1.LL.weight': 'recon_block1.LL_weight',
        'recon_block1.LH.weight': 'recon_block1.LH_weight',
        'recon_block1.HL.weight': 'recon_block1.HL_weight',
        'recon_block1.HH.weight': 'recon_block1.HH_weight',
        'recon_block2.LL.weight': 'recon_block2.LL_weight',
        'recon_block2.LH.weight': 'recon_block2.LH_weight',
        'recon_block2.HL.weight': 'recon_block2.HL_weight',
        'recon_block2.HH.weight': 'recon_block2.HH_weight',
        'recon_block3.LL.weight': 'recon_block3.LL_weight',
        'recon_block3.LH.weight': 'recon_block3.LH_weight',
        'recon_block3.HL.weight': 'recon_block3.HL_weight',
        'recon_block3.HH.weight': 'recon_block3.HH_weight',
    }
    
    # For each recon weight, create a fresh tensor and copy data
    for old_name, new_name in recon_weights_map.items():
        if old_name in state_dict:
            # Create fresh tensor with numpy to break any shared memory
            tensor_data = state_dict[old_name].cpu().numpy()
            new_tensor = torch.tensor(tensor_data, dtype=state_dict[old_name].dtype)
            new_state_dict[new_name] = new_tensor
    
    # Load weights using strict=False to allow for missing/extra keys
    missing_keys, unexpected_keys = decoder.load_state_dict(new_state_dict, strict=False)
    
    return decoder