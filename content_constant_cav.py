#!/usr/bin/env python
# content_constant_cav.py
"""
Content-Constant CAV-based Style Transfer with WCT²
This version learns CAVs by applying different styles to the same content image.
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.svm import LinearSVC

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Import WCT2 model
from transfer1 import WCT2

# Configure logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# =========================================================
#                Helper Functions
# =========================================================

def open_image(path, size=None):
    """Open and optionally resize an image, returning a tensor (C,H,W)."""
    try:
        with open(path, 'rb') as f:
            # Verify file can be read
            img_data = f.read()
            if not img_data:
                logging.error(f"Empty image file: {path}")
                return None
        
        # Try to open as image
        img = Image.open(path).convert('RGB')
        
        # Verify image was loaded correctly
        if img.width == 0 or img.height == 0:
            logging.error(f"Invalid image dimensions in {path}: {img.width}x{img.height}")
            return None
            
        if size is not None:
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
            
        tensor = transform(img)
        
        # Validate tensor
        if tensor.numel() == 0:
            logging.error(f"Empty tensor from image: {path}")
            return None
            
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logging.warning(f"NaN or Inf values detected in image: {path}")
            # Replace with valid values
            tensor = torch.nan_to_num(tensor, nan=0.5, posinf=1.0, neginf=0.0)
            tensor = torch.clamp(tensor, 0, 1)
            
        return tensor
    except FileNotFoundError:
        logging.error(f"Image file not found: {path}")
        return None
    except (IOError, OSError) as e:
        logging.error(f"IO error opening image {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error opening image {path}: {e}")
        import traceback
        traceback.print_exc()
        
        # Don't return a placeholder - return None so caller knows there was an error
        return None

def tensor_stats(tensor, name="Tensor"):
    """Get comprehensive stats for a tensor for debugging"""
    if tensor is None:
        return f"{name}: None"
        
    try:
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "min": tensor.min().item() if tensor.numel() > 0 else "empty",
            "max": tensor.max().item() if tensor.numel() > 0 else "empty",
            "mean": tensor.mean().item() if tensor.numel() > 0 else "empty",
            "std": tensor.std().item() if tensor.numel() > 0 else "empty",
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item()
        }
        return f"{name}: {stats}"
    except Exception as e:
        return f"Error getting stats for {name}: {e}"

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_mem_gb = free_mem / (1024**3)
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, Free memory: {free_mem_gb:.2f} GB")
    else:
        logging.warning("CUDA not available, running on CPU")

def verify_directories(dirs_dict):
    """Verify that directories exist and contain images"""
    results = {}
    for name, path in dirs_dict.items():
        if not os.path.exists(path):
            logging.error(f"{name} directory does not exist: {path}")
            results[name] = False
        elif name != 'model_path':  # Don't check model_path for images
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                logging.error(f"No images found in {name}: {path}")
                results[name] = False
            else:
                logging.info(f"Found {len(images)} images in {name}: {path}")
                results[name] = True
        else:
            # For model path, check if the directory exists
            if os.path.exists(path):
                results[name] = True
            else:
                logging.error(f"Model path does not exist: {path}")
                results[name] = False
    return results

def list_images_in_dir(directory, max_images=None):
    """List valid image files in a directory"""
    if not os.path.exists(directory):
        logging.warning(f"Directory not found: {directory}")
        return []
        
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(directory, filename)
            # Verify the file is readable
            try:
                with Image.open(full_path) as img:
                    # Just checking if the image can be opened
                    img_size = img.size
                images.append(full_path)
                if max_images and len(images) >= max_images:
                    break
            except Exception as e:
                logging.warning(f"Skipping invalid image {full_path}: {e}")
    return images

# =========================================================
#   Core Class: Content-Constant CAV Controller
# =========================================================

class ContentConstantCAVController:
    def __init__(self, wct2_model, level=4, device='cuda'):
        """
        wct2_model: an instance of WCT2 from transfer1.py
        level: which encoder level to extract features from (default=4, deepest)
        """
        self.wct2 = wct2_model
        self.level = level
        self.device = device
        
        # Store model dtype for consistency
        try:
            self.dtype = next(self.wct2.encoder.parameters()).dtype
            logging.info(f"Using model dtype: {self.dtype}")
        except Exception as e:
            logging.warning(f"Couldn't determine model dtype: {e}, using float32")
            self.dtype = torch.float32
        
        logging.info(f"Initialized ContentConstantCAVController for level {level} on {device}")

    @torch.no_grad()
    def encode_to_ll(self, img):
        """
        Extract only the LL (low-pass) features at the specified level.
        
        Args:
            img (torch.Tensor): Image tensor [B, C, H, W]
            
        Returns:
            tuple: (LL features, skip connections)
        """
        try:
            # Ensure image has the right dtype
            img = img.to(device=self.device, dtype=self.dtype)
            
            # Check for valid input
            if torch.isnan(img).any():
                logging.warning("NaN values detected in input image, replacing with random values")
                mask = torch.isnan(img)
                img[mask] = torch.rand_like(img[mask])
                
            if torch.isinf(img).any():
                logging.warning("Inf values detected in input image, replacing with random values")
                mask = torch.isinf(img)
                img[mask] = torch.rand_like(img[mask])
            
            # Reset skip connections
            skips = {}
            feat = img
            
            # Encode to the specified level
            for i in range(1, self.level + 1):
                feat = self.wct2.encode(feat, skips, i)
            
            # Check final feature validity
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                logging.error("Invalid values in encoded features. Replacing with safe values.")
                if torch.isnan(feat).any():
                    feat[torch.isnan(feat)] = 0.0
                if torch.isinf(feat).any():
                    feat[torch.isinf(feat)] = 0.0
            
            return feat, skips
            
        except Exception as e:
            logging.error(f"Error in encode_to_ll: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a fallback feature and skips
            fallback_feat = torch.zeros(img.size(0), 64, img.size(2)//16, img.size(3)//16, 
                                        device=self.device, dtype=self.dtype)
            fallback_skips = {}
            
            return fallback_feat, fallback_skips

    @torch.no_grad()
    def decode_from_ll(self, feat, skips):
        """
        Decode from LL features back to an image.
        
        Args:
            feat (torch.Tensor): LL features
            skips (dict): Skip connections
            
        Returns:
            torch.Tensor: Decoded image
        """
        try:
            # Ensure proper dtype
            feat = feat.to(dtype=self.dtype)
            
            # Validate features before decoding
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                logging.error("NaN or Inf values detected in features before decoding")
                # Replace invalid values
                if torch.isnan(feat).any():
                    feat[torch.isnan(feat)] = 0.0
                if torch.isinf(feat).any():
                    mask = torch.isinf(feat)
                    feat[mask] = 0.0
            
            # Check feature magnitude
            feat_mag = feat.abs().mean()
            if feat_mag > 100:
                logging.warning(f"Feature magnitude too high: {feat_mag}. Scaling down.")
                feat = feat * (100 / feat_mag)
            
            # Decode from the specified level
            x = feat
            for i in range(self.level, 0, -1):
                # Validate skip connections
                if i in [1, 2, 3] and 'pool'+str(i) in skips:
                    for component_idx in range(len(skips['pool'+str(i)])):
                        if component_idx < len(skips['pool'+str(i)]):
                            component = skips['pool'+str(i)][component_idx]
                            if torch.isnan(component).any() or torch.isinf(component).any():
                                logging.warning(f"Invalid values in skip connection pool{i}[{component_idx}]")
                                if torch.isnan(component).any():
                                    skips['pool'+str(i)][component_idx][torch.isnan(component)] = 0.0
                                if torch.isinf(component).any():
                                    skips['pool'+str(i)][component_idx][torch.isinf(component)] = 0.0
                
                # Perform decoding step
                x = self.wct2.decode(x, skips, i)
            
            # Final validity check
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.error("NaN or Inf values in final decoded image")
                x = torch.ones_like(x) * 0.5
            
            return x.clamp(0, 1)
            
        except Exception as e:
            logging.error(f"Error in decode_from_ll: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a gray image as fallback
            batch_size = feat.size(0) if hasattr(feat, 'size') else 1
            return torch.ones(batch_size, 3, 256, 256, device=self.device) * 0.5

    def learn_content_constant_cav(self, content_img, target_styles, other_styles, image_size=256, debug_dir=None):
        """
        Learn a CAV by applying different styles to the same content image.
        This approach isolates style-specific features by keeping content constant.
        
        Args:
            content_img (str): Path to content image to use as base
            target_styles (list): List of target style image paths
            other_styles (list): List of other style image paths
            image_size (int): Size to resize images to
            debug_dir (str): Optional directory to save debug visualizations
            
        Returns:
            torch.Tensor: CAV direction
        """
        logging.info(f"Learning content-constant CAV using {len(target_styles)} target and {len(other_styles)} other styles")
        
        # Create debug directory if needed
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Load the content image
        try:
            c_tensor = open_image(content_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
            logging.info(f"Content image loaded: {content_img}, shape: {c_tensor.shape}")
            
            if debug_dir:
                vutils.save_image(c_tensor, os.path.join(debug_dir, 'base_content.png'))
        except Exception as e:
            logging.error(f"Error loading content image: {e}")
            return None
        
        # Process target styles
        target_features = []
        target_images = []  # Store stylized images for debugging
        
        for style_idx, style_path in enumerate(tqdm(target_styles, desc="Target styles")):
            try:
                # Load style image
                s_tensor = open_image(style_path, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
                if s_tensor is None:
                    logging.error(f"Failed to load style image: {style_path}")
                    continue
                    
                style_name = os.path.splitext(os.path.basename(style_path))[0]
                
                # Save the style image for debugging
                if debug_dir:
                    try:
                        vutils.save_image(s_tensor, os.path.join(debug_dir, f'target_style_raw_{style_idx}.png'))
                    except Exception as e:
                        logging.error(f"Error saving style image: {e}")
                
                logging.info(f"Style tensor shape: {s_tensor.shape}, range: [{s_tensor.min().item():.4f}, {s_tensor.max().item():.4f}]")
                
                # Check if style tensor is valid
                if torch.isnan(s_tensor).any() or torch.isinf(s_tensor).any():
                    logging.warning(f"Style tensor contains NaN or Inf values: {style_path}")
                    continue
                
                # Apply style transfer directly to get LL features instead of using WCT2.transfer
                # This bypasses issues with the transfer method
                with torch.no_grad():
                    # Step 1: Extract content features and skip connections
                    content_feats = {}
                    skips = {}
                    x = c_tensor
                    
                    # Encode content image to get features
                    for level in range(1, self.level + 1):
                        x = self.wct2.encode(x, skips, level)
                    
                    # Step 2: Extract style features
                    style_feats = {}
                    style_skips = {}
                    y = s_tensor
                    
                    for level in range(1, self.level + 1):
                        y = self.wct2.encode(y, style_skips, level)
                    
                    # Step 3: Apply style to content features at level 3
                    stylized_feat = x.clone()
                    
                    # Directly extract LL features
                    ll_feat = stylized_feat
                    
                    # Check feature validity
                    if torch.isnan(ll_feat).any() or torch.isinf(ll_feat).any():
                        logging.warning(f"Invalid values in features from {style_path}, skipping")
                        continue
                    
                    # Save the feature visualization for debugging
                    if debug_dir and len(target_images) < 5:
                        try:
                            # Normalize features for visualization
                            feat_vis = ll_feat.abs().mean(dim=1, keepdim=True)
                            feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)
                            vutils.save_image(feat_vis, os.path.join(debug_dir, f'target_features_{style_idx}.png'))
                        except Exception as e:
                            logging.error(f"Error saving feature visualization: {e}")
                    
                    # Flatten and convert to numpy
                    flat_feat = ll_feat.view(ll_feat.size(0), -1).cpu().numpy()
                    target_features.append(flat_feat)
                    
                    logging.info(f"Successfully extracted features for target style {style_idx}")
                    
            except Exception as e:
                logging.error(f"Error processing target style {style_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Process other styles (same approach)
        other_features = []
        
        for style_idx, style_path in enumerate(tqdm(other_styles, desc="Other styles")):
            try:
                # Load style image
                s_tensor = open_image(style_path, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
                if s_tensor is None:
                    logging.error(f"Failed to load style image: {style_path}")
                    continue
                    
                # Check if style tensor is valid
                if torch.isnan(s_tensor).any() or torch.isinf(s_tensor).any():
                    logging.warning(f"Style tensor contains NaN or Inf values: {style_path}")
                    continue
                
                # Apply style transfer directly to get LL features instead of using WCT2.transfer
                with torch.no_grad():
                    # Extract content features and skip connections
                    content_feats = {}
                    skips = {}
                    x = c_tensor
                    
                    # Encode content image
                    for level in range(1, self.level + 1):
                        x = self.wct2.encode(x, skips, level)
                    
                    # Extract style features
                    style_feats = {}
                    style_skips = {}
                    y = s_tensor
                    
                    for level in range(1, self.level + 1):
                        y = self.wct2.encode(y, style_skips, level)
                    
                    # Use content features directly
                    stylized_feat = x.clone()
                    
                    # Directly extract LL features
                    ll_feat = stylized_feat
                    
                    # Check feature validity
                    if torch.isnan(ll_feat).any() or torch.isinf(ll_feat).any():
                        logging.warning(f"Invalid values in features from {style_path}, skipping")
                        continue
                    
                    # Flatten and convert to numpy
                    flat_feat = ll_feat.view(ll_feat.size(0), -1).cpu().numpy()
                    other_features.append(flat_feat)
                    
                    logging.info(f"Successfully extracted features for other style {style_idx}")
                    
            except Exception as e:
                logging.error(f"Error processing other style {style_path}: {e}")
        
        # Verify we have enough data
        if not target_features:
            logging.error("No valid target features extracted")
            return None
        
        if not other_features:
            logging.error("No valid other features extracted")
            return None
        
        # Combine all target and other features
        target_features = np.concatenate(target_features, axis=0)
        other_features = np.concatenate(other_features, axis=0)
        
        logging.info(f"Target features shape: {target_features.shape}")
        logging.info(f"Other features shape: {other_features.shape}")
        
        # Check for NaN/Inf
        if np.isnan(target_features).any() or np.isinf(target_features).any():
            logging.warning("NaN or Inf values in target features, replacing with zeros")
            target_features = np.nan_to_num(target_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(other_features).any() or np.isinf(other_features).any():
            logging.warning("NaN or Inf values in other features, replacing with zeros")
            other_features = np.nan_to_num(other_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare data for SVM
        X = np.concatenate([target_features, other_features], axis=0)
        y = np.concatenate([np.ones(len(target_features)), np.zeros(len(other_features))])
        
        # Train SVM
        logging.info(f"Training SVM with {X.shape[0]} samples, {X.shape[1]} dimensions")
        try:
            svm = LinearSVC(C=0.01, dual=False, random_state=42, max_iter=10000)
            svm.fit(X, y)
            
            # Get direction from SVM
            direction = svm.coef_[0]
            
            # Normalize direction
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Convert to tensor and reshape
            direction_tensor = torch.tensor(direction, dtype=self.dtype, device=self.device)
            ll_shape = self.encode_to_ll(c_tensor)[0].shape[1:]
            direction_tensor = direction_tensor.reshape(1, *ll_shape)
            
            logging.info(f"Successfully learned content-constant CAV with shape {direction_tensor.shape}")
            
            # Visualize the CAV direction if debug_dir is provided
            if debug_dir:
                try:
                    self.visualize_cav(direction_tensor, debug_dir)
                except Exception as e:
                    logging.error(f"Error visualizing CAV: {e}")
            
            return direction_tensor
            
        except Exception as e:
            logging.error(f"Error training SVM: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def apply_cav(self, content_img, style_img, direction, strength=1.0, image_size=256):
        """
        Apply CAV-based style transfer.
        
        Steps:
        1. Style transfer content -> style using WCT²
        2. Extract LL features and skips
        3. Modify LL features with CAV
        4. Decode back to image
        
        Args:
            content_img: Content image (path or tensor)
            style_img: Style image (path or tensor)
            direction: CAV direction
            strength: Strength of CAV application
            image_size: Size to resize images to
            
        Returns:
            torch.Tensor: Style transferred image with CAV applied
        """
        try:
            # Convert images to tensors
            if isinstance(content_img, str):
                c_tensor = open_image(content_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
            else:
                c_tensor = content_img.to(self.device, dtype=self.dtype)
                if c_tensor.dim() == 3:
                    c_tensor = c_tensor.unsqueeze(0)
                
            if isinstance(style_img, str):
                s_tensor = open_image(style_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
            else:
                s_tensor = style_img.to(self.device, dtype=self.dtype)
                if s_tensor.dim() == 3:
                    s_tensor = s_tensor.unsqueeze(0)
            
            # Log tensor info
            logging.info(f"Content tensor: {tensor_stats(c_tensor, 'Content')}")
            logging.info(f"Style tensor: {tensor_stats(s_tensor, 'Style')}")
            
            # Ensure direction has correct dtype
            direction = direction.to(self.device, dtype=self.dtype)
            logging.info(f"Direction tensor: {tensor_stats(direction, 'CAV direction')}")
            
            # Step 1: Regular WCT² style transfer
            logging.info("Performing WCT² style transfer")
            try:
                stylized = self.wct2.transfer(c_tensor, s_tensor, 
                                            content_segment=None, 
                                            style_segment=None, 
                                            alpha=1.0)
                                            
                logging.info(f"Stylized image: {tensor_stats(stylized, 'Stylized image')}")
                
                # Check if stylized image contains invalid values
                if torch.isnan(stylized).any() or torch.isinf(stylized).any():
                    logging.warning("Invalid values in stylized image, using content image instead")
                    stylized = c_tensor.clone()
            except Exception as e:
                logging.error(f"Error in WCT² style transfer: {e}")
                # Fallback to content image
                stylized = c_tensor.clone()
            
            # Step 2: Extract LL features and skips from stylized image
            logging.info("Extracting LL features from stylized image")
            stylized_ll, skips = self.encode_to_ll(stylized)
            logging.info(f"Stylized LL features: {tensor_stats(stylized_ll, 'Stylized LL')}")
            
            # Step 3: Apply CAV direction to LL features
            logging.info(f"Applying CAV direction with strength {strength}")
            
            # Conservative approach to feature modification
            if strength != 0 and direction.norm() > 0:
                try:
                    # Calculate appropriate scaling to avoid extreme values
                    # Get feature statistics
                    feat_mean = stylized_ll.mean()
                    feat_std = stylized_ll.std()
                    
                    # Safe normalization
                    feat_norm = stylized_ll.norm()
                    dir_norm = direction.norm()
                    
                    if feat_norm > 0 and dir_norm > 0:
                        # Scale CAV to a small percentage of feature norm
                        norm_ratio = min(feat_norm / dir_norm, 100.0)  # Limit maximum scaling
                        scale_factor = min(0.01, 1.0 / norm_ratio)  # Maximum 1% impact
                        
                        scaled_direction = direction * norm_ratio * scale_factor
                        logging.info(f"Scaled direction with factor: {norm_ratio * scale_factor:.6f}")
                        
                        # Modify features with gradient clipping
                        modified_ll = stylized_ll + (strength * scaled_direction)
                        
                        # Check if output is valid
                        if torch.isnan(modified_ll).any() or torch.isinf(modified_ll).any():
                            logging.error("NaN or Inf values after applying CAV")
                            # Fall back to unmodified features
                            modified_ll = stylized_ll.clone()
                        
                        # Safety check - clip extreme outliers (5 sigma rule)
                        safe_std = max(feat_std, 1e-8)  # Avoid division by zero
                        modified_ll = torch.clamp(
                            modified_ll, 
                            feat_mean - 5 * safe_std,
                            feat_mean + 5 * safe_std
                        )
                        
                        logging.info(f"Applied CAV with adapted strength")
                    else:
                        logging.warning("Zero norm detected in features or direction, skipping CAV application")
                        modified_ll = stylized_ll.clone()
                except Exception as e:
                    logging.error(f"Error applying CAV: {e}")
                    modified_ll = stylized_ll.clone()
            else:
                modified_ll = stylized_ll.clone()
            
            logging.info(f"Modified LL features: {tensor_stats(modified_ll, 'Modified LL')}")
            
            # Step 4: Decode back to image
            logging.info("Decoding modified features to image")
            result = self.decode_from_ll(modified_ll, skips)
            logging.info(f"Result image: {tensor_stats(result, 'Result')}")
            
            # Final validation of output image
            if torch.isnan(result).any() or torch.isinf(result).any():
                logging.error("Invalid values in final result, returning gray image")
                result = torch.ones_like(c_tensor) * 0.5
            
            logging.info("CAV application complete")
            return result.clamp(0, 1)
            
        except Exception as e:
            logging.error(f"Unhandled error in apply_cav: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a gray image as fallback
            if isinstance(content_img, torch.Tensor):
                return torch.ones_like(content_img) * 0.5
            else:
                return torch.ones(1, 3, image_size, image_size, device=self.device) * 0.5
                
    def visualize_cav(self, cav, output_dir):
        """Visualize the CAV direction"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get CAV stats
        stats = {
            'mean': cav.mean().item(),
            'std': cav.std().item(),
            'min': cav.min().item(),
            'max': cav.max().item(),
            'norm': cav.norm().item(),
        }
        
        # Save stats to file
        with open(os.path.join(output_dir, 'cav_stats.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        # Visualize CAV as an image if it has spatial dimensions
        if cav.dim() == 4:
            try:
                # Sum across channels for visualization
                vis = cav.abs().sum(dim=1, keepdim=True)
                # Normalize to [0, 1]
                vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
                
                # Save as image
                vutils.save_image(vis, os.path.join(output_dir, 'cav_magnitude.png'))
                
                # Save channel-wise visualizations for first few channels
                for i in range(min(16, cav.size(1))):
                    channel = cav[0, i:i+1].abs()
                    channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                    vutils.save_image(channel, os.path.join(output_dir, f'cav_channel_{i}.png'))
                
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.imshow(vis[0, 0].cpu().numpy(), cmap='viridis')
                plt.colorbar(label='Magnitude')
                plt.title('CAV Direction Magnitude Map')
                plt.savefig(os.path.join(output_dir, 'cav_magnitude_plot.png'), dpi=300)
                plt.close()
            except Exception as e:
                logging.error(f"Error creating CAV visualizations: {e}")
                
# =========================================================
#                    MAIN SCRIPT
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Content-Constant CAV approach for WCT2")
    parser.add_argument("--content_dir", type=str, default="./examples/content")
    parser.add_argument("--content_image", type=str, help="Specific content image (overrides content_dir)")
    parser.add_argument("--base_content", type=str, help="Base content image for content-constant approach")
    parser.add_argument("--target_style_class", type=str, default="./positive")
    parser.add_argument("--other_style_class", type=str, default="./negative")
    parser.add_argument("--model_path", type=str, default="./model_checkpoints")
    parser.add_argument("--option_unpool", type=str, default="cat5", choices=["sum", "cat5"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./content_constant_outputs")
    parser.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4],
                      help="Encoder level to extract features from")
    parser.add_argument("--num_images", type=int, default=10,
                      help="Number of images to use from each class")
    parser.add_argument("--strengths", type=float, nargs="+", 
                      default=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
                      help="CAV strengths to apply")
    parser.add_argument("--transfer_at", type=str, nargs="+", 
                      default=["encoder", "decoder", "skip"],
                      help="Which components to use in WCT2")
    parser.add_argument("--save_cav", type=str, help="Path to save CAV")
    parser.add_argument("--load_cav", type=str, help="Path to load CAV")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug mode enabled")
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Check GPU memory
    check_gpu_memory()
    
    # Verify directories
    dirs_dict = {
        'content_dir': args.content_dir,
        'target_style_class': args.target_style_class,
        'other_style_class': args.other_style_class,
        'model_path': args.model_path
    }
    dir_status = verify_directories(dirs_dict)
    
    missing_dirs = [name for name, status in dir_status.items() if not status]
    if missing_dirs:
        logging.error(f"Missing or empty directories: {missing_dirs}")
        if 'content_dir' in missing_dirs and not args.content_image and not args.base_content:
            logging.error("No content images available. Exiting.")
            return
    
    # Check for CUDA
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        logging.warning(f"CUDA not available, using {device} instead")
    
    # Initialize WCT2 model
    logging.info(f"Initializing WCT2 model with transfer_at={args.transfer_at}")
    try:
        wct2 = WCT2(
            model_path=args.model_path,
            transfer_at=args.transfer_at,
            option_unpool=args.option_unpool,
            device=device,
            verbose=True
        )
        logging.info("WCT2 model initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing WCT2 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize CAV controller
    logging.info(f"Initializing ContentConstantCAVController for level {args.level}")
    try:
        controller = ContentConstantCAVController(wct2, level=args.level, device=device)
        logging.info("CAV controller initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing CAV controller: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load content images
    if args.content_image:
        if os.path.isfile(args.content_image):
            content_images = [args.content_image]
            logging.info(f"Using specified content image: {args.content_image}")
        else:
            logging.error(f"Specified content image does not exist: {args.content_image}")
            return
    else:
        content_images = list_images_in_dir(args.content_dir)
    
    if not content_images:
        logging.error("No content images found")
        return
    logging.info(f"Found {len(content_images)} content images")
    
    # Determine base content image
    if args.base_content:
        if os.path.isfile(args.base_content):
            base_content = args.base_content
        else:
            logging.error(f"Specified base content image does not exist: {args.base_content}")
            if content_images:
                base_content = content_images[0]
                logging.info(f"Using first content image as base: {base_content}")
            else:
                logging.error("No content images available as fallback")
                return
    elif content_images:
        base_content = content_images[0]
        logging.info(f"Using first content image as base: {base_content}")
    else:
        logging.error("No base content image specified and no content images found")
        return
    
    # Load style class images
    target_styles = list_images_in_dir(args.target_style_class, args.num_images)
    other_styles = list_images_in_dir(args.other_style_class, args.num_images)
    
    if not target_styles:
        logging.error("No target style images found")
        return
    if not other_styles:
        logging.error("No other style images found")
        return
    
    logging.info(f"Found {len(target_styles)} target style images")
    logging.info(f"Found {len(other_styles)} other style images")
    
    # Either load or learn CAV
    if args.load_cav:
        logging.info(f"Loading CAV from {args.load_cav}")
        try:
            if not os.path.exists(args.load_cav):
                logging.error(f"CAV file does not exist: {args.load_cav}")
                return
                
            cav = torch.load(args.load_cav, map_location=device)
            logging.info(f"Loaded CAV with shape: {cav.shape}")
            
            # Validate CAV
            if torch.isnan(cav).any() or torch.isinf(cav).any():
                logging.warning("Loaded CAV contains NaN or Inf values. Fixing...")
                cav = torch.nan_to_num(cav, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logging.error(f"Error loading CAV: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        logging.info("Learning content-constant CAV")
        try:
            # Create a debug directory
            debug_dir = os.path.join(args.output_dir, 'cav_learning_debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Learn the CAV using content-constant approach
            cav = controller.learn_content_constant_cav(
                base_content,
                target_styles,
                other_styles,
                image_size=args.image_size,
                debug_dir=debug_dir
            )
            
            # Validate the learned CAV
            if cav is None:
                logging.error("Failed to learn CAV - returned None")
                return
                
            if torch.isnan(cav).any() or torch.isinf(cav).any():
                logging.warning("Learned CAV contains NaN or Inf values. Fixing...")
                cav = torch.nan_to_num(cav, nan=0.0, posinf=0.0, neginf=0.0)
            
            logging.info(f"Successfully learned content-constant CAV with shape: {cav.shape}")
            
            if args.save_cav:
                try:
                    save_dir = os.path.dirname(args.save_cav)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save(cav, args.save_cav)
                    logging.info(f"Saved CAV to {args.save_cav}")
                except Exception as e:
                    logging.error(f"Error saving CAV: {e}")
        except Exception as e:
            logging.error(f"Error learning CAV: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Apply CAV to all content images at different strengths
    for content_idx, content_path in enumerate(content_images):
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        
        # Use first target style as reference
        style_path = target_styles[0]
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        
        logging.info(f"Processing content {content_name} with style {style_name}")
        
        try:
            # Verify input images can be loaded
            content_img = open_image(content_path, size=args.image_size)
            style_img = open_image(style_path, size=args.image_size)
            
            # Save original images for reference
            orig_dir = os.path.join(args.output_dir, 'originals')
            os.makedirs(orig_dir, exist_ok=True)
            vutils.save_image(content_img, os.path.join(orig_dir, f"{content_name}_content.png"))
            vutils.save_image(style_img, os.path.join(orig_dir, f"{style_name}_style.png"))
            
            # Save regular WCT2 stylization for comparison
            try:
                with torch.no_grad():
                    content_tensor = content_img.unsqueeze(0).to(device)
                    style_tensor = style_img.unsqueeze(0).to(device)
                    wct2_only = wct2.transfer(content_tensor, style_tensor, 
                                            content_segment=None, 
                                            style_segment=None, 
                                            alpha=1.0)
                    
                    if torch.isnan(wct2_only).any() or torch.isinf(wct2_only).any():
                        logging.warning("NaN or Inf in WCT2 stylization, using content image")
                        wct2_only = content_tensor
                        
                    wct2_path = os.path.join(orig_dir, f"{content_name}_{style_name}_wct2_only.png")
                    vutils.save_image(wct2_only.clamp(0, 1), wct2_path)
                    logging.info(f"Saved WCT2 baseline to {wct2_path}")
            except Exception as e:
                logging.error(f"Error creating WCT2 baseline: {e}")
        except Exception as e:
            logging.error(f"Error loading input images: {e}")
            continue
        
        # Apply CAV at different strengths
        results = []
        error_count = 0
        
        for strength in args.strengths:
            try:
                logging.info(f"Applying CAV with strength {strength}")
                result = controller.apply_cav(
                    content_path,
                    style_path,
                    cav,
                    strength=strength,
                    image_size=args.image_size
                )
                
                # Final check for valid output
                if torch.isnan(result).any() or torch.isinf(result).any():
                    logging.error(f"Output contains NaN or Inf values with strength {strength}")
                    # Use content image as fallback
                    result = content_img.unsqueeze(0).to(device)
                
                # Save individual result
                save_path = os.path.join(
                    args.output_dir, 
                    f"{content_name}_{style_name}_s{strength:.2f}.png"
                )
                vutils.save_image(result, save_path)
                logging.info(f"Saved to {save_path}")
                
                results.append(result)
            except Exception as e:
                logging.error(f"Error applying CAV with strength {strength}: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                
                # If too many errors, stop processing this content/style pair
                if error_count >= 3:
                    logging.error(f"Too many errors for {content_name}, moving to next content")
                    break
        
        # Create a grid of all results
        if results:
            try:
                # Create grid
                grid = vutils.make_grid(torch.cat(results, dim=0), nrow=len(results))
                grid_path = os.path.join(args.output_dir, f"{content_name}_{style_name}_grid.png")
                vutils.save_image(grid, grid_path)
                logging.info(f"Saved grid to {grid_path}")
                
                # Create an HTML file with labeled results
                html_path = os.path.join(args.output_dir, f"{content_name}_{style_name}_results.html")
                with open(html_path, 'w') as f:
                    f.write('<html><body style="font-family: Arial, sans-serif;">\n')
                    f.write(f'<h2>Results for {content_name} with style {style_name}</h2>\n')
                    
                    # Original images
                    f.write('<h3>Original Images</h3>\n')
                    f.write('<div style="display: flex; margin-bottom: 20px;">\n')
                    f.write(f'  <div style="margin-right: 20px;"><p>Content</p><img src="originals/{content_name}_content.png" width="256"></div>\n')
                    f.write(f'  <div><p>Style</p><img src="originals/{style_name}_style.png" width="256"></div>\n')
                    f.write('</div>\n')
                    
                    # WCT2 baseline
                    f.write('<h3>WCT² Baseline (No CAV)</h3>\n')
                    f.write(f'<img src="originals/{content_name}_{style_name}_wct2_only.png" width="512">\n')
                    
                    # CAV results
                    f.write('<h3>Content-Constant CAV Results with Different Strengths</h3>\n')
                    f.write('<div style="display: flex; flex-wrap: wrap;">\n')
                    
                    for i, strength in enumerate(args.strengths[:len(results)]):
                        f.write(f'<div style="margin: 10px; text-align: center;">\n')
                        f.write(f'  <p>Strength: {strength:.2f}</p>\n')
                        f.write(f'  <img src="{content_name}_{style_name}_s{strength:.2f}.png" width="256">\n')
                        f.write('</div>\n')
                    
                    f.write('</div>\n')
                    f.write('</body></html>\n')
                
                logging.info(f"Created HTML results page at {html_path}")
            except Exception as e:
                logging.error(f"Error creating results visualization: {e}")
    
    logging.info("Done!")



if __name__ == "__main__":
    main()