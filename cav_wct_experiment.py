import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA  # Import PCA
from sklearn.preprocessing import StandardScaler  # For feature scaling

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Import your WCT2 model
from transfer1 import WCT2
from utils.io import open_image, load_segment  # Import load_segment

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def open_image(path, size=None):
    """Open and optionally resize an image, returning a tensor (C,H,W)."""
    img = Image.open(path).convert('RGB')
    if size is not None:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    return transform(img)


class PCABasedCAVController:
    def __init__(self, wct2_model, level=4, device='cuda', n_components=90):
        self.wct2 = wct2_model
        self.level = level
        self.device = device
        self.n_components = n_components  # Number of PCA components to keep
        self.pca = None  # Will be initialized during learning
        self.scaler = StandardScaler()  # For feature scaling
        
        # Store model dtype for consistency
        self.dtype = next(self.wct2.encoder.parameters()).dtype
        
        logging.info(f"Initialized PCABasedCAVController for level {level} on {device} with {n_components} PCA components")

    @torch.no_grad()
    def encode_to_ll(self, img):
        # Ensure image has the right dtype
        img = img.to(device=self.device, dtype=self.dtype)
        
        # Reset skip connections
        skips = {}
        feat = img
        
        # Encode to the specified level
        for i in range(1, self.level + 1):
            feat = self.wct2.encode(feat, skips, i)
        
        return feat, skips

    @torch.no_grad()
    def decode_from_ll(self, feat, skips):
        # Ensure proper dtype
        feat = feat.to(dtype=self.dtype)
        x = feat
        for i in range(self.level, 0, -1):
            x = self.wct2.decode(x, skips, i)
        
        return x.clamp(0, 1)

    @torch.no_grad()
    def get_ll_features(self, img, image_size=256):
        # Convert input to tensor
        if isinstance(img, str):
            img_tensor = open_image(img, size=image_size).unsqueeze(0).to(self.device)
        elif isinstance(img, Image.Image):
            img_tensor = open_image(img, size=image_size).unsqueeze(0).to(self.device)
        else:
            # Assume it's already a tensor
            img_tensor = img.to(self.device)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
        
        # Extract features
        feat, skips = self.encode_to_ll(img_tensor)
        return feat, skips

    def learn_class_direction(self, target_imgs, other_imgs, image_size=256, batch_size=1):
        """
        Enhanced version with extensive debugging and diagnostics for class direction learning.
        """
        logging.info(f"DEBUGGING: Learning PCA-based CAV between {len(target_imgs)} target and {len(other_imgs)} other styles")
        
        # Validate input images exist and are readable
        logging.info("DEBUGGING: Validating input images...")
        valid_target_imgs = []
        valid_other_imgs = []
        
        for img_path in target_imgs:
            try:
                # Try to open image to verify it exists and is valid
                test_img = Image.open(img_path).convert('RGB')
                test_img.close()
                valid_target_imgs.append(img_path)
            except Exception as e:
                logging.error(f"DEBUGGING: Invalid target image {img_path}: {e}")
        
        for img_path in other_imgs:
            try:
                test_img = Image.open(img_path).convert('RGB')
                test_img.close()
                valid_other_imgs.append(img_path)
            except Exception as e:
                logging.error(f"DEBUGGING: Invalid other image {img_path}: {e}")
        
        logging.info(f"DEBUGGING: Valid images - Target: {len(valid_target_imgs)}/{len(target_imgs)}, Other: {len(valid_other_imgs)}/{len(other_imgs)}")
        
        if len(valid_target_imgs) == 0 or len(valid_other_imgs) == 0:
            raise ValueError("No valid images found in at least one class")
        
        # First, determine the feature shape by getting features from the first image
        logging.info("DEBUGGING: Extracting feature shape from first image...")
        first_img = valid_target_imgs[0] if valid_target_imgs else valid_other_imgs[0]
        try:
            first_feat, _ = self.get_ll_features(first_img, image_size=image_size)
            feat_shape = first_feat.shape
            logging.info(f"DEBUGGING: Feature shape from encoder level {self.level}: {feat_shape}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error extracting features from first image: {e}")
            raise
        
        # Store the flattened feature size for later reshaping
        flat_feat_size = np.prod(feat_shape[1:])
        logging.info(f"DEBUGGING: Flattened feature size: {flat_feat_size}")
        
        # Extract features from all images with detailed error reporting
        all_features = []
        all_labels = []
        
        # NEW: Track feature statistics for each class for later analysis
        target_features = []
        other_features = []
        
        # Additional diagnostics
        target_feature_stats = []
        other_feature_stats = []
        
        # Process target images
        for i in tqdm(range(0, len(valid_target_imgs), batch_size), desc="Target class encoding"):
            batch_imgs = valid_target_imgs[i:i+batch_size]
            for img_path in batch_imgs:
                try:
                    # Extract LL features
                    ll_feat, _ = self.get_ll_features(img_path, image_size=image_size)
                    
                    # Log feature statistics for debugging
                    feat_mean = torch.mean(ll_feat).item()
                    feat_std = torch.std(ll_feat).item()
                    feat_min = torch.min(ll_feat).item()
                    feat_max = torch.max(ll_feat).item()
                    target_feature_stats.append({
                        'path': img_path,
                        'mean': feat_mean,
                        'std': feat_std,
                        'min': feat_min,
                        'max': feat_max
                    })
                    
                    # Ensure same shape as first feature (in case of irregularities)
                    if ll_feat.shape != feat_shape:
                        logging.warning(f"DEBUGGING: Feature shape mismatch: {ll_feat.shape} vs {feat_shape}. Resizing.")
                        ll_feat = F.interpolate(ll_feat, size=feat_shape[2:], mode='bilinear', align_corners=False)
                    
                    # IMPROVEMENT: Apply channel-wise normalization for better comparison
                    ch_mean = torch.mean(ll_feat, dim=[2, 3], keepdim=True)
                    ch_std = torch.std(ll_feat, dim=[2, 3], keepdim=True) + 1e-5
                    ll_feat_norm = (ll_feat - ch_mean) / ch_std
                    
                    # Flatten and convert to numpy
                    flat_feat = ll_feat_norm.view(ll_feat.size(0), -1).cpu().numpy()
                    all_features.append(flat_feat)
                    all_labels.extend([1] * flat_feat.shape[0])  # 1 for target class
                    target_features.append(flat_feat)
                except Exception as e:
                    logging.error(f"DEBUGGING: Error processing target image {img_path}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        # Process other images with similar diagnostics
        for i in tqdm(range(0, len(valid_other_imgs), batch_size), desc="Other class encoding"):
            batch_imgs = valid_other_imgs[i:i+batch_size]
            for img_path in batch_imgs:
                try:
                    # Extract LL features
                    ll_feat, _ = self.get_ll_features(img_path, image_size=image_size)
                    
                    # Log feature statistics for debugging
                    feat_mean = torch.mean(ll_feat).item()
                    feat_std = torch.std(ll_feat).item()
                    feat_min = torch.min(ll_feat).item()
                    feat_max = torch.max(ll_feat).item()
                    other_feature_stats.append({
                        'path': img_path,
                        'mean': feat_mean,
                        'std': feat_std,
                        'min': feat_min,
                        'max': feat_max
                    })
                    
                    # Ensure same shape as first feature
                    if ll_feat.shape != feat_shape:
                        logging.warning(f"DEBUGGING: Feature shape mismatch: {ll_feat.shape} vs {feat_shape}. Resizing.")
                        ll_feat = F.interpolate(ll_feat, size=feat_shape[2:], mode='bilinear', align_corners=False)
                    
                    # Apply channel-wise normalization
                    ch_mean = torch.mean(ll_feat, dim=[2, 3], keepdim=True)
                    ch_std = torch.std(ll_feat, dim=[2, 3], keepdim=True) + 1e-5
                    ll_feat_norm = (ll_feat - ch_mean) / ch_std
                        
                    # Flatten and convert to numpy
                    flat_feat = ll_feat_norm.view(ll_feat.size(0), -1).cpu().numpy()
                    all_features.append(flat_feat)
                    all_labels.extend([0] * flat_feat.shape[0])  # 0 for other class
                    other_features.append(flat_feat)
                except Exception as e:
                    logging.error(f"DEBUGGING: Error processing other image {img_path}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        # Log detailed feature statistics 
        logging.info("DEBUGGING: Target class feature statistics:")
        for stat in target_feature_stats:
            logging.info(f"  {os.path.basename(stat['path'])}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
        
        logging.info("DEBUGGING: Other class feature statistics:")
        for stat in other_feature_stats:
            logging.info(f"  {os.path.basename(stat['path'])}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
        
        # Combine all features
        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            logging.info(f"DEBUGGING: Combined features shape: {all_features.shape}")
        else:
            logging.error("DEBUGGING: No features extracted")
            raise ValueError("No features extracted")
        
        # Check for NaN or infinite values
        nan_count = np.isnan(all_features).sum()
        inf_count = np.isinf(all_features).sum()
        if nan_count > 0 or inf_count > 0:
            logging.warning(f"DEBUGGING: Features contain {nan_count} NaN and {inf_count} infinite values. Replacing with zeros.")
            all_features = np.nan_to_num(all_features)
        
        # IMPROVEMENT: Analyze class separability before proceeding
        if target_features and other_features:
            target_concat = np.concatenate(target_features, axis=0)
            other_concat = np.concatenate(other_features, axis=0)
            
            # Calculate mean vectors for each class
            target_mean = np.mean(target_concat, axis=0)
            other_mean = np.mean(other_concat, axis=0)
            
            # Calculate feature statistics for debugging
            logging.info(f"DEBUGGING: Target features: shape={target_concat.shape}, mean={np.mean(target_concat):.4f}, std={np.std(target_concat):.4f}")
            logging.info(f"DEBUGGING: Other features: shape={other_concat.shape}, mean={np.mean(other_concat):.4f}, std={np.std(other_concat):.4f}")
            
            # Calculate class separation metric (distance between means)
            mean_diff = np.linalg.norm(target_mean - other_mean)
            logging.info(f"DEBUGGING: Class separation metric (mean difference norm): {mean_diff:.4f}")
            
            if mean_diff < 0.5:  # Arbitrary threshold, adjust based on experimentation
                logging.warning("DEBUGGING: Low class separation detected. CAV may not be effective.")
                logging.warning("DEBUGGING: Consider using more distinctive style classes.")
        
        # Scale features
        try:
            logging.info("DEBUGGING: Scaling features...")
            scaled_features = self.scaler.fit_transform(all_features)
            logging.info(f"DEBUGGING: Scaled features shape: {scaled_features.shape}")
            logging.info(f"DEBUGGING: Scaled features: mean={np.mean(scaled_features):.4f}, std={np.std(scaled_features):.4f}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error scaling features: {e}")
            raise
        
        # IMPROVEMENT: Select optimal number of PCA components based on explained variance
        try:
            logging.info("DEBUGGING: Running initial PCA to analyze variance distribution...")
            from sklearn.decomposition import PCA
            
            # Run a full PCA first to analyze variance distribution
            temp_pca = PCA(n_components=min(scaled_features.shape))
            temp_pca.fit(scaled_features)
            
            # Log variance distribution for debugging
            logging.info(f"DEBUGGING: PCA variance ratios (first 10): {temp_pca.explained_variance_ratio_[:10]}")
            
            # Find optimal number of components that explain at least 95% variance
            explained_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            logging.info(f"DEBUGGING: Cumulative explained variance (first 10): {explained_variance[:10]}")
            
            # Find first index where cumulative variance ≥ 0.95
            # Handle case where even all components don't reach 0.95
            if explained_variance[-1] < 0.95:
                logging.warning("DEBUGGING: Even all components don't explain 95% variance. Using all components.")
                optimal_components = len(explained_variance)
            else:
                optimal_components = np.argmax(explained_variance >= 0.95) + 1
            
            # Use at least 10 components, but no more than self.n_components
            n_components = min(max(optimal_components, 10), self.n_components)
            
            logging.info(f"DEBUGGING: Optimal PCA components for 95% variance: {optimal_components}")
            logging.info(f"DEBUGGING: Using {n_components} PCA components")
        except Exception as e:
            logging.error(f"DEBUGGING: Error analyzing PCA components: {e}")
            # Fallback to a reasonable default
            n_components = min(50, min(scaled_features.shape))
            logging.info(f"DEBUGGING: Falling back to {n_components} PCA components")
        
        # Apply PCA with optimized components
        try:
            logging.info(f"DEBUGGING: Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Log variance explained by each component
            logging.info(f"DEBUGGING: Top 5 component variance ratios: {self.pca.explained_variance_ratio_[:5]}")
            logging.info(f"DEBUGGING: Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            logging.info(f"DEBUGGING: PCA features shape: {pca_features.shape}")
            logging.info(f"DEBUGGING: PCA features: mean={np.mean(pca_features):.4f}, std={np.std(pca_features):.4f}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error applying PCA: {e}")
            raise
        
        # Check labels for imbalance
        label_counts = {0: all_labels.count(0), 1: all_labels.count(1)}
        logging.info(f"DEBUGGING: Label distribution: {label_counts}")
        if label_counts[0] == 0 or label_counts[1] == 0:
            logging.error("DEBUGGING: One of the classes has zero samples. Cannot train SVM.")
            raise ValueError("One of the classes has zero samples. Cannot train SVM.")
        
        # Perform train/test split to verify CAV generalization
        try:
            logging.info("DEBUGGING: Performing train/test split...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                pca_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
            )
            logging.info(f"DEBUGGING: Train set: {X_train.shape}, Test set: {X_test.shape}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error in train/test split: {e}")
            raise
        
        # IMPROVEMENT: Try different SVM types if regularization doesn't work
        best_c = None
        best_acc = 0
        best_svm = None
        
        try:
            # First try linear SVC with different regularization values
            logging.info("DEBUGGING: Training LinearSVC with different C values...")
            for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                try:
                    svm = LinearSVC(C=c, class_weight='balanced', dual=False, max_iter=10000, random_state=42)
                    svm.fit(X_train, y_train)
                    acc = svm.score(X_test, y_test)
                    logging.info(f"DEBUGGING: LinearSVC with C={c} test accuracy: {acc:.4f}")
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_c = c
                        best_svm = svm
                except Exception as e:
                    logging.error(f"DEBUGGING: Error training LinearSVC with C={c}: {e}")
            
            if best_svm is None:
                # Try RBF kernel SVM if linear doesn't work
                logging.info("DEBUGGING: LinearSVC failed. Trying SVM with RBF kernel...")
                from sklearn.svm import SVC
                for c in [0.1, 1.0, 10.0]:
                    for gamma in ['scale', 'auto']:
                        try:
                            svm = SVC(C=c, kernel='rbf', gamma=gamma, class_weight='balanced', random_state=42)
                            svm.fit(X_train, y_train)
                            acc = svm.score(X_test, y_test)
                            logging.info(f"DEBUGGING: SVC with kernel=rbf, C={c}, gamma={gamma} test accuracy: {acc:.4f}")
                            
                            if acc > best_acc:
                                best_acc = acc
                                best_c = f"C={c}, gamma={gamma}, kernel=rbf"
                                best_svm = svm
                        except Exception as e:
                            logging.error(f"DEBUGGING: Error training SVC with C={c}, gamma={gamma}: {e}")
            
            if best_svm is None:
                # As a last resort, try a simple logistic regression
                logging.info("DEBUGGING: SVM methods failed. Trying LogisticRegression...")
                from sklearn.linear_model import LogisticRegression
                try:
                    logreg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000)
                    logreg.fit(X_train, y_train)
                    acc = logreg.score(X_test, y_test)
                    logging.info(f"DEBUGGING: LogisticRegression test accuracy: {acc:.4f}")
                    
                    best_acc = acc
                    best_c = "LogisticRegression"
                    best_svm = logreg
                except Exception as e:
                    logging.error(f"DEBUGGING: Error training LogisticRegression: {e}")
                    
        except Exception as e:
            logging.error(f"DEBUGGING: Error in SVM training process: {e}")
            raise
        
        if best_svm is None:
            logging.error("DEBUGGING: All classification methods failed. Cannot learn class direction.")
            raise RuntimeError("All classification methods failed. Cannot learn class direction.")
        
        # Get the decision boundary
        if hasattr(best_svm, 'coef_'):
            # Linear models (LinearSVC, LogisticRegression)
            logging.info("DEBUGGING: Using linear model coefficients as direction")
            pca_direction = torch.tensor(best_svm.coef_[0], dtype=self.dtype, device=self.device)
        else:
            # Non-linear models (SVC with rbf kernel)
            logging.error("DEBUGGING: Model doesn't have a linear decision boundary. Cannot extract direction.")
            raise ValueError("Model doesn't have a linear decision boundary. Cannot extract direction.")
        
        # Log coefficient statistics
        coef_mean = pca_direction.mean().item()
        coef_std = pca_direction.std().item()
        coef_min = pca_direction.min().item()
        coef_max = pca_direction.max().item()
        logging.info(f"DEBUGGING: Direction coefficients: mean={coef_mean:.4f}, std={coef_std:.4f}, min={coef_min:.4f}, max={coef_max:.4f}")
        
        # Check if the direction is meaningful (not all zeros or NaN)
        if torch.isnan(pca_direction).any():
            logging.error("DEBUGGING: Direction contains NaN values.")
            raise ValueError("Direction contains NaN values.")
        if pca_direction.norm() < 1e-6:
            logging.error("DEBUGGING: Direction norm is nearly zero.")
            raise ValueError("Direction norm is nearly zero.")
        
        # Calculate feature importance for each PCA component
        feature_importance = np.abs(pca_direction.cpu().numpy())
        top_indices = np.argsort(feature_importance)[::-1]
        logging.info(f"DEBUGGING: Top 5 most important PCA components: {top_indices[:5]}")
        logging.info(f"DEBUGGING: Importance values for top components: {feature_importance[top_indices[:5]]}")
        
        # Apply soft thresholding to focus on most important components
        importance_threshold = np.percentile(feature_importance, 25)  # Cut off bottom 25%
        importance_mask = feature_importance > importance_threshold
        enhanced_pca_direction = pca_direction.clone()
        
        # Zero out less important components
        for i in range(len(enhanced_pca_direction)):
            if not importance_mask[i]:
                enhanced_pca_direction[i] = 0.0
        
        # Normalize direction to have unit norm
        enhanced_pca_direction = enhanced_pca_direction / (enhanced_pca_direction.norm() + 1e-8)
        
        # Project back to original space
        try:
            logging.info("DEBUGGING: Projecting direction back to original feature space...")
            original_direction = torch.tensor(
                self.pca.components_.T @ enhanced_pca_direction.cpu().numpy(),
                dtype=self.dtype, device=self.device
            )
            logging.info(f"DEBUGGING: Original space direction: shape={original_direction.shape}, norm={original_direction.norm().item():.4f}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error projecting direction back to original space: {e}")
            raise
        
        # VERIFICATION STEP: Check that we have the right number of elements for reshaping
        logging.info(f"DEBUGGING: Original direction size: {original_direction.size(0)}, expected: {flat_feat_size}")
        
        if original_direction.size(0) != flat_feat_size:
            logging.warning(f"DEBUGGING: Size mismatch: {original_direction.size(0)} vs expected {flat_feat_size}")
            if original_direction.size(0) > flat_feat_size:
                logging.warning(f"DEBUGGING: Truncating direction vector to match feature size")
                original_direction = original_direction[:flat_feat_size]
            else:
                logging.warning(f"DEBUGGING: Padding direction vector to match feature size")
                padding = torch.zeros(flat_feat_size - original_direction.size(0), dtype=self.dtype, device=self.device)
                original_direction = torch.cat([original_direction, padding])
        
        # Reshape to match LL features shape (keeping batch dim)
        try:
            direction = original_direction.reshape(1, *feat_shape[1:])
            logging.info(f"DEBUGGING: Reshaped CAV direction to {direction.shape}")
        except Exception as e:
            logging.error(f"DEBUGGING: Error reshaping direction: {e}")
            raise
        
        # Calculate and store channel importance
        channel_importance = torch.mean(direction.abs(), dim=[2, 3])
        top_channels = torch.argsort(channel_importance[0], descending=True)
        logging.info(f"DEBUGGING: Top 5 most important channels: {top_channels[:5].cpu().numpy()}")
        logging.info(f"DEBUGGING: Channel importance values: {channel_importance[0, top_channels[:5]].cpu().numpy()}")
        
        # Final verification - check if the direction is usable
        dir_mean = direction.mean().item()
        dir_std = direction.std().item()
        dir_norm = direction.norm().item()
        logging.info(f"DEBUGGING: Final direction: mean={dir_mean:.4f}, std={dir_std:.4f}, norm={dir_norm:.4f}")
        
        if dir_norm < 1e-6:
            logging.error("DEBUGGING: Final direction has too small norm. Not usable.")
            raise ValueError("Final direction has too small norm. Not usable.")
        
        # Return both the original direction and PCA components for later use
        logging.info("DEBUGGING: Successfully created CAV direction!")
        return {
            'direction': direction,
            'pca': self.pca,
            'scaler': self.scaler,
            'pca_direction': enhanced_pca_direction,  # Now using enhanced version
            'original_pca_direction': pca_direction,  # Keep original for reference
            'svm': best_svm,
            'test_accuracy': best_acc,
            'feature_shape': feat_shape,  # Store this for consistency checks later
            'channel_importance': channel_importance,  # NEW: Store channel importance
            'top_channels': top_channels.cpu().numpy(),  # NEW: Store indices of top channels
            'class_separation': mean_diff if 'mean_diff' in locals() else None,  # Store separation metric
        }

    @torch.no_grad()
    def apply_cav(self, content_img, style_img, cav_dict, strength=1.0, image_size=256, alpha=1.0, 
            stylization_method="wct"):
        """
        Apply CAV with enhanced options for better visual distinction.
        """
        # Extract direction from cav_dict
        direction = cav_dict['direction']
        
        # Convert images to tensors
        if isinstance(content_img, str):
            c_tensor = open_image(content_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
            # Try to load content segmentation mask by replacing directory in path
            content_segment_path = content_img.replace('/content/', '/content_segment/').replace('\\content\\', '\\content_segment\\')
            if os.path.exists(content_segment_path):
                content_segment = load_segment(content_segment_path, image_size)
                logging.info(f"Loaded content segmentation from {content_segment_path}")
            else:
                # Create default mask if specific mask doesn't exist
                content_segment = torch.ones(1, image_size, image_size).to(self.device)
                logging.info(f"Using default content segmentation mask")
        else:
            c_tensor = content_img.to(self.device, dtype=self.dtype)
            content_segment = torch.ones(1, c_tensor.size(2), c_tensor.size(3)).to(self.device)
            
        if isinstance(style_img, str):
            s_tensor = open_image(style_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
            # Try to load style segmentation mask
            if 'positive' in style_img:
                style_segment_path = style_img.replace('/positive/', '/positive_masks/').replace('\\positive\\', '\\positive_masks\\')
            else:
                style_segment_path = style_img.replace('/negative/', '/negative_masks/').replace('\\negative\\', '\\negative_masks\\')
            
            if os.path.exists(style_segment_path):
                style_segment = load_segment(style_segment_path, image_size)
                logging.info(f"Loaded style segmentation from {style_segment_path}")
            else:
                # Create default mask if specific mask doesn't exist
                style_segment = torch.ones(1, image_size, image_size).to(self.device)
                logging.info(f"Using default style segmentation mask")
        else:
            s_tensor = style_img.to(self.device, dtype=self.dtype)
            style_segment = torch.ones(1, s_tensor.size(2), s_tensor.size(3)).to(self.device)
        
        # Ensure direction has correct dtype
        direction = direction.to(self.device, dtype=self.dtype)
        
        # Step 1: Regular style transfer
        logging.info(f"Performing {stylization_method.upper()} style transfer with alpha={alpha}")
        stylized = None
        try:
            if stylization_method.lower() == "wct":
                logging.info(f"Using segmentation masks for style transfer")
                stylized = self.wct2.transfer(c_tensor, s_tensor, 
                                    content_segment=content_segment, 
                                    style_segment=style_segment, 
                                    alpha=alpha)
            else:
                # AdaIN-style transfer (faster but less detailed)
                content_feat, skips = self.encode_to_ll(c_tensor)
                style_feat, _ = self.encode_to_ll(s_tensor)
                
                # AdaIN operation
                content_mean = torch.mean(content_feat, dim=[2, 3], keepdim=True)
                content_std = torch.std(content_feat, dim=[2, 3], keepdim=True) + 1e-5
                style_mean = torch.mean(style_feat, dim=[2, 3], keepdim=True)
                style_std = torch.std(style_feat, dim=[2, 3], keepdim=True) + 1e-5
                
                normalized = (content_feat - content_mean) / content_std
                adapted = normalized * style_std + style_mean
                combined = alpha * adapted + (1 - alpha) * content_feat
                
                stylized = self.decode_from_ll(combined, skips)
        except Exception as e:
            logging.error(f"Error in style transfer: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        # Fallback to content image if style transfer failed
        if stylized is None:
            logging.warning("Style transfer failed. Using content image as fallback.")
            stylized = c_tensor.clone()
        
        # Step 2: Extract LL features and skips from stylized image
        logging.info("Extracting LL features from stylized image")
        stylized_ll, skips = self.encode_to_ll(stylized)
        
        # Get raw content and style features for visualization/debugging
        content_ll, content_skips = self.encode_to_ll(c_tensor)
        style_ll, _ = self.encode_to_ll(s_tensor)
        
        # Compute feature statistics for better normalization
        c_mean = torch.mean(content_ll)
        c_std = torch.std(content_ll)
        s_mean = torch.mean(style_ll)
        s_std = torch.std(style_ll)
        stylized_mean = torch.mean(stylized_ll)
        stylized_std = torch.std(stylized_ll)
        
        logging.info(f"Feature stats - Content: mean={c_mean:.4f}, std={c_std:.4f}")
        logging.info(f"Feature stats - Style: mean={s_mean:.4f}, std={s_std:.4f}")
        logging.info(f"Feature stats - Stylized: mean={stylized_mean:.4f}, std={stylized_std:.4f}")
        
        # Step 3: Apply CAV direction to LL features with adaptive scaling
        logging.info(f"Applying CAV direction with strength {strength}")
        
        # Initialize modified_ll to stylized_ll as a fallback
        modified_ll = stylized_ll.clone()
        
        # Normalize CAV to have similar scale as features, but with higher strength
        if strength != 0 and direction.norm() > 0:
            try:
                # Get channel importance if available
                channel_importance = None
                if 'channel_importance' in cav_dict:
                    channel_importance = cav_dict['channel_importance'].to(self.device)
                
                # Calculate the mean activation magnitude to better scale the CAV
                mean_activation = stylized_ll.abs().mean()
                direction_magnitude = direction.abs().mean()
                
                # Use a more conservative scale factor to prevent artifacts
                scale_factor = (mean_activation / direction_magnitude) * 0.2
                
                # Apply scaled direction with channel-wise normalization for better results
                scaled_direction = direction * scale_factor
                
                # Use channel importance as a mask if available (focus on relevant channels)
                if channel_importance is not None:
                    # Create an importance mask that emphasizes top channels
                    # Threshold at 50th percentile of channel importance
                    threshold = torch.quantile(channel_importance, 0.5)
                    importance_mask = (channel_importance > threshold).float()
                    importance_mask = importance_mask.unsqueeze(-1).unsqueeze(-1)
                    
                    # Apply mask to direction
                    scaled_direction = scaled_direction * importance_mask
                
                # Apply CAV modification with gradient clipping
                # CRITICAL FIX: Reduce extreme values that cause white spots
                cav_modification = torch.clamp(strength * scaled_direction, -2.0, 2.0)
                
                # Apply to features
                # OPTION 1: Direct addition (can cause artifacts)
                # modified_ll = stylized_ll + cav_modification
                
                # OPTION 2: Channel-wise normalization (safer approach)
                # First normalize stylized features
                stylized_mean = torch.mean(stylized_ll, dim=[2, 3], keepdim=True)
                stylized_std = torch.std(stylized_ll, dim=[2, 3], keepdim=True) + 1e-5
                normalized_stylized = (stylized_ll - stylized_mean) / stylized_std
                
                # Apply direction to normalized features (safer)
                modified_normalized = normalized_stylized + cav_modification
                
                # Apply tanh to constrain values (CRITICAL FIX FOR WHITE SPOTS)
                modified_normalized = torch.tanh(modified_normalized)
                
                # Denormalize back
                modified_ll = (modified_normalized * stylized_std) + stylized_mean
                
                # CRITICAL FIX: Apply robust feature blending to prevent extreme activations
                # This smoothly blends in the original stylized features to keep stability
                blend_factor = min(abs(strength) * 0.1, 0.5)  # Adaptive blending based on strength
                modified_ll = (1 - blend_factor) * modified_ll + blend_factor * stylized_ll
                
                logging.info(f"Applied CAV with adaptive scaling (factor: {scale_factor:.6f})")
                
                # Preserve content structure for high strength values
                if abs(strength) > 2.0:
                    # Blend with original content features to preserve structure
                    content_weight = min(0.3, abs(strength) * 0.05)  # Adaptive content preservation
                    modified_ll = (1 - content_weight) * modified_ll + content_weight * content_ll
                    logging.info(f"Applied content preservation blend for high strength")
            except Exception as e:
                logging.error(f"Error during CAV application: {e}")
                import traceback
                logging.error(traceback.format_exc())
                # Fall back to stylized features
                modified_ll = stylized_ll
        
        # CRITICAL FIX: More aggressive sanitization of feature values
        # Check for NaN or Inf values and replace
        if torch.isnan(modified_ll).any() or torch.isinf(modified_ll).any():
            logging.warning("NaN or Inf values detected in modified features. Replacing with cleaned values.")
            cleaned_features = torch.nan_to_num(modified_ll, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Apply modest clamping to prevent extreme values
            modified_ll = torch.clamp(cleaned_features, -10.0, 10.0)
        
        # CRITICAL FIX: Consider using a blend of content and modified skips
        # This helps preserve spatial structure and prevent artifacts
        blended_skips = {}
        for key, value in skips.items():
            if key in content_skips:
                # Check the type of value to determine how to blend
                if isinstance(value, torch.Tensor) and isinstance(content_skips[key], torch.Tensor):
                    # For tensors, we can use element-wise multiplication and addition
                    blended_skips[key] = 0.7 * value + 0.3 * content_skips[key]
                elif isinstance(value, (list, tuple)) and isinstance(content_skips[key], (list, tuple)):
                    # For sequences (lists/tuples), we need to create a new sequence
                    if len(value) == len(content_skips[key]):
                        # If they're the same length, blend element-wise
                        blended_skips[key] = []
                        for v_item, c_item in zip(value, content_skips[key]):
                            if isinstance(v_item, torch.Tensor) and isinstance(c_item, torch.Tensor):
                                blended_skips[key].append(0.7 * v_item + 0.3 * c_item)
                            else:
                                # If not tensors, just use the stylized value
                                blended_skips[key].append(v_item)
                    else:
                        # If not same length, just use original
                        blended_skips[key] = value
                else:
                    # If types don't match or aren't blendable, use original
                    blended_skips[key] = value
            else:
                # If not in content_skips, use original
                blended_skips[key] = value
        
        # Step 4: Decode back to image
        logging.info("Decoding modified features to image")
        try:
            # Use blended skips to decode (more stable)
            result = self.decode_from_ll(modified_ll, blended_skips)
            
            # CRITICAL FIX: Ensure the result is a valid image with proper clamping
            result = torch.clamp(result, 0.0, 1.0)
            
            # Check for NaN or Inf values in the result
            if torch.isnan(result).any() or torch.isinf(result).any():
                logging.warning("NaN or Inf values in result image. Replacing with fallback.")
                result = torch.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
                result = torch.clamp(result, 0.0, 1.0)
        except Exception as e:
            logging.error(f"Error during decoding: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Fall back to original stylized image
            result = stylized
        
        logging.info("CAV application complete")
        return result


    def visualize_pca_components(self, cav_dict, content_img, output_dir, image_size=256, num_components=5, scale=3.0):
        """Visualize top PCA components by applying them to content image."""
        if isinstance(content_img, str):
            c_tensor = open_image(content_img, size=image_size).unsqueeze(0).to(self.device, dtype=self.dtype)
        else:
            c_tensor = content_img.to(self.device, dtype=self.dtype)
        
        # Extract LL features and skips
        content_ll, skips = self.encode_to_ll(c_tensor)
        
        # Get the PCA components back to image space
        pca_components = cav_dict['pca'].components_
        
        # Limit to top components
        num_components = min(num_components, len(pca_components))
        
        results = [c_tensor]  # Original image
        
        for i in range(num_components):
            # Get component direction and reshape
            comp = torch.tensor(pca_components[i], dtype=self.dtype, device=self.device)
            comp = comp.reshape(1, *content_ll.shape[1:])
            
            # Normalize component
            norm_ratio = content_ll.norm() / comp.norm()
            scaled_comp = comp * norm_ratio * 0.01 * scale
            
            # Apply positive direction
            pos_ll = content_ll + scaled_comp
            pos_result = self.decode_from_ll(pos_ll, skips)
            
            # Apply negative direction
            neg_ll = content_ll - scaled_comp
            neg_result = self.decode_from_ll(neg_ll, skips)
            
            results.extend([pos_result, neg_result])
            
            # Save individual images
            vutils.save_image(pos_result, 
                             os.path.join(output_dir, f"pca_comp{i}_pos.png"))
            vutils.save_image(neg_result, 
                             os.path.join(output_dir, f"pca_comp{i}_neg.png"))
        
        # Make a grid of the images
        grid = vutils.make_grid(torch.cat(results, dim=0), nrow=3)
        grid_path = os.path.join(output_dir, "pca_components_visualization.png")
        vutils.save_image(grid, grid_path)
        logging.info(f"Saved PCA components visualization to {grid_path}")

    def learn_style_aware_direction(self, content_img, positive_styles, negative_styles, 
                              image_size=256, stylization_method="wct"):
        """Learn CAV direction based on applying multiple styles to one content image."""
        logging.info(f"Learning style-aware CAV using {len(positive_styles)} positive and {len(negative_styles)} negative styles")
        
        positive_features = []
        negative_features = []
        
        # Load content image once
        if isinstance(content_img, str):
            content_tensor = open_image(content_img, size=image_size).unsqueeze(0).to(self.device)
        else:
            content_tensor = content_img.to(self.device)
            
        # Default content segment mask
        content_segment = torch.ones(1, image_size, image_size).to(self.device)
        
        # Process positive styles
        for style_path in tqdm(positive_styles, desc="Processing positive styles"):
            try:
                # Load style image
                style_tensor = open_image(style_path, size=image_size).unsqueeze(0).to(self.device)
                style_segment = torch.ones(1, image_size, image_size).to(self.device)
                
                # Style transfer
                if stylization_method == "wct":
                    try:
                        # Try to load actual style segment if available
                        if 'positive' in style_path:
                            style_segment_path = style_path.replace('/positive/', '/positive_masks/').replace('\\positive\\', '\\positive_masks\\')
                            if os.path.exists(style_segment_path):
                                style_segment = load_segment(style_segment_path, image_size)
                        
                        stylized = self.wct2.transfer(content_tensor, style_tensor, 
                                            content_segment=content_segment,
                                            style_segment=style_segment)
                    except Exception as e:
                        logging.error(f"WCT transfer failed for {style_path}: {e}")
                        # Fallback to AdaIN
                        stylization_method = "adain"
                
                if stylization_method == "adain":
                    # AdaIN transfer
                    content_feat, skips = self.encode_to_ll(content_tensor)
                    style_feat, _ = self.encode_to_ll(style_tensor)
                    
                    content_mean = torch.mean(content_feat, dim=[2, 3], keepdim=True)
                    content_std = torch.std(content_feat, dim=[2, 3], keepdim=True) + 1e-5
                    style_mean = torch.mean(style_feat, dim=[2, 3], keepdim=True)
                    style_std = torch.std(style_feat, dim=[2, 3], keepdim=True) + 1e-5
                    
                    normalized = (content_feat - content_mean) / content_std
                    adapted = normalized * style_std + style_mean
                    combined = 1.0 * adapted + 0.0 * content_feat
                    
                    stylized = self.decode_from_ll(combined, skips)
                
                # Extract features
                feat, _ = self.encode_to_ll(stylized)
                
                # Normalize features (channel-wise)
                mean = torch.mean(feat, dim=[2, 3], keepdim=True)
                std = torch.std(feat, dim=[2, 3], keepdim=True) + 1e-5
                norm = (feat - mean) / std
                
                # Flatten and store
                flat = norm.view(norm.size(0), -1).cpu().numpy()
                positive_features.append(flat)
                
                # Save stylized image for reference
                output_dir = "./style_aware_stylized"
                os.makedirs(output_dir, exist_ok=True)
                style_name = os.path.splitext(os.path.basename(style_path))[0]
                
                vutils.save_image(stylized, os.path.join(output_dir, f"positive_{style_name}.png"))
                
            except Exception as e:
                logging.error(f"Error processing positive style {style_path}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Process negative styles using the same process
        for style_path in tqdm(negative_styles, desc="Processing negative styles"):
            try:
                # Load style image
                style_tensor = open_image(style_path, size=image_size).unsqueeze(0).to(self.device)
                style_segment = torch.ones(1, image_size, image_size).to(self.device)
                
                # Style transfer
                if stylization_method == "wct":
                    try:
                        # Try to load actual style segment if available
                        if 'negative' in style_path:
                            style_segment_path = style_path.replace('/negative/', '/negative_masks/').replace('\\negative\\', '\\negative_masks\\')
                            if os.path.exists(style_segment_path):
                                style_segment = load_segment(style_segment_path, image_size)
                        
                        stylized = self.wct2.transfer(content_tensor, style_tensor, 
                                            content_segment=content_segment,
                                            style_segment=style_segment)
                    except Exception as e:
                        logging.error(f"WCT transfer failed for {style_path}: {e}")
                        # Fallback to AdaIN
                        stylization_method = "adain"
                
                if stylization_method == "adain":
                    # AdaIN transfer
                    content_feat, skips = self.encode_to_ll(content_tensor)
                    style_feat, _ = self.encode_to_ll(style_tensor)
                    
                    content_mean = torch.mean(content_feat, dim=[2, 3], keepdim=True)
                    content_std = torch.std(content_feat, dim=[2, 3], keepdim=True) + 1e-5
                    style_mean = torch.mean(style_feat, dim=[2, 3], keepdim=True)
                    style_std = torch.std(style_feat, dim=[2, 3], keepdim=True) + 1e-5
                    
                    normalized = (content_feat - content_mean) / content_std
                    adapted = normalized * style_std + style_mean
                    combined = 1.0 * adapted + 0.0 * content_feat
                    
                    stylized = self.decode_from_ll(combined, skips)
                
                # Extract features
                feat, _ = self.encode_to_ll(stylized)
                
                # Normalize features
                mean = torch.mean(feat, dim=[2, 3], keepdim=True)
                std = torch.std(feat, dim=[2, 3], keepdim=True) + 1e-5
                norm = (feat - mean) / std
                
                # Flatten and store
                flat = norm.view(norm.size(0), -1).cpu().numpy()
                negative_features.append(flat)
                
                # Save stylized image for reference
                output_dir = "./style_aware_stylized"
                os.makedirs(output_dir, exist_ok=True)
                style_name = os.path.splitext(os.path.basename(style_path))[0]
                
                vutils.save_image(stylized, os.path.join(output_dir, f"negative_{style_name}.png"))
                
            except Exception as e:
                logging.error(f"Error processing negative style {style_path}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Combine features
        if not positive_features or not negative_features:
            raise ValueError("Failed to extract features from stylized images")
        
        pos_features = np.concatenate(positive_features, axis=0)
        neg_features = np.concatenate(negative_features, axis=0)
        
        # Log feature statistics
        logging.info(f"Positive features: {pos_features.shape}, mean={np.mean(pos_features):.4f}, std={np.std(pos_features):.4f}")
        logging.info(f"Negative features: {neg_features.shape}, mean={np.mean(neg_features):.4f}, std={np.std(neg_features):.4f}")
        
        # Compute the mean direction
        direction = torch.tensor(np.mean(pos_features - neg_features, axis=0), 
                            dtype=self.dtype, device=self.device)
        
        # Reshape to match feature dimensions
        feat_shape = self.encode_to_ll(content_tensor)[0].shape
        direction = direction.reshape(1, *feat_shape[1:])
        
        # Normalize direction to unit length
        direction_norm = direction.norm()
        logging.info(f"Direction norm before normalization: {direction_norm:.4f}")
        
        if direction_norm > 0:
            direction = direction / direction_norm
        
        logging.info(f"Created style-aware CAV direction with shape {direction.shape}")
        
        return {
            'direction': direction,
            'feature_shape': feat_shape,
        }

def verify_cav(controller, cav_dict, content_img, style_img, output_dir, image_size=256):
    """Simple verification of CAV by visualizing extreme values."""
    logging.info("Verifying CAV with extreme values...")
    
    # Apply with extreme negative and positive values
    strengths = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    results = []
    
    for strength in strengths:
        try:
            result = controller.apply_cav(
                content_img, style_img, cav_dict, 
                strength=strength, 
                image_size=image_size,
                stylization_method="adain"  # Try AdaIN for verification
            )
            results.append(result)
            
            # Save individual result
            vutils.save_image(result, os.path.join(output_dir, f"verify_cav_s{strength}.png"))
        except Exception as e:
            logging.error(f"Error in verification with strength {strength}: {e}")
    
    # Save grid of results
    if results:
        grid = vutils.make_grid(torch.cat(results, dim=0), nrow=len(results))
        grid_path = os.path.join(output_dir, "cav_verification_grid.png")
        vutils.save_image(grid, grid_path)
        logging.info(f"Saved verification grid to {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="PCA-based CAV approach for WCT2")
    parser.add_argument("--content_dir", type=str, default="./examples/content")
    parser.add_argument("--content_image", type=str, help="Specific content image (overrides content_dir)")
    parser.add_argument("--target_style_class", type=str, default="./positive")
    parser.add_argument("--other_style_class", type=str, default="./negative")
    parser.add_argument("--content_segment", type=str, default=None, help="Directory containing content segmentation masks")
    parser.add_argument("--target_style_segment", type=str, default="./positive_masks", help="Directory containing target style segmentation masks")
    parser.add_argument("--other_style_segment", type=str, default="./negative_masks", help="Directory containing other style segmentation masks")
    parser.add_argument("--model_path", type=str, default="./model_checkpoints")
    parser.add_argument("--option_unpool", type=str, default="cat5", choices=["sum", "cat5"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./pca_cav_outputs")
    parser.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4],
                      help="Encoder level to extract features from")
    parser.add_argument("--num_images", type=int, default=15,
                      help="Number of images to use from each class")
    parser.add_argument("--strengths", type=float, nargs="+", 
                    default=[-30,-20,-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
                    help="CAV strengths to apply")
    parser.add_argument("--transfer_at", type=str, nargs="+", 
                      default=["encoder", "decoder", "skip"],
                      help="Which components to use in WCT2")
    parser.add_argument("--save_cav", type=str, help="Path to save CAV")
    parser.add_argument("--load_cav", type=str, help="Path to load CAV")
    parser.add_argument("--n_components", type=int, default=30,
                      help="Number of PCA components to keep")
    parser.add_argument("--visualize_pca", action="store_true", 
                      help="Visualize PCA components")
    parser.add_argument("--stylization_method", type=str, default="wct", 
                      choices=["wct", "adain"], 
                      help="Method for stylization")
    parser.add_argument("--verify_cav", action="store_true",
                      help="Generate verification images of CAV with extreme values")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Check for CUDA
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        logging.warning(f"CUDA not available, using {device} instead")
    
    # Initialize WCT2 model
    logging.info(f"Initializing WCT2 model with transfer_at={args.transfer_at}")
    wct2 = WCT2(
        model_path=args.model_path,
        transfer_at=args.transfer_at,
        option_unpool=args.option_unpool,
        device=device,
        verbose=True
    )
    
    # Initialize PCA-based CAV controller
    logging.info(f"Initializing PCA-based CAV controller for level {args.level} with {args.n_components} components")
    controller = PCABasedCAVController(wct2, level=args.level, device=device, n_components=args.n_components)
    
    # Helper function to list images in a directory
    def list_images_in_dir(directory, max_images=None):
        if not os.path.exists(directory):
            logging.warning(f"Directory not found: {directory}")
            return []
            
        images = []
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(directory, filename))
                if max_images and len(images) >= max_images:
                    break
        return images
    
    # Load content images
    if args.content_image:
        content_images = [args.content_image]
    else:
        content_images = list_images_in_dir(args.content_dir)
    
    if not content_images:
        logging.error("No content images found")
        return
    logging.info(f"Found {len(content_images)} content images")
    
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
    
    # Check for segmentation masks
    if args.target_style_segment:
        logging.info(f"Using target style segmentation from {args.target_style_segment}")
    if args.other_style_segment:
        logging.info(f"Using other style segmentation from {args.other_style_segment}")
    
    # Either load or learn CAV
    if args.load_cav:
        logging.info(f"Loading CAV from {args.load_cav}")
        try:
            cav_dict = torch.load(args.load_cav, map_location=device)
        except Exception as e:
            logging.error(f"Error loading CAV: {e}")
            return
    else:
        logging.info("Learning PCA-based CAV")
        try:
            cav_dict = controller.learn_class_direction(
                target_styles, 
                other_styles,
                image_size=args.image_size
            )
            
            if args.save_cav:
                torch.save(cav_dict, args.save_cav)
                logging.info(f"Saved CAV to {args.save_cav}")
        except Exception as e:
            logging.error(f"Error learning CAV: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Visualize PCA components if requested
    if args.visualize_pca:
        logging.info("Visualizing PCA components")
        controller.visualize_pca_components(
            cav_dict,
            content_images[0],
            args.output_dir,
            image_size=args.image_size
        )
    
    # # Verify CAV with extreme values if requested
    # if args.verify_cav:
    #     logging.info("Verifying CAV with extreme values")
    #     verify_cav(
    #         controller,
    #         cav_dict,
    #         content_images[0],
    #         target_styles[0],
    #         args.output_dir,
    #         image_size=args.image_size
    #     )
    
    # # Apply CAV at different strengths
    # for content_idx, content_path in enumerate(content_images):
    #     content_name = os.path.splitext(os.path.basename(content_path))[0]
        
    #     # Use first target style as reference
    #     style_path = target_styles[0]
    #     style_name = os.path.splitext(os.path.basename(style_path))[0]
        
    #     logging.info(f"Processing content {content_name} with style {style_name}")
        
    #     # Apply CAV at different strengths
    #     results = []
    #     for strength in args.strengths:
    #         try:
    #             logging.info(f"Applying CAV with strength {strength}")
    #             result = controller.apply_cav(
    #                 content_path,
    #                 style_path,
    #                 cav_dict,
    #                 strength=strength,
    #                 image_size=args.image_size,
    #                 stylization_method=args.stylization_method
    #             )
                
    #             # Save individual result
    #             save_path = os.path.join(
    #                 args.output_dir, 
    #                 f"{content_name}_{style_name}_s{strength:.2f}.png"
    #             )
    #             vutils.save_image(result, save_path)
    #             logging.info(f"Saved to {save_path}")
                
    #             results.append(result)
    #         except Exception as e:
    #             logging.error(f"Error applying CAV with strength {strength}: {e}")
    #             import traceback
    #             traceback.print_exc()
        
    #     # Create a grid of all results
    #     if results:
    #         grid = vutils.make_grid(torch.cat(results, dim=0), nrow=len(results))
    #         grid_path = os.path.join(args.output_dir, f"{content_name}_{style_name}_grid.png")
    #         vutils.save_image(grid, grid_path)
    #         logging.info(f"Saved grid to {grid_path}")

    
    
# Add this at the end of the main() function, replacing or adding to the existing style-aware CAV section

    content_img = content_images[0]  # Use first content image

    logging.info("Learning style-aware CAV")
    try:
        style_aware_cav_dict = controller.learn_style_aware_direction(
            content_img,              # Use a single content image
            target_styles,            # Use all positive styles
            other_styles,             # Use all negative styles
            image_size=args.image_size,
            stylization_method=args.stylization_method
        )

        # Create subdirectory for style-aware CAV results
        style_aware_dir = os.path.join(args.output_dir, "style_aware_cav")
        os.makedirs(style_aware_dir, exist_ok=True)
        
        # Apply CAV at different strengths
        content_name = os.path.splitext(os.path.basename(content_img))[0]
        style_name = "style_aware"  # Generic name since we're using multiple styles
        
        results = []
        # Apply the CAV to the content image with different strengths
        for strength in args.strengths:
            try:
                logging.info(f"Applying style-aware CAV with strength {strength}")
                result = controller.apply_cav(
                    content_img,
                    target_styles[0],  # Use first positive style as reference
                    style_aware_cav_dict,
                    strength=strength,
                    image_size=args.image_size,
                    stylization_method=args.stylization_method
                )

                # Save individual result
                save_path = os.path.join(
                    style_aware_dir, 
                    f"{content_name}_{style_name}_s{strength:.2f}.png"
                )
                vutils.save_image(result, save_path)
                logging.info(f"Saved to {save_path}")
                
                results.append(result)
            except Exception as e:
                logging.error(f"Error applying style-aware CAV with strength {strength}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Create a grid of all results
        if results:
            grid = vutils.make_grid(torch.cat(results, dim=0), nrow=len(results))
            grid_path = os.path.join(style_aware_dir, f"{content_name}_{style_name}_grid.png")
            vutils.save_image(grid, grid_path)
            logging.info(f"Saved grid to {grid_path}")
            
        # Save the CAV if requested
        if args.save_cav:
            style_aware_cav_path = args.save_cav.replace('.pt', '_style_aware.pt')
            torch.save(style_aware_cav_dict, style_aware_cav_path)
            logging.info(f"Saved style-aware CAV to {style_aware_cav_path}")
    except Exception as e:
        logging.error(f"Error in style-aware CAV process: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()