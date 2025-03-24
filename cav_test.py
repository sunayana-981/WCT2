# Create a diagnostic file called diagnostic_cav.py with this simplified approach:

import os
import torch
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as transforms
from transfer1 import WCT2  # Your working WCT2 implementation

# Simple image loading
def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    return transform(Image.open(path).convert('RGB')).unsqueeze(0)

# Create output directory
os.makedirs('diagnostic_outputs', exist_ok=True)

# 1. Load a content image and a style image
content_path = "./examples/content/in00.png"  # Replace with your content image
style_path = "./examples/style/in00.png"
content_seg_path = "./examples/content_seg/in00.png"
style_seg_path = "./examples/style_seg/in00.png"
content = load_image(content_path, 256)
style = load_image(style_path, 256)

# 2. Initialize WCT2
wct2 = WCT2(
    model_path='./model_checkpoints',
    transfer_at=['encoder', 'decoder', 'skip'],
    option_unpool='cat5',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

device = wct2.device
content = content.to(device)
style = style.to(device)

# 3. First, test regular style transfer
print("Testing regular style transfer...")
with torch.no_grad():
    styled = wct2.transfer(content, style, content_seg, None, alpha=1.0)
    vutils.save_image(styled, 'diagnostic_outputs/1_regular_styled.png')
    print("Regular style transfer saved")

# 4. Now, manually encode content, manipulate features, and decode
print("Testing direct feature manipulation...")
with torch.no_grad():
    # Encode content to get features
    content_skips = {}
    feat = content
    for level in [1, 2, 3, 4]:  # Encode to level 4
        feat = wct2.encode(feat, content_skips, level)
        print(f"Level {level} content feature shape: {feat.shape}")
    
    # Save original features
    orig_feat = feat.clone()
    
    # Try extremely small manipulations
    strengths = [0.0, 0.00001, 0.0001, 0.001, 0.01]
    
    for strength in strengths:
        # Create a simple "dummy CAV" - just add a small amount to all values
        modified_feat = orig_feat + strength
        
        # Decode
        result = modified_feat
        for level in [4, 3, 2, 1]:  # Decode from level 4 down
            result = wct2.decode(result, content_skips, level)
            print(f"Level {level} decoded shape: {result.shape}")
        
        # Save result
        vutils.save_image(result.clamp(0, 1), f'diagnostic_outputs/2_direct_strength_{strength}.png')
        print(f"Saved direct manipulation with strength {strength}")

    # Try extremely small random noise as a CAV
    for strength in [0.0001, 0.001]:
        random_direction = torch.randn_like(orig_feat) * strength
        modified_feat = orig_feat + random_direction
        
        # Decode
        result = modified_feat
        for level in [4, 3, 2, 1]:
            result = wct2.decode(result, content_skips, level)
        
        # Save result
        vutils.save_image(result.clamp(0, 1), f'diagnostic_outputs/3_random_noise_{strength}.png')
        print(f"Saved random noise with strength {strength}")

print("Diagnostic tests complete. Check diagnostic_outputs/ directory.")