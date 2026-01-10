import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# =========================================================
# 1. Colorspace Constants (from src/utils/colorspace.py)
# =========================================================

# Standard DCI-P3 RGB to XYZ matrix (Normalized):
_M_P3_TO_XYZ = np.array([
    [0.48657, 0.26567, 0.19822],
    [0.22897, 0.69174, 0.07929],
    [0.00000, 0.04511, 1.04394]
], dtype=np.float32)

# Inverse: XYZ to DCI-P3 RGB
_M_XYZ_TO_P3 = np.linalg.inv(_M_P3_TO_XYZ).astype(np.float32)

def xyz_to_p3_linear_numpy(xyz_img):
    """
    Args: xyz_img (H, W, 3)
    Returns: p3_img (H, W, 3)
    """
    # rgb = xyz @ matrix.T
    return np.dot(xyz_img, _M_XYZ_TO_P3.T)

def p3_to_xyz_linear_numpy(p3_img):
    """
    Args: p3_img (H, W, 3)
    Returns: xyz_img (H, W, 3)
    """
    return np.dot(p3_img, _M_P3_TO_XYZ.T)


# =========================================================
# 2. Model Architecture (from src/model/network.py)
# =========================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MyHDR_ITM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(MyHDR_ITM_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  # 3 for RGB Gain
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

        # Initialize Output Bias
        if hasattr(self.outc.conv, 'bias') and self.outc.conv.bias is not None:
            nn.init.constant_(self.outc.conv.bias, -2.0)

    def forward(self, sdr_input):
        # Encoder
        x1 = self.inc(sdr_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        # Gain >= 1.0 (Physical Constraint)
        min_gain = 1.0
        max_gain = 20.0 
        sigmoid_out = torch.sigmoid(logits)
        raw_gain = min_gain + (max_gain - min_gain) * sigmoid_out
        
        # Luma-Aware Structural Mask
        luma = 0.2095 * sdr_input[:, 0:1, :, :] + \
               0.7216 * sdr_input[:, 1:2, :, :] + \
               0.0689 * sdr_input[:, 2:3, :, :]
               
        low_thresh = 0.01 
        high_thresh = 0.10
        mask = torch.clamp((luma - low_thresh) / (high_thresh - low_thresh), 0.0, 1.0)
        mask = mask * mask * (3 - 2 * mask)
        
        gain_delta = raw_gain - 1.0
        gain_map = 1.0 + gain_delta * mask
        
        hdr_pred = sdr_input * gain_map
        return gain_map, hdr_pred

# =========================================================
# 3. Inference Logic
# =========================================================

def build_model():
    return MyHDR_ITM_UNet(n_channels=3, n_classes=3)

def load_model(pth_path, device="cuda"):
    model = build_model()
    print(f"Loading checkpoint from: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    # Handle state_dict or raw checkpoint
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    
    # Strip 'module.' prefix if present
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
        
    model.load_state_dict(new_state, strict=True)
    model.eval().to(device)
    return model

@torch.no_grad()
def infer(model, in_npy, out_npy, device="cuda"):
    # 1. Load Data
    # input expected: HWC, XYZ (relative 0-1)
    x = np.load(in_npy).astype(np.float32) 
    
    # 2. Preprocess
    # A. XYZ Relative -> P3 Relative
    x_p3 = xyz_to_p3_linear_numpy(x)
    
    # B. P3 Relative -> P3 Nits (0-48 nits, using 48.0 scalar)
    # Note: Training assumed 1.0 = 48 nits
    x_nits = x_p3 * 48.0
    
    # C. To Tensor (1, 3, H, W)
    x_t = torch.from_numpy(x_nits).permute(2,0,1).unsqueeze(0).to(device)

    # 3. Inference
    # Model returns (gain_map, hdr_pred)
    _, y_t = model(x_t)
    
    # 4. Postprocess
    # y_t is P3 Nits (Absolute)
    y_p3_nits = y_t.squeeze(0).permute(1,2,0).float().cpu().numpy()
    
    # A. P3 Nits -> XYZ Nits (Absolute)
    y_xyz_nits = p3_to_xyz_linear_numpy(y_p3_nits)
    
    # B. Clamp / Container Formatting
    # Min=0.005 (Cinema HDR Black), Max=300 (Target Peak) unless specified differently
    y_final = np.clip(y_xyz_nits, 0.005, 300.0)

    # 5. Save
    print(f"Saving to {out_npy}...")
    np.save(out_npy, y_final.astype(np.float32))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--in_npy", required=True, help="Input NPY (XYZ Relative)")
    ap.add_argument("--out_npy", required=True, help="Output NPY (XYZ Absolute)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.pth, args.device)
    infer(model, args.in_npy, args.out_npy, args.device)
