
import struct
import numpy as np
import sys
import os

def read_data(path):
    print(f"Reading {path}...")
    if path.endswith(".npy"):
        return np.load(path)
    
    # Legacy .bin support
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ZH01":
            print("Error: Invalid magic number or format")
            return None
        w, h, c = struct.unpack("iii", f.read(12))
        print(f"Header: {w}x{h}x{c}")
        raw = f.read()
        data = np.frombuffer(raw, dtype=np.float32).reshape(h, w, c)
    return data

def analyze_xyz(data):
    print("-" * 30)
    print("Stats (XYZ):")
    channels = ['X', 'Y', 'Z']
    for i in range(3):
        c_data = data[:, :, i]
        print(f"  {channels[i]}: Min={c_data.min():.6f}, Max={c_data.max():.6f}, Mean={c_data.mean():.6f}")

def save_preview(data, output_png):
    try:
        import cv2
    except ImportError:
        print("Opencv not installed, skipping preview generation")
        return

    # XYZ -> RGB (Rec.709 D65 approx for viewing)
    # Simple Matrix (Inverse of sRGB/709 -> XYZ)
    # This assumes 'data' is Absolute XYZ.
    
    # First: normalize brightness for viewing
    # If Max > 10.0, assume it's NITS (absolute).
    y_max = data[:,:,1].max()
    is_absolute = y_max > 2.0
    
    view_data = data.copy()
    
    if is_absolute:
        print(f"Detected Absolute Nits (Max={y_max:.2f}). Applying Simple Tone Mapping for Preview...")
        # Simple Reinhard-like or just Gamma compression
        # Normalize 300 nits -> 1.0 for viewing
        view_data = view_data / 300.0
    else:
        # Assume 0-1 relative
        pass
        
    # XYZ -> Linear RGB
    # Matrix: XYZ to sRGB(D65)
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ]).T
    
    # Flatten
    h, w, c = view_data.shape
    flat = view_data.reshape(-1, 3)
    rgb_lin = flat @ M
    rgb_lin = rgb_lin.reshape(h, w, 3)
    
    # Clip negatives
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    
    # Gamma Encoding for Display
    # Apply sRGB transfer function or simple gamma 2.2
    rgb_disp = np.power(rgb_lin, 1/2.2)
    
    # Exposure Compensation if too dark (often HDR linear looks dark)
    if is_absolute: 
        # Boost exposure slightly?
        # rgb_disp = rgb_disp * 1.2
        pass
        
    rgb_disp = np.clip(rgb_disp, 0.0, 1.0)
    
    img = (rgb_disp * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_png, img)
    print(f"Saved preview to {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_dump.py <file.npy>")
        sys.exit(1)
        
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    data = read_data(path)
    if data is not None:
        analyze_xyz(data)
        save_preview(data, "preview_out.png")

