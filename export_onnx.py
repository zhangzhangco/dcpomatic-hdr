
import torch
import argparse
import os
from infer_one import load_model, build_model

def export(pth_path, onnx_path):
    # 1. Load Model
    # Note: load_model internally calls .eval() and handles device
    # But for export we usually want CPU to be safe
    model = build_model()
    
    # Load weights manually to avoid 'cuda' dependency if on pure cpu machine
    # Copied logic from infer_one.py but forced to map_location='cpu'
    print(f"Loading checkpoint from: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()

    # 2. Creates dummy input
    # Shape: (1, 3, 256, 256) - size doesn't matter much if dynamic
    dummy_input = torch.randn(1, 3, 256, 256)

    # 3. Export
    # Input names: ['input']
    # Output names: ['gain_map', 'hdr_pred']
    # Dynamic axes: Allow variable H, W
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11, # Safe compatible version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['gain_map', 'hdr_pred'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'gain_map': {0: 'batch_size', 2: 'height', 3: 'width'},
            'hdr_pred': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print("Export success.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--onnx", required=True, help="Path to output .onnx")
    args = ap.parse_args()
    
    export(args.pth, args.onnx)
