import os
import sys
import pickle
import zipfile

def inspect_with_torch(file_path):
    try:
        import torch
        print("PyTorch is installed. Loading model...")
        # Load weights on CPU
        data = torch.load(file_path, map_location='cpu')
        print(f"Loaded successfully with torch. Type: {type(data)}")
        
        if isinstance(data, dict):
            print("\n--- Keys in the dictionary ---")
            for k in list(data.keys())[:20]:
                val = data[k]
                if isinstance(val, dict):
                    print(f"  {k}: dict with keys {list(val.keys())[:10]} (total {len(val)})")
                elif hasattr(val, 'shape'):
                    print(f"  {k}: Tensor of shape {list(val.shape)}")
                else:
                    print(f"  {k}: {type(val)} (length/value if small: {str(val)[:100]})")
            if len(data.keys()) > 20:
                print(f"  ... and {len(data.keys()) - 20} more keys.")
            
            # Look for specific signatures
            # Check for RVC (Retrieval-based Voice Conversion) keys
            rvc_keys = ['weight', 'config', 'params', 'info']
            if any(k in data for k in ['params', 'config', 'info', 'sr', 'f0', 'version']):
                print("\n[Detection] Looks like a custom configuration or audio/TTS/RVC model.")
            
            # Check for weights/state_dict key
            for k in ['model', 'state_dict', 'generator']:
                if k in data and isinstance(data[k], dict):
                    print(f"\nFound nested state dict key: '{k}' with {len(data[k])} weight tensors.")
                    print("Sample layers:")
                    for l_key in list(data[k].keys())[:10]:
                        tens = data[k][l_key]
                        shape = list(tens.shape) if hasattr(tens, 'shape') else 'no shape'
                        print(f"  {l_key}: {shape}")
        else:
            print(f"Loaded object is not a dictionary. Type: {type(data)}")
            print(str(data)[:1000])
            
    except Exception as e:
        print(f"Failed to load using torch: {e}")

def inspect_without_torch(file_path):
    print("\nAttempting to inspect file structure without torch...")
    try:
        if zipfile.is_zipfile(file_path):
            print("File is a ZIP archive (standard for modern PyTorch / Safetensors formats).")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"Contains {len(file_list)} items inside the archive. First 20 items:")
                for name in file_list[:20]:
                    print(f"  {name}")
                if len(file_list) > 20:
                    print(f"  ... and {len(file_list) - 20} more files.")
                
                # Check for pickle files inside
                pkl_files = [f for f in file_list if f.endswith('.pkl') or 'data.pkl' in f or 'pkl' in f]
                if pkl_files:
                    print(f"\nFound pickle files inside zip: {pkl_files}")
        else:
            print("File is not a ZIP archive. It might be a legacy PyTorch file or raw pickle.")
            # Try to read raw pickle header
            with open(file_path, 'rb') as f:
                header = f.read(100)
                print(f"File header (first 100 bytes): {header}")
    except Exception as e:
        print(f"Failed to inspect zip structure: {e}")

if __name__ == '__main__':
    path = r"C:\Users\Osama\Downloads\y4k2_mimic.pth"
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)
        
    print(f"Inspecting file: {path}")
    print(f"File size: {os.path.getsize(path)} bytes")
    
    # Try importing torch first
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("PyTorch is not installed in the current environment.")
        
    if has_torch:
        inspect_with_torch(path)
    
    inspect_without_torch(path)
