import zipfile
import pickle
import sys

# Create a robust mock of the torch module and its tensor rebuilders
# so pickle can load the data structure without needing the full PyTorch package.
class MockTensor:
    def __init__(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        self.shape = size
        self.stride = stride
        self.storage = storage
    def __repr__(self):
        return f"Tensor(shape={list(self.shape)})"

class MockStorage:
    def __init__(self, *args, **kwargs):
        pass

class MockRebuilder:
    def _rebuild_tensor(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        return MockTensor(storage, storage_offset, size, stride, requires_grad, backward_hooks)
    
    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        return MockTensor(storage, storage_offset, size, stride, requires_grad, backward_hooks)

    def _rebuild_parameter(self, data, requires_grad, backward_hooks):
        return data

# Inject our mocks into the sys.modules before unpickling
import types
torch_module = types.ModuleType('torch')
torch_utils_module = types.ModuleType('torch._utils')
rebuilder = MockRebuilder()

torch_utils_module._rebuild_tensor = rebuilder._rebuild_tensor
torch_utils_module._rebuild_tensor_v2 = rebuilder._rebuild_tensor_v2
torch_utils_module._rebuild_parameter = rebuilder._rebuild_parameter

torch_module._utils = torch_utils_module
torch_module.HalfStorage = MockStorage
torch_module.FloatStorage = MockStorage
torch_module.DoubleStorage = MockStorage
torch_module.LongStorage = MockStorage
torch_module.IntStorage = MockStorage
torch_module.ShortStorage = MockStorage
torch_module.ByteStorage = MockStorage
torch_module.CharStorage = MockStorage
torch_module.BoolStorage = MockStorage

sys.modules['torch'] = torch_module
sys.modules['torch._utils'] = torch_utils_module

def inspect_pkl(file_path):
    print(f"Opening ZIP: {file_path}")
    with zipfile.ZipFile(file_path, 'r') as z:
        # Find the .pkl file
        pkl_path = None
        for name in z.namelist():
            if name.endswith('.pkl'):
                pkl_path = name
                break
                
        if not pkl_path:
            print("No .pkl file found in zip.")
            return
            
        print(f"Reading pickle entry: {pkl_path}")
        with z.open(pkl_path) as f:
            try:
                # Custom Unpickler to handle potential missing classes gracefully
                class CustomUnpickler(pickle.Unpickler):
                    def persistent_load(self, pid):
                        # PyTorch persistent IDs are usually tuples: ('storage', data_type, key, location, size)
                        # We return a dummy storage object to satisfy pickle
                        return MockStorage()

                    def find_class(self, module, name):
                        try:
                            return super().find_class(module, name)
                        except Exception:
                            # Return a placeholder class for any unknown module/class
                            class UnknownClass:
                                def __init__(self, *args, **kwargs):
                                    self._class_name = f"{module}.{name}"
                                    self._args = args
                                    self._kwargs = kwargs
                                def __repr__(self):
                                    return f"<{self._class_name}>"
                            return UnknownClass
                            
                data = CustomUnpickler(f).load()
                print(f"Successfully deserialized. Root object type: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"Found dict with {len(data)} keys.")
                    print("\n--- Keys and value representations ---")
                    for k, v in list(data.items())[:100]:
                        if isinstance(v, dict):
                            print(f"  {k}: dict with {len(v)} keys (keys: {list(v.keys())[:10]})")
                        else:
                            val_str = str(v)
                            if len(val_str) > 120:
                                val_str = val_str[:120] + "..."
                            print(f"  {k}: {type(v).__name__} -> {val_str}")
                    if len(data) > 100:
                        print(f"  ... and {len(data) - 100} more keys.")
                elif isinstance(data, list):
                    print(f"Found list of length {len(data)}. Elements:")
                    for i, item in enumerate(data[:10]):
                        print(f"  [{i}]: {item}")
                else:
                    print("Root object value representation:")
                    print(data)
                    
            except Exception as e:
                print(f"Error unpickling: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    import os
    possible_paths = [
        r"C:\Users\Osama\Downloads\chess_mimic_model.pth",
        r"C:\Users\Osama\Downloads\y4k2_mimic.pth"
    ]
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if path:
        inspect_pkl(path)
    else:
        print("Error: Model file not found in Downloads.")
