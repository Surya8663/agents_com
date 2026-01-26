# app/compat/numpy_fix.py (IMPROVED)
"""
Robust NumPy compatibility fix for all versions.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

print("🔧 Applying robust NumPy compatibility fix...")

try:
    import numpy as np
    
    print(f"✅ NumPy {np.__version__} detected")
    
    # Apply compatibility fixes for NumPy 2.x
    if np.__version__.startswith('2.'):
        print("⚠️  NumPy 2.x detected - applying compatibility layer")
        
        # Create sctypes for older library compatibility
        if not hasattr(np, 'sctypes'):
            np.sctypes = {
                'int': [np.int8, np.int16, np.int32, np.int64],
                'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                'float': [np.float16, np.float32, np.float64],
                'complex': [np.complex64, np.complex128],
                'others': [bool, object, bytes, str, np.void]
            }
        
        # Add deprecated aliases
        deprecated_aliases = {
            'bool': bool,
            'int': int,
            'float': float,
            'complex': complex,
            'object': object,
            'str': str,
            'bytes': bytes,
        }
        
        for alias, value in deprecated_aliases.items():
            if not hasattr(np, alias):
                setattr(np, alias, value)
        
        print("✅ Applied comprehensive NumPy 2.x compatibility")
    
    # Also fix for PyTorch compatibility
    try:
        # Monkey patch torch to avoid numpy warnings
        import torch
        
        original_warn = torch._tensor_str._tensor_str._warn_numpy_bug
        
        def patched_warn(*args, **kwargs):
            pass  # Suppress the warning
        
        torch._tensor_str._tensor_str._warn_numpy_bug = patched_warn
        print("✅ Suppressed PyTorch numpy warnings")
        
    except Exception as e:
        print(f"⚠️  PyTorch warning suppression: {e}")
    
    print("✅ NumPy compatibility setup complete")
    
except Exception as e:
    print(f"❌ NumPy compatibility error: {e}")
    print("⚠️  Continuing anyway - some features may not work")