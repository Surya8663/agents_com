# app/compat/pytorch_full_fix.py
"""
Complete PyTorch 2.10+ compatibility fix for YOLO.
"""
import torch
import torch.serialization
import torch.nn as nn

print("🔧 Applying PyTorch 2.10+ compatibility fixes...")

# Add ALL necessary classes to safe globals
safe_classes = [
    # PyTorch core classes
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    torch.nn.modules.linear.Linear,
    
    # Common containers
    torch.nn.ModuleList,
    torch.nn.ModuleDict,
    
    # Torch utilities
    torch.Size,
    torch.Tensor,
    torch.device,
    torch.dtype,
]

# Try to add ultralytics classes
try:
    from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Detect
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
    from ultralytics.nn.modules.block import DFL
    
    safe_classes.extend([
        Conv, Bottleneck, C2f, SPPF, Detect,
        DetectionModel, SegmentationModel, ClassificationModel,
        DFL
    ])
    print("✅ Added ultralytics classes to safe globals")
except ImportError as e:
    print(f"⚠️  Could not import some ultralytics classes: {e}")

# Apply the fix
try:
    torch.serialization.add_safe_globals(safe_classes)
    print("✅ Applied comprehensive PyTorch 2.10 compatibility fix")
except Exception as e:
    print(f"⚠️  Compatibility fix application: {e}")

# Also set environment variable as fallback
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
print("✅ Set environment variable to allow loading")