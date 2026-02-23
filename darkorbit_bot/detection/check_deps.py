
print("Step 1: Start")
import os
print("Step 2: OS imported")
import json
print("Step 3: JSON imported")
try:
    import torch
    print(f"Step 4: Torch imported (Version: {torch.__version__}, CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"Step 4 ERROR: {e}")
except Exception as e:
    print(f"Step 4 CRITICAL: {e}")

try:
    from PIL import Image
    print("Step 5: PIL imported")
except Exception as e:
    print(f"Step 5 ERROR: {e}")

try:
    print("Step 6: Importing Transformers...")
    from transformers import AutoProcessor, AutoModelForCausalLM
    import transformers
    print(f"Step 6: Transformers imported (Version: {transformers.__version__})")
except Exception as e:
    print(f"Step 6 ERROR: {e}")

print("Step 7: All Good")
