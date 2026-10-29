"""Creates the seeded random test checkpoint (model/make_random_checkpoint.py) on a fresh clone, so the integer model
and kernel tests run without NVIDIA's gated weights.

Author: Nasir Ali, C-DAC Noida.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs" / "models" / "random_model1_fp16.safetensors"

if not CKPT.exists():
    subprocess.run([sys.executable, str(ROOT / "model" / "make_random_checkpoint.py")], check=True, cwd=ROOT)
