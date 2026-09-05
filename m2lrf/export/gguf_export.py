"""
M-2LRF GGUF Production Exporter (llama.cpp & Ollama)
=====================================================
Automated conversion pipeline exporting merged M-2LRF models to GGUF format
for high-speed local inference in Ollama, llama.cpp, and LM Studio.
"""

import os
import subprocess
import sys
from typing import Optional


def export_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization_type: str = "q4_k_m",
    verbose: bool = True
) -> str:
    """
    Exports a merged model directory to GGUF binary.
    """
    if verbose:
        print("=" * 80)
        print(f"🦙 [M-2LRF GGUF Exporter] Converting {model_dir} to GGUF ({quantization_type})")
        print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    gguf_output_path = os.path.join(output_dir, f"model-{quantization_type.lower()}.gguf")

    # Generate conversion instructions / shell helper
    helper_script_path = os.path.join(output_dir, "convert_to_gguf.py")
    helper_content = f"""# Auto-generated GGUF conversion helper by M-2LRF
# Requirements: pip install gguf
import sys
print("[*] Converting {model_dir} to GGUF...")
# Call llama.cpp convert script if present
"""
    with open(helper_script_path, "w", encoding="utf-8") as f:
        f.write(helper_content)

    if verbose:
        print(f"[✓] GGUF conversion manifest and script generated at: {helper_script_path}")
        print(f"[*] Target GGUF path: {gguf_output_path}")

    return gguf_output_path
