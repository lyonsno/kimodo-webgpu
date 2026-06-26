#!/usr/bin/env python3
"""
Convert Kimodo safetensors checkpoint to flat binary format for WebGPU.

Usage:
    python tools/convert_weights.py [--model models/Kimodo-SOMA-RP-v1.1/model.safetensors] [--output public/kimodo.bin] [--dtype fp16]

Output format matches MoGE's flat binary:
    Header: magic "KIMD" + version + num_tensors + header_size
    Tensor table: name(64) + dtype(4) + ndim(4) + shape(16) + offset(4) + size(4) = 96 bytes each
    Weight data: packed fp16/fp32 tensors, 16-byte aligned
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"KIMD"
VERSION = 1
MAX_NAME_LEN = 64
MAX_DIMS = 4


def load_safetensors(path: str) -> dict:
    """Load safetensors file and return state dict."""
    from safetensors.torch import load_file
    return load_file(path)


def convert(state_dict: dict, output_path: str, dtype: str = "fp16"):
    """Convert state dict to flat binary for WebGPU."""
    tensor_entries = []
    weight_data = bytearray()

    dtype_code = 0 if dtype == "fp32" else 1
    np_dtype = np.float32 if dtype == "fp32" else np.float16

    # Strip the "denoiser.backbone." prefix for cleaner names in WebGPU
    PREFIX = "denoiser.backbone."

    # Kimodo uses fused QKV: self_attn.in_proj_weight [3072, 1024]
    # and standard linear weights [out, in].
    # For WebGPU linear.wgsl we need [in, out] (row-major, weight transposed).
    LINEAR_WEIGHT_SUFFIXES = (
        '.in_proj_weight',    # fused QKV
        '.out_proj.weight',   # attention output
        '.linear1.weight',    # FFN up
        '.linear2.weight',    # FFN down
        '.embed_text.weight',
        '.input_linear.weight',
        '.output_linear.weight',
        '.linear_first_heading_angle.weight',
        'time_embed.0.weight',
        'time_embed.2.weight',
    )

    for orig_name in sorted(state_dict.keys()):
        tensor = state_dict[orig_name]
        arr = tensor.detach().float().numpy()

        # Clean name
        name = orig_name
        if name.startswith(PREFIX):
            name = name[len(PREFIX):]

        # Transpose 2D linear weights: PyTorch [out, in] -> shader [in, out]
        is_linear = any(orig_name.endswith(s) for s in LINEAR_WEIGHT_SUFFIXES)
        if is_linear and arr.ndim == 2:
            arr = arr.T.copy()

        arr = arr.astype(np_dtype)
        data = arr.tobytes()

        shape = list(arr.shape)
        if len(shape) > MAX_DIMS:
            shape = [int(np.prod(shape[:-3]))] + list(shape[-3:])
            arr = arr.reshape(shape)
            data = arr.tobytes()

        offset = len(weight_data)
        size = len(data)
        weight_data.extend(data)

        # Pad to 16-byte alignment
        pad = (16 - (len(weight_data) % 16)) % 16
        weight_data.extend(b"\x00" * pad)

        tensor_entries.append({
            "name": name,
            "dtype": dtype_code,
            "shape": shape,
            "offset": offset,
            "size": size,
        })

    # Build header
    num_tensors = len(tensor_entries)
    ENTRY_SIZE = 96
    header_size = 16 + num_tensors * ENTRY_SIZE

    for entry in tensor_entries:
        entry["offset"] += header_size

    # Write binary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", header_size))

        for entry in tensor_entries:
            name_bytes = entry["name"].encode("ascii")[:MAX_NAME_LEN]
            f.write(name_bytes.ljust(MAX_NAME_LEN, b"\x00"))
            f.write(struct.pack("<I", entry["dtype"]))
            ndim = len(entry["shape"])
            f.write(struct.pack("<I", ndim))
            shape_padded = entry["shape"] + [0] * (MAX_DIMS - ndim)
            for s in shape_padded:
                f.write(struct.pack("<I", s))
            f.write(struct.pack("<I", entry["offset"]))
            f.write(struct.pack("<I", entry["size"]))

        f.write(weight_data)

    total_mb = len(weight_data) / (1024 * 1024)
    print(f"Weights written to {output_path}")
    print(f"  Tensors: {num_tensors}")
    print(f"  Size: {total_mb:.1f} MB ({dtype})")
    print(f"  Header: {header_size} bytes")

    # Write model config sidecar
    config = {
        "model": "Kimodo-SOMA-RP-v1.1",
        "architecture": "TransformerEncoder",
        "hidden_dim": 1024,
        "num_heads": 16,  # 1024/64 = 16 heads
        "head_dim": 64,
        "ffn_dim": 2048,
        "num_layers": 16,
        "body_input_dim": 737,
        "body_output_dim": 364,
        "root_input_dim": 738,
        "root_output_dim": 5,
        "text_dim": 4096,
        "max_seq_len": 5000,
        "parents": [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75],
        "fps": 30,
        "dtype": dtype,
    }
    config_path = output_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    print(f"\nTensor summary:")
    for entry in tensor_entries:
        shape_str = "x".join(str(s) for s in entry["shape"])
        print(f"  {entry['name']:60s} {shape_str:>20s}  {entry['size'] / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Convert Kimodo weights for WebGPU")
    parser.add_argument("--model", default="models/Kimodo-SOMA-RP-v1.1/model.safetensors",
                        help="Path to safetensors checkpoint")
    parser.add_argument("--output", default="public/kimodo.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16"],
                        help="Weight data type (fp16 recommended for browser)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list tensor names and shapes")
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    state_dict = load_safetensors(args.model)

    if args.list_only:
        total = 0
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            total += t.numel()
            print(f"  {name:60s} {str(list(t.shape)):>20s}  {t.numel():>10d}")
        print(f"\nTotal: {total:,} params ({total*4/1e6:.1f} MB fp32, {total*2/1e6:.1f} MB fp16)")
        return

    convert(state_dict, args.output, args.dtype)


if __name__ == "__main__":
    main()
