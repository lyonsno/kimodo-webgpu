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
import os
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"KIMD"
VERSION = 1
MAX_NAME_LEN = 64
MAX_DIMS = 4


def load_safetensors(path: str) -> dict:
    """Load safetensors file and return state dict.

    Prefers the numpy loader so weight conversion needs only numpy+safetensors,
    not a full torch install. Falls back to the torch loader for checkpoints
    holding dtypes numpy cannot represent (e.g. bfloat16).
    """
    try:
        from safetensors.numpy import load_file
        return load_file(path)
    except Exception:
        from safetensors.torch import load_file
        return {k: v.numpy() for k, v in load_file(path).items()}


def build_config(dtype: str) -> dict:
    """The sidecar config for a generated binary.

    These are NOT inferred from the checkpoint — they were established by
    per-layer comparison against the PyTorch reference. An earlier version
    guessed 16x128 from hidden_dim/64; the real backbone is 8x128, post-norm,
    GELU. Guessing here silently corrupts attention head splitting and yields
    plausible-looking garbage motion, so do not "simplify" these into arithmetic.
    """
    return {
        "model": "Kimodo-SOMA-RP-v1.1",
        "architecture": "TransformerEncoder",
        "hidden_dim": 1024,
        "num_heads": 8,
        "head_dim": 128,
        "norm_first": False,
        "activation": "gelu",
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


def _compatible(have, want) -> bool:
    """Recursive compatibility check between an on-disk value and a generated one.

    Two rules that a bare `==` gets wrong:

    - `True == 1` in Python, at every nesting depth. A `parents` list containing
      booleans would compare equal to one containing ints, so booleans are held
      distinct from numbers explicitly.
    - `30 == 30.0`, but JSON round-trips can legitimately produce either and the
      browser consumes both as `Number`. Integral floats are therefore accepted
      as equal to their integer counterparts.
    """
    if isinstance(have, bool) or isinstance(want, bool):
        return isinstance(have, bool) and isinstance(want, bool) and have == want
    if isinstance(have, (int, float)) and isinstance(want, (int, float)):
        return float(have) == float(want)
    if isinstance(have, list) and isinstance(want, list):
        return len(have) == len(want) and all(_compatible(h, w) for h, w in zip(have, want))
    if isinstance(have, dict) and isinstance(want, dict):
        return have.keys() == want.keys() and all(_compatible(have[k], want[k]) for k in want)
    return type(have) is type(want) and have == want


def check_existing_config(config_path: Path, config: dict):
    """Fail non-zero unless an existing sidecar is compatible with `config`.

    Every generated key must be present and compatible, including `dtype`.
    Comparing only shared keys meant an empty `{}` was reported as "matching".

    Extra keys on disk are deliberately ALLOWED: additive fields from a newer
    writer should not block a conversion. This is a compatibility policy, not
    exact key-set equality.
    """
    if not config_path.exists():
        return

    try:
        existing = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(
            f"Existing config {config_path} is unreadable: {e}\n"
            "Refusing to write a binary that would be paired with a broken "
            "sidecar. Delete or repair the file to regenerate it."
        )
    if not isinstance(existing, dict):
        raise SystemExit(
            f"Existing config {config_path} is not a JSON object.\n"
            "Refusing to write a binary paired with an unusable sidecar."
        )

    drift = {}
    for key, want in config.items():
        if key not in existing:
            drift[key] = ("<missing>", want)
        elif not _compatible(existing[key], want):
            drift[key] = (existing[key], want)

    if drift:
        lines = "\n".join(
            f"  {k}: on disk {have!r}, generated {want!r}" for k, (have, want) in sorted(drift.items())
        )
        raise SystemExit(
            f"Existing config {config_path} is incompatible with this conversion:\n"
            f"{lines}\n"
            "No binary was written. The checked-in config encodes measured "
            "architecture; delete it only if you intend to regenerate it."
        )


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
        # state_dict values are numpy arrays (safetensors.numpy) or torch
        # tensors (torch fallback); normalize both to float32 numpy.
        if hasattr(tensor, "detach"):
            arr = tensor.detach().float().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)

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

    # Validate the sidecar BEFORE touching the binary. Writing 540 MB first and
    # then refusing to update the config left a new binary paired with a stale
    # config while the process still exited zero.
    config = build_config(dtype)
    config_path = output_path.with_suffix(".json")
    check_existing_config(config_path, config)

    # --- Publication transaction for the binary/config PAIR ---
    #
    # The unit of consumption is the pair: the browser fetches kimodo.bin and
    # kimodo.json together, so atomicity must cover both, not each file alone.
    #
    # Discipline:
    #  * Every artifact is staged to a UNIQUE per-run temp file (pid+token) and
    #    fsynced before any publication. Fixed temp names let two concurrent
    #    conversions share an inode, allowing a failed run to mutate the
    #    already-published destination in place.
    #  * The config (when absent) is published BEFORE the binary. build_config
    #    is deterministic per dtype, so a config-without-binary intermediate is
    #    self-healing: the next run finds it compatible and publishes only the
    #    binary. A binary-without-config intermediate is a broken install.
    #  * If binary publication fails after config publication, the config is
    #    rolled back to its prior state (removed, since it did not exist).
    #  * Cleanup removes only this run's own temps; foreign temp files are
    #    never touched.
    import secrets
    run_tag = f"{os.getpid()}.{secrets.token_hex(4)}"
    tmp_path = output_path.parent / f"{output_path.name}.{run_tag}.tmp"
    config_missing = not config_path.exists()
    cfg_tmp = config_path.parent / f"{config_path.name}.{run_tag}.tmp" if config_missing else None

    def _cleanup_own_temps():
        for t in (tmp_path, cfg_tmp):
            if t is not None:
                try:
                    t.unlink()
                except OSError:
                    pass

    try:
        # Stage the binary.
        with open(tmp_path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<I", num_tensors))
            f.write(struct.pack("<I", header_size))

            for entry in tensor_entries:
                # Encode before writing so a non-ASCII name fails here, while
                # the destination is still untouched.
                name_bytes = entry["name"].encode("ascii")[:MAX_NAME_LEN]
                f.write(name_bytes.ljust(MAX_NAME_LEN, b"\x00"))
                f.write(struct.pack("<I", entry["dtype"]))
                ndim = len(entry["shape"])
                f.write(struct.pack("<I", ndim))
                shape_padded = entry["shape"] + [0] * (MAX_DIMS - ndim)
                for sh in shape_padded:
                    f.write(struct.pack("<I", sh))
                f.write(struct.pack("<I", entry["offset"]))
                f.write(struct.pack("<I", entry["size"]))

            f.write(weight_data)
            f.flush()
            os.fsync(f.fileno())

        # Stage the config (fresh installs only) before publishing ANYTHING.
        if config_missing:
            with open(cfg_tmp, "w") as f:
                json.dump(config, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

        # Publish: config first (self-healing intermediate), then binary.
        if config_missing:
            os.replace(cfg_tmp, config_path)
        try:
            os.replace(tmp_path, output_path)
        except BaseException:
            if config_missing:
                # Roll the config back to its prior state (absent).
                try:
                    config_path.unlink()
                except OSError:
                    pass
            raise
    except BaseException:
        _cleanup_own_temps()
        raise

    total_mb = len(weight_data) / (1024 * 1024)
    print(f"Weights written to {output_path}")
    print(f"  Tensors: {num_tensors}")
    print(f"  Size: {total_mb:.1f} MB ({dtype})")
    print(f"  Header: {header_size} bytes")
    if config_missing:
        print(f"Config written to {config_path}")
    else:
        print(f"Config already present and matching: {config_path}")

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
            # numpy arrays expose .size; torch tensors expose .numel(). The
            # numpy loader is now the default route, so .numel() alone raised
            # AttributeError here.
            count = t.numel() if hasattr(t, "numel") else np.asarray(t).size
            total += count
            print(f"  {name:60s} {str(list(t.shape)):>20s}  {count:>10d}")
        print(f"\nTotal: {total:,} params ({total*4/1e6:.1f} MB fp32, {total*2/1e6:.1f} MB fp16)")
        return

    convert(state_dict, args.output, args.dtype)


if __name__ == "__main__":
    main()
