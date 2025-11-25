
import math
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from .config import VALUE_BYTES, INDEX_BYTES, MIN_NNZ

# Cache: B_i_bytes (int) -> {"fraction", "entries", "bytes"}
COMPRESSION_CACHE: Dict[int, Dict[str, float]] = {}


def _float_param_count(state_dict: Dict[str, torch.Tensor]) -> int:
    """Count floating-point parameters in a state_dict."""
    return sum(
        t.numel()
        for t in state_dict.values()
        if torch.is_tensor(t) and t.is_floating_point()
    )


def compute_old_bits_per_entry(
    state_dict: Dict[str, torch.Tensor],
    value_bits: int = 32,
    index_mode: str = "per-model",
    fixed_index_bits: Optional[int]= None,
) -> int:
    """
    value_bits: bits used for the value (32 in FedBand).
    index_mode:
        - 'per-model': ceil(log2(#float params))
        - 'per-layer': max_j ceil(log2(numel(layer_j)))
    fixed_index_bits: if set, overrides the above (e.g. 24).
    """
    if fixed_index_bits is not None:
        idx_bits = int(fixed_index_bits)
    elif index_mode == "per-model":
        idx_bits = math.ceil(math.log2(max(1, _float_param_count(state_dict))))
    elif index_mode == "per-layer":
        idx_bits = max(
            math.ceil(math.log2(max(1, t.numel())))
            for t in state_dict.values()
            if torch.is_tensor(t) and t.is_floating_point()
        )
    else:
        idx_bits = 32  # safe fallback

    return value_bits + idx_bits  # bits / entry


BYTES_PER_ENTRY = VALUE_BYTES + INDEX_BYTES


def mb_to_fraction_via_entries(
    old_mb: float,
    param_count: int,
    old_bits_per_entry: int,
    bytes_full: float,
) -> Tuple[float, int, float]:
    """
    Convert FedBand's 'old' MB accounting to a fraction of BYTES_FULL.

    Returns:
        frac      : fraction in [0,1] of full dense model
        k         : number of kept entries
        bytes_now : effective bytes with (VALUE_BYTES+INDEX_BYTES)
    """
    bits = old_mb * 8.0 * 1e6  # MB -> bits
    k = max(0, int(bits // old_bits_per_entry))
    bytes_now = k * BYTES_PER_ENTRY
    frac = min(1.0, bytes_now / float(bytes_full))
    return frac, k, bytes_now


@torch.no_grad()
def _sd_zero_like(model_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """State_dict of zeros, preserving shapes/dtypes for float tensors."""
    return {
        k: (torch.zeros_like(v) if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in model_sd.items()
    }


@torch.no_grad()
def compress_delta_topk_by_bytes(
    delta_state: Dict[str, torch.Tensor],
    byte_budget: int,
    value_bytes: int = VALUE_BYTES,
    index_bytes: int = INDEX_BYTES,
    min_keep: int = MIN_NNZ,
):
    """
    Global top-k over |delta| across floating-point tensors only.

    Returns:
        sparse_update: { name: {'shape': shape, 'idx': LongTensor[k_i], 'val': Tensor[k_i]} }
        nnz_total: total kept entries.
    """
    recs = []  # (name, flat_tensor, numel, dtype)
    total_elems = 0
    for name, t in delta_state.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        v = t.detach().contiguous().view(-1)
        if v.numel() == 0:
            continue
        recs.append((name, v, v.numel(), t.dtype))
        total_elems += v.numel()

    if total_elems == 0 or byte_budget <= 0:
        return {}, 0

    per_entry = value_bytes + index_bytes
    k_total = int(byte_budget // per_entry)
    if min_keep is not None:
        k_total = max(k_total, int(min_keep))
    k_total = min(k_total, total_elems)
    if k_total <= 0:
        return {}, 0

    dev = recs[0][1].device

    abs_flat = torch.empty(total_elems, dtype=torch.float32, device=dev)
    sgn_flat = torch.empty(total_elems, dtype=torch.float32, device=dev)

    boundaries = [0]
    names = []
    dtypes = []
    o = 0
    for name, v, n, dt in recs:
        abs_flat[o:o + n] = v.abs().to(torch.float32)
        sgn_flat[o:o + n] = v.to(torch.float32)
        o += n
        boundaries.append(o)
        names.append(name)
        dtypes.append(dt)

    _, top_idx = torch.topk(abs_flat, k_total, largest=True, sorted=False)

    bounds_t = torch.tensor(boundaries, device=dev, dtype=torch.long)
    layer_j = torch.bucketize(top_idx, bounds_t[1:])
    local_i = top_idx - bounds_t[layer_j]

    sparse_update = {}
    m = len(names)
    for j in range(m):
        mask = layer_j == j
        if not mask.any():
            continue
        idx_j = local_i[mask].to(torch.long)
        val_j = sgn_flat[top_idx[mask]].to(dtypes[j])
        name = names[j]
        t_ref = delta_state[name]
        sparse_update[name] = {
            "shape": t_ref.shape,
            "idx": idx_j,
            "val": val_j,
        }

    return sparse_update, int(top_idx.numel())


def print_round_bandwidth_report(
    title,
    selected_clients,
    used_bytes_per_client,
    round_used_bytes,
    round_cap_bytes,
    bytes_full: float,
):
    dense_size = float(bytes_full)
    print(f"\n=== {title} Round Bandwidth Report ===")
    for cid in selected_clients:
        actual = used_bytes_per_client[cid]
        comp = actual / dense_size
        print(
            f"Client {cid:3d} | Actual={actual/1e6:6.2f}MB  Comp={comp:4.2f}×dense"
        )

    print("-----------------------------------")
    print(f"Actual  total: {round_used_bytes/1e6:.2f} MB")
    print(f"Limit   total: {round_cap_bytes/1e6:.2f} MB")
    util = (round_used_bytes / round_cap_bytes) * 100.0 if round_cap_bytes > 0 else 0.0
    left = max(0.0, round_cap_bytes - round_used_bytes)
    print(f"Utilization : {util:.1f}%   Left: {left/1e6:.2f} MB")
    print("===================================\n")


def shares_from_bytes(per_client_bytes, B_bytes: float, eps: float = 1e-12):
    """Convert per-client absolute bytes to shares that sum to ~1."""
    x = np.array(per_client_bytes, dtype=np.float64)
    S = float(x.sum())
    if S <= eps or B_bytes <= eps:
        return (np.ones_like(x) / len(x)).tolist()
    return (x / S).tolist()


def get_or_compute_compression(
    B_i_bytes: float,
    model_ref_state: Dict[str, torch.Tensor],
    old_bits_per_entry: int,
    bytes_full: float,
):
    """
    FedBand caching of compression levels.

    Returns:
        info : dict with keys:
               - "fraction": CR ∈ [0,1]
               - "entries" : number of kept params (k)
               - "bytes"   : effective bytes
        hit  : bool, True if cache hit
    """
    B_key = int(round(B_i_bytes))
    if B_key in COMPRESSION_CACHE:
        return COMPRESSION_CACHE[B_key], True

    float_elems = _float_param_count(model_ref_state)
    B_i_MB = B_i_bytes / 1e6

    frac, k, bytes_eff = mb_to_fraction_via_entries(
        old_mb=B_i_MB,
        param_count=float_elems,
        old_bits_per_entry=old_bits_per_entry,
        bytes_full=bytes_full,
    )

    info = {
        "fraction": bytes_eff / float(bytes_full),
        "entries": k,
        "bytes": bytes_eff,
    }
    COMPRESSION_CACHE[B_key] = info
    return info, False

