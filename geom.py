import numpy as np
import math

def radar_points(values, angles):
    # values in [0,1], angles radians
    x = values * np.cos(angles)
    y = values * np.sin(angles)
    return np.stack([x,y], axis=1)

def polygon_area(poly):
    # poly Nx2 closed not required
    x = poly[:,0]; y = poly[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def point_in_poly(px, py, poly):
    # ray casting
    x = poly[:,0]; y = poly[:,1]
    n = len(poly)
    inside = False
    j = n-1
    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def raster_mask(poly, grid_n=160, span=1.05):
    """Create boolean mask on a grid for a polygon.

    NOTE: This implementation is vectorized via matplotlib.path.Path for speed.
    poly: (N,2) array of vertices
    grid_n: grid resolution
    span: coordinate span, grid covers [-span, span] on both axes
    """
    from matplotlib.path import Path

    xs = np.linspace(-span, span, grid_n, dtype=np.float32)
    ys = np.linspace(-span, span, grid_n, dtype=np.float32)
    # Create grid of points (x,y) with shape (grid_n*grid_n, 2)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    path = Path(poly.astype(np.float32), closed=True)
    inside = path.contains_points(pts)
    return inside.reshape((grid_n, grid_n))


def overlap_uniqueness(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    jacc = inter / union
    return 1.0 - jacc


def team_discrete_stats(masks, indices):
    """Compute discrete-geometry stats for a team.

    Returns dict with:
      - union_cells
      - exclusive_cells (cells covered by exactly one member)
      - overlap_cells   (cells covered by 2+ members)
      - diversity_exclusive_over_union

    `masks` is a list/array of boolean (H,W) masks.
    `indices` is an iterable of member indices.
    """
    idx = list(indices)
    if len(idx) == 0:
        return {
            "union_cells": 0,
            "exclusive_cells": 0,
            "overlap_cells": 0,
            "diversity_exclusive_over_union": 0.0,
        }
    # accumulate coverage count
    h, w = masks[idx[0]].shape
    cnt = np.zeros((h, w), dtype=np.int16)
    for i in idx:
        cnt += masks[i].astype(np.int16)

    union = int((cnt > 0).sum())
    excl = int((cnt == 1).sum())
    ovlp = int((cnt > 1).sum())
    div = (excl / union) if union > 0 else 0.0
    return {
        "union_cells": union,
        "exclusive_cells": excl,
        "overlap_cells": ovlp,
        "diversity_exclusive_over_union": float(div),
        "count_grid": cnt,
    }


# -------------------------------
# Mask caching helpers
# -------------------------------

def pack_mask(mask: np.ndarray) -> np.ndarray:
    """Pack boolean mask (H,W) into uint8 bytes using np.packbits."""
    flat = mask.astype(np.uint8).reshape(-1)
    return np.packbits(flat)

def unpack_mask(packed: np.ndarray, shape: tuple) -> np.ndarray:
    """Unpack uint8 packed bits into boolean mask with given shape."""
    bits = np.unpackbits(packed).astype(bool)
    need = int(np.prod(shape))
    bits = bits[:need]
    return bits.reshape(shape)

def save_masks_cache(path: str, packed_masks: np.ndarray, areas: np.ndarray, grid_n: int, span: float) -> None:
    """Save packed masks + areas to a .npz."""
    np.savez_compressed(path, packed_masks=packed_masks, areas=areas, grid_n=np.array([grid_n], dtype=np.int32), span=np.array([span], dtype=np.float32))

def load_masks_cache(path: str):
    """Load packed masks + areas from a .npz."""
    data = np.load(path, allow_pickle=False)
    packed_masks = data["packed_masks"]
    areas = data["areas"]
    grid_n = int(data["grid_n"][0])
    span = float(data["span"][0])
    return packed_masks, areas, grid_n, span
