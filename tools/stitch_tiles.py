"""Offline model-agnostic weighted stitching of restored tiles.

Example:
 python stitch_tiles.py --manifest ./tiles/manifest.json --predictions ./results/fusediff --output ./stitched/fusediff

Prediction filenames must match the tile filenames recorded in manifest.json.
Outputs RGB/gray 8-bit PNG. No resize is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def axis_weight(length, left_overlap, right_overlap, eps=0.05):
    """Smooth transition only where neighbors overlap; no attenuation at image boundary."""
    w = np.ones(length, dtype=np.float32)
    if left_overlap > 0:
        t = (np.arange(left_overlap, dtype=np.float32) + 0.5) / left_overlap
        w[:left_overlap] *= eps + (1 - eps) * (0.5 - 0.5 * np.cos(np.pi * t))
    if right_overlap > 0:
        t = (np.arange(right_overlap, dtype=np.float32) + 0.5) / right_overlap
        w[-right_overlap:] *= eps + (1 - eps) * (0.5 + 0.5 * np.cos(np.pi * t))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    for rec in manifest['images']:
        h, w = rec['height'], rec['width']
        entries = rec['tiles']
        xs = sorted(set(e['x'] for e in entries))
        ys = sorted(set(e['y'] for e in entries))
        accum = None
        weights = np.zeros((h, w, 1), dtype=np.float64)
        for entry in entries:
            x, y = entry['x'], entry['y']
            vw, vh = entry['valid_width'], entry['valid_height']
            p = args.predictions / entry['file']
            if not p.is_file():
                raise FileNotFoundError(f'Missing prediction: {p}')
            with Image.open(p) as src:
                arr = np.asarray(src.convert('RGB') if src.mode not in ('L', 'RGB') else src, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.shape[0] < vh or arr.shape[1] < vw:
                raise ValueError(f'Prediction smaller than expected: {p}')
            arr = arr[:vh, :vw]
            if accum is None:
                accum = np.zeros((h, w, arr.shape[2]), dtype=np.float64)
            if arr.shape[2] != accum.shape[2]:
                raise ValueError('All predictions must have the same channel count')
            ix, iy = xs.index(x), ys.index(y)
            left = max(0, xs[ix-1] + manifest['tile_size'] - x) if ix > 0 else 0
            right = max(0, x + vw - xs[ix+1]) if ix + 1 < len(xs) else 0
            top = max(0, ys[iy-1] + manifest['tile_size'] - y) if iy > 0 else 0
            bottom = max(0, y + vh - ys[iy+1]) if iy + 1 < len(ys) else 0
            wx = axis_weight(vw, min(left, vw), min(right, vw))
            wy = axis_weight(vh, min(top, vh), min(bottom, vh))
            weight = (wy[:, None] * wx[None, :])[..., None]
            accum[y:y+vh, x:x+vw] += arr * weight
            weights[y:y+vh, x:x+vw] += weight
        if np.any(weights == 0):
            raise RuntimeError(f'Uncovered pixels for {rec["source"]}')
        out = np.clip(np.rint(accum / weights), 0, 255).astype(np.uint8)
        if out.shape[2] == 1:
            out = out[..., 0]
        Image.fromarray(out).save(args.output / f'{rec["stem"]}.png')
        print(f'{rec["stem"]}: reconstructed {w}x{h}')


if __name__ == '__main__':
    main()
