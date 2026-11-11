"""Offline, model-agnostic overlapping tiling for images and optional masks and ground-truth images.

Example:
 python crop_tiles.py --images ./original --masks ./masks --output ./tiles --tile-size 1024 --overlap 128

Produces output/images, output/masks (optional), output/ground_truth (optional), and output/manifest.json.
Requires Pillow and NumPy. Input images should be 8-bit RGB/gray PNG/TIFF/etc.
"""
import argparse
import json
from pathlib import Path
from PIL import Image

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def starts(length, tile, overlap):
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError('Require tile > 0 and 0 <= overlap < tile')
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, tile - overlap))
    last = length - tile
    if out[-1] != last:
        out.append(last)
    return out


def find_matching(directory, stem, kind):
    matches = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTS and p.stem == stem]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {kind} for {stem}, found {len(matches)}')
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--masks', type=Path, default=None)
    ap.add_argument('--ground-truth', '--gt', dest='ground_truth', type=Path, default=None,
                    help='Directory of matching shadow-free ground truth images')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tile-size', type=int, default=1024)
    ap.add_argument('--overlap', type=int, default=128)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'images').mkdir(exist_ok=True)
    if args.masks:
        (args.output / 'masks').mkdir(exist_ok=True)
    if args.ground_truth:
        (args.output / 'ground_truth').mkdir(exist_ok=True)
    paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        raise RuntimeError('No input images found')
    stems = [p.stem for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError('Input image stems must be unique')
    manifest = {'tile_size': args.tile_size, 'minimum_overlap': args.overlap, 'images': []}
    for path in paths:
        with Image.open(path) as src:
            img = src.copy()
        w, h = img.size
        mask = None
        if args.masks:
            with Image.open(find_matching(args.masks, path.stem, 'mask')) as src:
                mask = src.copy()
            if mask.size != img.size:
                raise ValueError(f'Mask size mismatch: {path.name}')
        gt = None
        if args.ground_truth:
            with Image.open(find_matching(args.ground_truth, path.stem, 'ground truth')) as src:
                gt = src.copy()
            if gt.size != img.size:
                raise ValueError(f'Ground truth size mismatch: {path.name}: {gt.size} vs {img.size}')
        record = {'source': path.name, 'stem': path.stem, 'width': w, 'height': h, 'tiles': []}
        for yi, y in enumerate(starts(h, args.tile_size, args.overlap)):
            for xi, x in enumerate(starts(w, args.tile_size, args.overlap)):
                tile_name = f'{path.stem}__y{yi:03d}_x{xi:03d}.png'
                # No resize. Pad only if original dimension is smaller than tile_size.
                crop = img.crop((x, y, min(x + args.tile_size, w), min(y + args.tile_size, h)))
                valid_w, valid_h = crop.size
                if crop.size != (args.tile_size, args.tile_size):
                    padded = Image.new(img.mode, (args.tile_size, args.tile_size))
                    padded.paste(crop, (0, 0))
                    crop = padded
                crop.save(args.output / 'images' / tile_name)
                if mask is not None:
                    mask_crop = mask.crop((x, y, min(x + args.tile_size, w), min(y + args.tile_size, h)))
                    if mask_crop.size != (args.tile_size, args.tile_size):
                        padded = Image.new(mask.mode, (args.tile_size, args.tile_size))
                        padded.paste(mask_crop, (0, 0))
                        mask_crop = padded
                    mask_crop.save(args.output / 'masks' / tile_name)
                if gt is not None:
                    gt_crop = gt.crop((x, y, min(x + args.tile_size, w), min(y + args.tile_size, h)))
                    if gt_crop.size != (args.tile_size, args.tile_size):
                        padded = Image.new(gt.mode, (args.tile_size, args.tile_size))
                        padded.paste(gt_crop, (0, 0))
                        gt_crop = padded
                    gt_crop.save(args.output / 'ground_truth' / tile_name)
                record['tiles'].append({'file': tile_name, 'x': x, 'y': y, 'valid_width': valid_w, 'valid_height': valid_h})
        manifest['images'].append(record)
        print(f'{path.name}: {w}x{h}, {len(record["tiles"])} tiles')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved manifest:', args.output / 'manifest.json')


if __name__ == '__main__':
    main()
