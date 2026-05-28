"""
Generate visual comparison images and per-image metrics for the report.

Saves GT, bicubic, and SR outputs into:
    comparisons/<image-stem>/x<scale>/

Usage (single scale):
    python test.py --image-file data/butterfly.bmp --scale 3

All scales x2/x3/x4 at once (omit --scale):
    python test.py --image-file data/butterfly.bmp

With crop (loads crop.json if exists, otherwise opens matplotlib to draw):
    python test.py --image-file data/butterfly.bmp --crop
    python test.py --image-file data/butterfly.bmp --scale 3 --crop

crop.json is stored at the image level (comparisons/<image-stem>/crop.json)
and shared across all scales. To redraw the region, delete it and rerun with --crop.
"""

import argparse
import csv
import json
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, calc_ssim


def save_with_crop(img, path, crop, crop_dir=None):
    """Save full image and, if crop is given, save cropped version to crop_dir."""
    img.save(path)
    if crop is not None and crop_dir is not None:
        os.makedirs(crop_dir, exist_ok=True)
        cropped = img.crop(crop)
        cropped.save(os.path.join(crop_dir, os.path.basename(path)))


def run_model(weights_file, y, ycbcr, hr_y, out_dir, crop=None, crop_dir=None):
    """Run one SR model and return (psnr, ssim, run_name)."""
    run_dir = os.path.dirname(os.path.abspath(weights_file))
    config_path = os.path.join(run_dir, 'config.json')
    residual = True
    if os.path.exists(config_path):
        with open(config_path) as f:
            residual = json.load(f).get('residual', True)

    device = y.device
    model = SRCNN(residual=residual).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = float(calc_psnr(hr_y, preds))
    ssim = float(calc_ssim(hr_y, preds))

    preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds_np, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    run_name = os.path.basename(run_dir)
    save_with_crop(pil_image.fromarray(output), os.path.join(out_dir, '{}.png'.format(run_name)), crop, crop_dir)
    return psnr, ssim, run_name


def select_crop_interactively(hr_image, size=100):
    """Open matplotlib window, user clicks the top-left corner of the crop region.
    A square of `size x size` pixels is placed at that point."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(hr_image)
    ax.set_title('Click the TOP-LEFT corner of the crop region ({}×{} square)'.format(size, size))
    plt.tight_layout()

    print('Click the top-left corner of the {}x{} crop region...'.format(size, size))
    points = plt.ginput(1, timeout=0)
    plt.close()

    if len(points) == 1:
        x1, y1 = int(points[0][0]), int(points[0][1])
        crop = (x1, y1, x1 + size, y1 + size)
        print('Selected crop ({}x{} square): left={} top={} right={} bottom={}'.format(size, size, *crop))
        return crop

    print('No region selected.')
    return None


def process_scale(scale, image, crop, weights_files, out_root, device):
    """Run inference for one scale and save all outputs. Returns list of metric rows."""
    out_dir = os.path.join(out_root, 'x{}'.format(scale))
    os.makedirs(out_dir, exist_ok=True)

    image_width  = (image.width  // scale) * scale
    image_height = (image.height // scale) * scale
    hr_image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    # Clamp crop to this scale's HR image dimensions (coordinates are shared across scales)
    scaled_crop = None
    crop_dir = None
    if crop is not None:
        left  = max(0, min(crop[0], image_width  - 1))
        top   = max(0, min(crop[1], image_height - 1))
        right = max(left + 1, min(crop[2], image_width))
        bot   = max(top  + 1, min(crop[3], image_height))
        scaled_crop = (left, top, right, bot)
        crop_dir = os.path.join(out_dir, 'cropped')

    # Save GT
    save_with_crop(hr_image, os.path.join(out_dir, 'original.png'), scaled_crop, crop_dir)

    hr_np    = np.array(hr_image).astype(np.float32)
    hr_ycbcr = convert_rgb_to_ycbcr(hr_np)
    hr_y     = torch.from_numpy(hr_ycbcr[..., 0] / 255.).to(device).unsqueeze(0).unsqueeze(0)

    # Bicubic
    bicubic = hr_image.resize((image_width // scale, image_height // scale), resample=pil_image.BICUBIC)
    bicubic = bicubic.resize((image_width, image_height), resample=pil_image.BICUBIC)
    bicubic_name = 'x{}_bicubic'.format(scale)
    save_with_crop(bicubic, os.path.join(out_dir, '{}.png'.format(bicubic_name)), scaled_crop, crop_dir)

    bicubic_np    = np.array(bicubic).astype(np.float32)
    bicubic_ycbcr = convert_rgb_to_ycbcr(bicubic_np)
    bicubic_y     = torch.from_numpy(bicubic_ycbcr[..., 0] / 255.).to(device).unsqueeze(0).unsqueeze(0)
    bicubic_psnr  = float(calc_psnr(hr_y, bicubic_y))
    bicubic_ssim  = float(calc_ssim(hr_y, bicubic_y))

    y = torch.from_numpy(bicubic_ycbcr[..., 0] / 255.).to(device).unsqueeze(0).unsqueeze(0)

    rows = [(bicubic_name, bicubic_psnr, bicubic_ssim)]
    print('\n=== Scale x{} ==='.format(scale))
    print('{:<40} {:>10} {:>8}'.format('Model', 'PSNR (dB)', 'SSIM'))
    print('-' * 60)
    print('{:<40} {:>10.2f} {:>8.4f}'.format(bicubic_name, bicubic_psnr, bicubic_ssim))

    for wf in weights_files:
        wf_scaled = wf.format(scale)
        if not os.path.exists(wf_scaled):
            print('{:<40} [SKIP] not found'.format(wf_scaled))
            continue
        psnr, ssim, run_name = run_model(wf_scaled, y, bicubic_ycbcr, hr_y, out_dir, scaled_crop, crop_dir)
        rows.append((run_name, psnr, ssim))
        print('{:<40} {:>10.2f} {:>8.4f}'.format(run_name, psnr, ssim))

    csv_path = os.path.join(out_dir, 'x{}_metrics.csv'.format(scale))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'psnr_db', 'ssim'])
        for row in rows:
            writer.writerow([row[0], '{:.2f}'.format(row[1]), '{:.4f}'.format(row[2])])

    print('Saved to {}/'.format(out_dir))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=None,
                        help='Scale factor (2, 3, or 4). Omit to run all three.')
    parser.add_argument('--weights-files', type=str, nargs='+', default=None,
                        help='Checkpoint path templates with {} for scale. '
                             'Defaults to baseline + full L1.')
    parser.add_argument('--output-dir', type=str, default='comparisons',
                        help='Root output directory (default: comparisons/)')
    parser.add_argument('--crop', action='store_true',
                        help='Enable crop: loads crop.json if it exists, otherwise opens '
                             'matplotlib to draw the region and saves it to crop.json. '
                             'crop.json is shared across all scales.')
    parser.add_argument('--crop-size', type=int, default=100,
                        help='Side length of the square crop region in pixels (default: 100)')
    args = parser.parse_args()

    scales = [args.scale] if args.scale is not None else [2, 3, 4]

    if args.weights_files is None:
        weights_templates = [
            'outputs/x{}_mse_noaug_nosched_nores/best.pth',
            'outputs/x{}_l1/best.pth',
        ]
    else:
        weights_templates = args.weights_files

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img_stem = os.path.splitext(os.path.basename(args.image_file))[0]
    out_root = os.path.join(args.output_dir, img_stem)
    os.makedirs(out_root, exist_ok=True)

    image = pil_image.open(args.image_file).convert('RGB')

    # Crop is shared across scales — stored at the image level
    crop_json = os.path.join(out_root, 'crop.json')
    crop = None

    if args.crop:
        if os.path.exists(crop_json):
            with open(crop_json) as f:
                crop = tuple(json.load(f))
            print('Crop loaded from {}: {}'.format(crop_json, crop))
        else:
            # Select interactively on the x3 HR image (reference scale)
            ref_scale = 3 if 3 in scales else scales[0]
            ref_w = (image.width  // ref_scale) * ref_scale
            ref_h = (image.height // ref_scale) * ref_scale
            ref_hr = image.resize((ref_w, ref_h), resample=pil_image.BICUBIC)
            crop = select_crop_interactively(ref_hr, size=args.crop_size)
            if crop:
                with open(crop_json, 'w') as f:
                    json.dump(list(crop), f)
                print('Crop saved to {}: {}'.format(crop_json, crop))
            else:
                print('No region selected, skipping crop.')

    for scale in scales:
        process_scale(scale, image, crop, weights_templates, out_root, device)
