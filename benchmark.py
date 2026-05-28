"""
Evaluate multiple SRCNN checkpoints across multiple datasets.
Produces a CSV results table and a grouped bar chart PNG.

Each run directory must contain:
  - best.pth        (saved by train.py at end of training)
  - config.json     (saved by train.py, used to auto-detect residual setting)

Usage:
    python benchmark.py \\
        --run-dirs outputs/x3_mse_noaug_nosched_nores \\
                   outputs/x3_mse_nosched_nores \\
                   outputs/x3_mse_nores \\
                   outputs/x3_l1_nores \\
                   outputs/x3_l1 \\
                   outputs/x3_mse \\
        --labels "Baseline" "+Aug" "+Sched" "+L1" "+Residual (L1)" "Full (MSE)" \\
        --eval-files datasets/Set5_x3.h5 datasets/Set14_x3.h5 \\
        --output results/ablation.csv
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from models import SRCNN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim


# Reference numbers from published papers (Set5 x3 and Set14 x3)
# PSNR source: Dong et al. TPAMI 2016, Kim et al. CVPR 2016
# SSIM source: Dong et al. TPAMI 2016, Kim et al. CVPR 2016
PAPER_REFS = {
    'SRCNN': {
        'Set5_PSNR':  32.75, 'Set14_PSNR': 29.28,
        'Set5_SSIM':  0.9090, 'Set14_SSIM': 0.8209,
    },
    'VDSR': {
        'Set5_PSNR':  33.66, 'Set14_PSNR': 29.77,
        'Set5_SSIM':  0.9213, 'Set14_SSIM': 0.8314,
    },
}


def crop_border(tensor, border):
    """Crop `border` pixels from each side of a (1, 1, H, W) tensor."""
    if border <= 0:
        return tensor
    return tensor[:, :, border:-border, border:-border]


def evaluate(weights_file, eval_file, residual=True, border=0):
    """Return (psnr, ssim) averaged over all images in the eval set.

    border: pixels to crop from each side before computing metrics,
            matching the paper evaluation protocol (typically == scale).
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN(residual=residual).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    dataset = EvalDataset(eval_file)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
        preds_c  = crop_border(preds,  border)
        labels_c = crop_border(labels, border)
        psnr_meter.update(calc_psnr(preds_c, labels_c), len(inputs))
        ssim_meter.update(calc_ssim(preds_c, labels_c), len(inputs))
    return float(psnr_meter.avg), float(ssim_meter.avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', type=str, nargs='+', required=True,
                        help='Output directories from train.py (each must contain best.pth)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Display label for each run directory (same order). '
                             'Defaults to the directory basename.')
    parser.add_argument('--eval-files', type=str, nargs='+',
                        default=['datasets/Set5_x3.h5', 'datasets/Set14_x3.h5'],
                        help='HDF5 eval files to benchmark against')
    parser.add_argument('--output', type=str, default='results/ablation.csv',
                        help='Output CSV path (chart saved alongside it)')
    parser.add_argument('--no-chart', action='store_true', help='Skip the bar chart PNG')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(d.rstrip('/\\')) for d in args.run_dirs]
    if len(args.labels) != len(args.run_dirs):
        raise ValueError('--labels count must match --run-dirs count')

    # Derive short dataset names and scales from filenames: "Set5_x3.h5" -> "Set5", 3
    dataset_names = [
        os.path.splitext(os.path.basename(f))[0].split('_x')[0]
        for f in args.eval_files
    ]
    # Border crop = scale, matching VDSR (Kim et al., CVPR 2016) evaluation protocol
    dataset_scales = []
    for f in args.eval_files:
        try:
            scale = int(os.path.splitext(os.path.basename(f))[0].split('_x')[1])
        except (IndexError, ValueError):
            scale = 0
        dataset_scales.append(scale)
    # Column names: "Set5_PSNR", "Set5_SSIM", "Set14_PSNR", "Set14_SSIM", ...
    col_names = [
        '{}_{}'.format(ds, metric)
        for ds in dataset_names
        for metric in ('PSNR', 'SSIM')
    ]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cudnn.benchmark = True
    results = []

    for run_dir, label in zip(args.run_dirs, args.labels):
        weights_file = os.path.join(run_dir, 'best.pth')
        config_path  = os.path.join(run_dir, 'config.json')

        if not os.path.exists(weights_file):
            print('[SKIP] {} (best.pth not found)'.format(run_dir))
            continue

        residual = True
        if os.path.exists(config_path):
            with open(config_path) as f:
                residual = json.load(f).get('residual', True)

        row = {'label': label}
        print('\n{} (residual={})'.format(label, residual))
        for eval_file, ds_name, scale in zip(args.eval_files, dataset_names, dataset_scales):
            if not os.path.exists(eval_file):
                print('  [SKIP] {} not found'.format(eval_file))
                row[ds_name + '_PSNR'] = None
                row[ds_name + '_SSIM'] = None
                continue
            psnr, ssim = evaluate(weights_file, eval_file, residual=residual, border=scale)
            row[ds_name + '_PSNR'] = round(psnr, 2)
            row[ds_name + '_SSIM'] = round(ssim, 4)
            print('  {}: {:.2f} dB  SSIM {:.4f}  (border crop: {})'.format(ds_name, psnr, ssim, scale))
        results.append(row)

    if not results:
        print('No results to save.')
        exit(0)

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label'] + col_names)
        writer.writeheader()
        writer.writerows(results)
    print('\nResults saved to {}'.format(args.output))

    # Print console table
    col_w = max(len(r['label']) for r in results) + 2
    header = '{:<{}}'.format('Model', col_w) + ''.join('{:>12}'.format(c) for c in col_names)
    print('\n' + header)
    print('-' * len(header))
    for row in results:
        vals = ''.join(
            '{:>12}'.format('{:.2f} dB'.format(row[c]) if 'PSNR' in c else '{:.4f}'.format(row[c]))
            if row[c] is not None else '{:>12}'.format('N/A')
            for c in col_names
        )
        print('{:<{}}{}'.format(row['label'], col_w, vals))

    if args.no_chart:
        exit(0)

    # Bar chart: 2 rows (PSNR / SSIM) x N datasets
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(2, n_datasets, figsize=(6 * n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(len(results))
    ref_styles = [
        ('SRCNN', 'gray',  '--'),
        ('VDSR', 'black', ':'),
    ]
    labels_x = [r['label'] for r in results]

    for col, ds_name in enumerate(dataset_names):
        # PSNR row
        ax = axes[0][col]
        values = [row.get(ds_name + '_PSNR') for row in results]
        valid  = [v for v in values if v is not None]
        bars = ax.bar(x, values, width=0.55, color='steelblue', alpha=0.85)
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        '{:.2f}'.format(v), ha='center', va='bottom', fontsize=8)
        for ref_name, color, ls in ref_styles:
            key = ds_name + '_PSNR'
            if key in PAPER_REFS.get(ref_name, {}):
                ref_val = PAPER_REFS[ref_name][key]
                ax.axhline(ref_val, color=color, linestyle=ls, linewidth=1.3,
                           label='{}: {:.2f} dB'.format(ref_name, ref_val))
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('{} PSNR'.format(ds_name))
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.35)
        all_vals = valid + [
            PAPER_REFS[rn][ds_name + '_PSNR']
            for rn, _, _ in ref_styles
            if ds_name + '_PSNR' in PAPER_REFS.get(rn, {})
        ]
        if all_vals:
            ax.set_ylim(min(all_vals) - 0.5, max(all_vals) + 0.8)

        # SSIM row
        ax = axes[1][col]
        values = [row.get(ds_name + '_SSIM') for row in results]
        valid  = [v for v in values if v is not None]
        bars = ax.bar(x, values, width=0.55, color='darkorange', alpha=0.85)
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                        '{:.4f}'.format(v), ha='center', va='bottom', fontsize=8)
        for ref_name, color, ls in ref_styles:
            key = ds_name + '_SSIM'
            if key in PAPER_REFS.get(ref_name, {}):
                ref_val = PAPER_REFS[ref_name][key]
                ax.axhline(ref_val, color=color, linestyle=ls, linewidth=1.3,
                           label='{}: {:.4f}'.format(ref_name, ref_val))
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('SSIM')
        ax.set_title('{} SSIM'.format(ds_name))
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.35)
        all_vals = valid + [
            PAPER_REFS[rn][ds_name + '_SSIM']
            for rn, _, _ in ref_styles
            if ds_name + '_SSIM' in PAPER_REFS.get(rn, {})
        ]
        if all_vals:
            ax.set_ylim(min(all_vals) - 0.01, max(all_vals) + 0.02)

    plt.tight_layout()
    chart_path = os.path.splitext(args.output)[0] + '_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Chart saved to {}'.format(chart_path))
