"""
Evaluate a saved SRCNN checkpoint on an HDF5 eval dataset.

Reads config.json from the same directory as the weights file to auto-detect
the residual setting. Use --no-residual to override.

Usage:
    python eval.py --weights-file outputs/x3_l1/best.pth \
                   --eval-file datasets/Set14_x3.h5
"""

import argparse
import json
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from models import SRCNN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--no-residual', action='store_true',
                        help='Override: disable residual (auto-detected from config.json if present)')
    parser.add_argument('--border', type=int, default=None,
                        help='Pixels to crop from each border before computing metrics. '
                             'Defaults to scale extracted from eval filename (e.g. 3 for Set5_x3.h5). '
                             'Matches VDSR official evaluation protocol.')
    args = parser.parse_args()

    # Auto-detect border crop from eval filename if not specified.
    # Uses scale to match VDSR (Kim et al., CVPR 2016) evaluation protocol.
    if args.border is None:
        try:
            scale = int(os.path.splitext(os.path.basename(args.eval_file))[0].split('_x')[1])
            args.border = scale
        except (IndexError, ValueError):
            args.border = 0

    # Auto-detect residual from config.json written by train.py
    run_dir = os.path.dirname(os.path.abspath(args.weights_file))
    config_path = os.path.join(run_dir, 'config.json')
    residual = True
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        residual = cfg.get('residual', True)
        print('Config loaded from {}: residual={}'.format(config_path, residual))
    if args.no_residual:
        residual = False

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN(residual=residual).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    b = args.border
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for inputs, labels in eval_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
        if b > 0:
            preds  = preds[:, :, b:-b, b:-b]
            labels = labels[:, :, b:-b, b:-b]
        psnr_meter.update(calc_psnr(preds, labels), len(inputs))
        ssim_meter.update(calc_ssim(preds, labels), len(inputs))

    dataset_name = os.path.splitext(os.path.basename(args.eval_file))[0]
    print('PSNR on {}: {:.2f} dB  (border crop: {})'.format(dataset_name, psnr_meter.avg, b))
    print('SSIM on {}: {:.4f}  (border crop: {})'.format(dataset_name, ssim_meter.avg, b))
