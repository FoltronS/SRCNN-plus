"""
Regenerate a training curve PNG from a saved train_log.csv.

Usage:
    python plot_curve.py --run-dir outputs/x4_mse_noaug_nosched_nores
    python plot_curve.py --run-dir outputs/x4_l1   # any run dir with train_log.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory containing train_log.csv')
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        raise FileNotFoundError('train_log.csv not found in {}'.format(args.run_dir))

    epochs, train_losses, eval_psnrs = [], [], []
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            eval_psnrs.append(float(row['eval_psnr']))

    best_psnr = max(eval_psnrs)
    best_epoch = eval_psnrs.index(best_psnr)
    run_name = os.path.basename(args.run_dir.rstrip('/\\'))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, train_losses, color='steelblue', linewidth=1.5)
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Curve: {}'.format(run_name))
    ax1.grid(True, alpha=0.4)

    ax2.plot(epochs, eval_psnrs, color='darkorange', linewidth=1.5)
    ax2.axhline(y=best_psnr, color='red', linestyle='--', linewidth=1.2,
                label='Best {:.2f} dB (epoch {})'.format(best_psnr, best_epoch))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Eval PSNR (dB)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(args.run_dir, 'training_curve.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved to {}'.format(out_path))
