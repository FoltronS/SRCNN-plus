"""
Full training + benchmark pipeline.

For scale 3 (primary): trains all 6 ablation variants then runs benchmark.
For scale 2 or 4 (cross-scale): trains baseline + full model only then runs benchmark.

Usage:
    python run_ablation.py                   # x3 full ablation (default)
    python run_ablation.py --scale 2         # x2 baseline + full model
    python run_ablation.py --scale 4         # x4 baseline + full model
"""

import argparse
import os
import subprocess
import sys
import time


# 6-run ablation for x3
ABLATION_RUNS = [
    {'label': 'Baseline',           'loss': 'mse', 'no_augment': True,  'no_scheduler': True,  'no_residual': True},
    {'label': '+Augmentation',      'loss': 'mse', 'no_augment': False, 'no_scheduler': True,  'no_residual': True},
    {'label': '+LR Scheduler',      'loss': 'mse', 'no_augment': False, 'no_scheduler': False, 'no_residual': True},
    {'label': '+L1 Loss',           'loss': 'l1',  'no_augment': False, 'no_scheduler': False, 'no_residual': True},
    {'label': 'SRCNN+ (L1)',        'loss': 'l1',  'no_augment': False, 'no_scheduler': False, 'no_residual': False},
    {'label': 'SRCNN+ (MSE)',       'loss': 'mse', 'no_augment': False, 'no_scheduler': False, 'no_residual': False},
]

# 2-run comparison for x2 and x4
CROSS_SCALE_RUNS = [
    {'label': 'Baseline',        'loss': 'mse', 'no_augment': True,  'no_scheduler': True,  'no_residual': True},
    {'label': 'SRCNN+ (L1)',     'loss': 'l1',  'no_augment': False, 'no_scheduler': False, 'no_residual': False},
]


def run_dir_name(scale, run):
    """Reproduce the same auto-naming logic used in train.py."""
    name = 'x{}_{}'.format(scale, run['loss'])
    if run['no_augment']:   name += '_noaug'
    if run['no_scheduler']: name += '_nosched'
    if run['no_residual']:  name += '_nores'
    return name


def run_cmd(cmd, label):
    print('\n' + '=' * 60)
    print(label)
    print('=' * 60)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print('\n[FAILED] exit code {}. Stopping.'.format(result.returncode))
        sys.exit(result.returncode)
    print('\nDone in {:.0f} min {:.0f} sec'.format(elapsed // 60, elapsed % 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',        type=int, default=3, choices=[2, 3, 4])
    parser.add_argument('--train-file',   type=str, default=None,
                        help='Training HDF5 (default: datasets/91-image_x{scale}.h5)')
    parser.add_argument('--eval-file',    type=str, default=None,
                        help='Eval HDF5 for training (default: datasets/Set5_x{scale}.h5)')
    parser.add_argument('--outputs-dir',  type=str, default='outputs')
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--batch-size',   type=int, default=16)
    parser.add_argument('--num-epochs',   type=int, default=400)
    parser.add_argument('--num-workers',  type=int, default=0)
    parser.add_argument('--seed',         type=int, default=123)
    parser.add_argument('--eval-files',   type=str, nargs='+', default=None,
                        help='Datasets for benchmark.py. Defaults to Set5+Set14 for scale 3, '
                             'Set5 only for scale 2/4.')
    parser.add_argument('--results-dir',  type=str, default='results')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip benchmark.py after training')
    args = parser.parse_args()

    if args.train_file is None:
        args.train_file = 'datasets/91-image_x{}.h5'.format(args.scale)
    if args.eval_file is None:
        args.eval_file = 'datasets/Set5_x{}.h5'.format(args.scale)

    runs = ABLATION_RUNS if args.scale == 3 else CROSS_SCALE_RUNS

    if args.eval_files is None:
        set5  = 'datasets/Set5_x{}.h5'.format(args.scale)
        set14 = 'datasets/Set14_x{}.h5'.format(args.scale)
        if args.scale == 3:
            args.eval_files = [set5, set14]
        else:
            # Set14 x2/x4 requires prepare.py (use Set5 only unless user provides it)
            args.eval_files = [set5, set14] if os.path.exists(set14) else [set5]
            if not os.path.exists(set14):
                print('Note: {} not found, benchmark will use Set5 only.'.format(set14))
                print('      Run prepare.py on Set14 images to generate it.\n')

    python = sys.executable
    total  = len(runs)

    # Full Training
    mode = 'full ablation' if args.scale == 3 else 'baseline + full model'
    print('\n[Phase 1] Training {} variants at x{} ({})'.format(total, args.scale, mode))

    common_train = [
        python, 'train.py',
        '--train-file',  args.train_file,
        '--eval-file',   args.eval_file,
        '--outputs-dir', args.outputs_dir,
        '--scale',       str(args.scale),
        '--lr',          str(args.lr),
        '--batch-size',  str(args.batch_size),
        '--num-epochs',  str(args.num_epochs),
        '--num-workers', str(args.num_workers),
        '--seed',        str(args.seed),
    ]

    for i, run in enumerate(runs, 1):
        out_dir   = os.path.join(args.outputs_dir, run_dir_name(args.scale, run))
        best_pth  = os.path.join(out_dir, 'best.pth')

        if os.path.exists(best_pth):
            print('\n[SKIP] Run {}/{}: {} (best.pth already exists in {})'.format(
                i, total, run['label'], out_dir))
            continue

        flags = ['--loss', run['loss']]
        if run['no_augment']:   flags.append('--no-augment')
        if run['no_scheduler']: flags.append('--no-scheduler')
        if run['no_residual']:  flags.append('--no-residual')

        run_cmd(
            common_train + flags,
            'Run {}/{}: {}'.format(i, total, run['label'])
        )

    # Auto Benchmark
    if args.skip_benchmark:
        print('\n[Phase 2] Skipped. Run benchmark.py manually when ready.')
        sys.exit(0)

    print('\n[Phase 2] Running benchmark at x{}'.format(args.scale))

    run_dirs = [
        os.path.join(args.outputs_dir, run_dir_name(args.scale, run))
        for run in runs
    ]
    labels    = [run['label'] for run in runs]
    csv_name   = 'ablation_x{}.csv'.format(args.scale) if args.scale == 3 else 'comparison_x{}.csv'.format(args.scale)
    output_csv = os.path.join(args.results_dir, csv_name)

    benchmark_cmd = (
        [python, 'benchmark.py']
        + ['--run-dirs']   + run_dirs
        + ['--labels']     + labels
        + ['--eval-files'] + args.eval_files
        + ['--output', output_csv]
    )

    run_cmd(benchmark_cmd, 'Benchmark x{}: {}'.format(
        args.scale,
        ', '.join(os.path.basename(f) for f in args.eval_files)
    ))

    print('\n' + '=' * 60)
    print('Pipeline complete  (scale x{})'.format(args.scale))
    print('  Results CSV : {}'.format(output_csv))
    print('  Chart       : {}'.format(output_csv.replace('.csv', '_chart.png')))
    print('=' * 60)
