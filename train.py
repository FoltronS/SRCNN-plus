import argparse
import copy
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1'],
                        help='Loss function: l1 or mse (default: l1)')
    parser.add_argument('--no-augment',   action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-scheduler', action='store_true', help='Disable LR scheduler')
    parser.add_argument('--no-residual',  action='store_true', help='Disable residual learning')
    args = parser.parse_args()

    # Auto-name subdirectory based on active flags so runs never overwrite each other
    run_name = 'x{}_{}'.format(args.scale, args.loss)
    if args.no_augment:   run_name += '_noaug'
    if args.no_scheduler: run_name += '_nosched'
    if args.no_residual:  run_name += '_nores'
    args.outputs_dir = os.path.join(args.outputs_dir, run_name)

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # Save run config so eval.py / benchmark.py can auto-detect settings
    config = {
        'scale': args.scale,
        'loss': args.loss,
        'residual': not args.no_residual,
        'augment': not args.no_augment,
        'scheduler': not args.no_scheduler,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
    }
    with open(os.path.join(args.outputs_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SRCNN(residual=not args.no_residual).to(device)
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    train_dataset = TrainDataset(args.train_file, augment=not args.no_augment)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    train_losses = []
    eval_psnrs = []

    # Resume from the latest periodic checkpoint if one exists
    start_epoch = 0
    existing = [
        f for f in os.listdir(args.outputs_dir)
        if f.startswith('epoch_') and f.endswith('.pth')
    ]
    if existing:
        last_epoch = max(int(f[len('epoch_'):-len('.pth')]) for f in existing)
        resume_path = os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(last_epoch))
        model.load_state_dict(torch.load(resume_path, map_location=device))
        start_epoch = last_epoch + 1
        print('Resumed from {} (epoch {})'.format(resume_path, last_epoch))

        # Restore training history from CSV so the final plot covers all epochs
        log_path = os.path.join(args.outputs_dir, 'train_log.csv')
        if os.path.exists(log_path):
            with open(log_path, newline='') as f:
                for row in csv.DictReader(f):
                    if int(row['epoch']) < start_epoch:
                        train_losses.append(float(row['train_loss']))
                        eval_psnrs.append(float(row['eval_psnr']))
            if eval_psnrs:
                best_psnr = max(eval_psnrs)
                best_epoch = eval_psnrs.index(best_psnr)
                best_weights = copy.deepcopy(model.state_dict())

        # Restore scheduler state so LR is correct after resume
        if not args.no_scheduler:
            for _ in range(start_epoch):
                scheduler.step()

    # Prepare CSV: fresh start writes header; resume truncates to start_epoch rows
    log_path = os.path.join(args.outputs_dir, 'train_log.csv')
    if start_epoch == 0:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'eval_psnr'])
    else:
        # Keep only rows 0..start_epoch-1, discard anything after the resume point
        rows_to_keep = [['epoch', 'train_loss', 'eval_psnr']] + [
            [ep, '{:.6f}'.format(tl), '{:.4f}'.format(ep_psnr)]
            for ep, tl, ep_psnr in zip(range(start_epoch), train_losses, eval_psnrs)
        ]
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows_to_keep)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        train_losses.append(float(epoch_losses.avg))
        eval_psnrs.append(float(epoch_psnr.avg))

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, '{:.6f}'.format(epoch_losses.avg), '{:.4f}'.format(epoch_psnr.avg)])

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        if not args.no_scheduler:
            scheduler.step()

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

    print('Training log saved to {}'.format(log_path))

    # Save training curve plot
    epochs = list(range(len(train_losses)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, train_losses, color='steelblue', linewidth=1.5)
    ax1.set_ylabel('Train Loss ({})'.format(args.loss.upper()))
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
    curve_path = os.path.join(args.outputs_dir, 'training_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print('Training curve saved to {}'.format(curve_path))
