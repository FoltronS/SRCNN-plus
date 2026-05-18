# SRCNN — PyTorch Re-implementation

PyTorch re-implementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092) (Dong et al., TPAMI 2016), with improvements including data augmentation, learning rate scheduling, L1 loss, and residual learning.

---

## Setup

```bash
pip install -r requirements.txt
```

Activate the virtual environment before running any script:

```bash
source .venv/Scripts/activate     # Git Bash
.venv\Scripts\activate.bat        # Windows CMD
.venv\Scripts\Activate.ps1        # PowerShell
```

---

## Test

Download a pre-trained model into `weights/`:

| Scale | Link |
|-------|------|
| ×2 | https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=0 |
| ×3 | https://www.dropbox.com/s/zn4fdobm2kw0c58/srcnn_x3.pth?dl=0 |
| ×4 | https://www.dropbox.com/s/pd5b2ketm0oamhj/srcnn_x4.pth?dl=0 |

```bash
python test.py --weights-file weights/srcnn_x3.pth \
               --image-file data/butterfly_GT.bmp \
               --scale 3
```

Outputs are saved alongside the input image (`_bicubic_x3.bmp`, `_srcnn_x3.bmp`).

---

## Train

Download the HDF5 datasets into `datasets/`:

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0 |
| 91-image | 3 | Train | https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0 |
| 91-image | 4 | Train | https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0 |
| Set5 | 2 | Eval | https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0 |
| Set5 | 3 | Eval | https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0 |
| Set5 | 4 | Eval | https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0 |

Download both files for your chosen scale, then run:

```bash
python train.py --train-file datasets/91-image_x3.h5 \
                --eval-file  datasets/Set5_x3.h5 \
                --outputs-dir outputs \
                --scale 3 \
                --num-workers 0
```

> **Windows:** always set `--num-workers 0`.

Checkpoints are saved to `outputs/x3/`. The best epoch (highest eval PSNR) is saved as `outputs/x3/best.pth`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-file` | required | Training `.h5` file |
| `--eval-file` | required | Evaluation `.h5` file |
| `--outputs-dir` | required | Directory for checkpoints |
| `--scale` | 3 | Upscaling factor (2, 3, or 4) |
| `--lr` | 1e-4 | Learning rate |
| `--batch-size` | 16 | Batch size |
| `--num-epochs` | 400 | Number of epochs |
| `--num-workers` | 8 | DataLoader workers (set 0 on Windows) |
| `--seed` | 123 | Random seed |

---

## Prepare custom datasets (optional)

Only needed if you want to train on your own images instead of the 91-image set above.

```bash
# Training set
python prepare.py --images-dir path/to/train_images \
                  --output-path datasets/train_x3.h5 \
                  --patch-size 33 --stride 14 --scale 3

# Eval set
python prepare.py --images-dir path/to/eval_images \
                  --output-path datasets/eval_x3.h5 \
                  --scale 3 --eval
```
