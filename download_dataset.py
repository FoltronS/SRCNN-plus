"""
Downloads all datasets and pre-trained weights needed for SRCNN.

Usage:
    python download_dataset.py               # download all scales (2, 3, 4)
    python download_dataset.py --scales 3    # only scale 3
    python download_dataset.py --no-weights  # skip weight download
    python download_dataset.py --no-datasets # skip dataset download
"""

import argparse
import os
import urllib.request

from tqdm import tqdm

DATASET_URLS = {
    2: {
        "train": ("91-image_x2.h5", "https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=1"),
        "eval":  ("Set5_x2.h5",     "https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=1"),
    },
    3: {
        "train": ("91-image_x3.h5", "https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=1"),
        "eval":  ("Set5_x3.h5",     "https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=1"),
    },
    4: {
        "train": ("91-image_x4.h5", "https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=1"),
        "eval":  ("Set5_x4.h5",     "https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=1"),
    },
}

WEIGHTS_URLS = {
    2: ("srcnn_x2.pth", "https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=1"),
    3: ("srcnn_x3.pth", "https://www.dropbox.com/s/zn4fdobm2kw0c58/srcnn_x3.pth?dl=1"),
    4: ("srcnn_x4.pth", "https://www.dropbox.com/s/pd5b2ketm0oamhj/srcnn_x4.pth?dl=1"),
}

DATASETS_DIR = "datasets"
WEIGHTS_DIR  = "weights"

def download_file(url, dest):
    filename = os.path.basename(dest)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
              miniters=1, desc=f"  {filename}") as t:
        def hook(count, block_size, total_size):
            if total_size > 0:
                t.total = total_size
            t.update(count * block_size - t.n)
        urllib.request.urlretrieve(url, dest, reporthook=hook)

def main():
    parser = argparse.ArgumentParser(description="Download SRCNN datasets and weights.")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4],
                        choices=[2, 3, 4], metavar="N",
                        help="Upscaling factors to download (default: 2 3 4)")
    parser.add_argument("--no-datasets", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--no-weights",  action="store_true",
                        help="Skip weight download")
    args = parser.parse_args()

    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR,  exist_ok=True)

    if not args.no_datasets:
        print("\n=== Downloading datasets ===")
        for scale in args.scales:
            for split, (filename, url) in DATASET_URLS[scale].items():
                dest = os.path.join(DATASETS_DIR, filename)
                if os.path.exists(dest):
                    print(f"  [skip] {filename} already exists")
                else:
                    print(f"  Downloading {filename} (x{scale} {split}) ...")
                    download_file(url, dest)

    if not args.no_weights:
        print("\n=== Downloading pre-trained weights ===")
        for scale in args.scales:
            filename, url = WEIGHTS_URLS[scale]
            dest = os.path.join(WEIGHTS_DIR, filename)
            if os.path.exists(dest):
                print(f"  [skip] {filename} already exists")
            else:
                print(f"  Downloading {filename} ...")
                download_file(url, dest)

    print("\nDone.")
    print(f"  Datasets : {DATASETS_DIR}/")
    print(f"  Weights  : {WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()
