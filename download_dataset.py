"""
Prepares all datasets needed for SRCNN.

Priority order for each HDF5 file:
  1. Skip if already exists in datasets/
  2. Generate from local images in data/ using prepare.py
  3. Fall back to internet download (91-image and Set5 only; Set14 has no Dropbox mirror)

Local image directories expected:
  data/Train/          →  91-image_x{scale}.h5  (train mode)
  data/Test/Set5/      →  Set5_x{scale}.h5       (eval mode)
  data/Test/Set14/     →  Set14_x{scale}.h5      (eval mode)

Usage:
    python download_dataset.py                        # all scales, all steps
    python download_dataset.py --scales 3             # scale 3 only
    python download_dataset.py --no-weights           # skip weight download
"""

import argparse
import os
import subprocess
import sys
import urllib.request

from tqdm import tqdm

# Internet fallback (91-image train + Set5 eval only; no Dropbox mirror for Set14)
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

DATASETS_DIR   = "datasets"
WEIGHTS_DIR    = "weights"
LOCAL_TRAIN    = os.path.join("data", "Train")
LOCAL_SET5     = os.path.join("data", "Test", "Set5")
LOCAL_SET14    = os.path.join("data", "Test", "Set14")


def download_file(url, dest):
    filename = os.path.basename(dest)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
              miniters=1, desc="  {}".format(filename)) as t:
        def hook(count, block_size, total_size):
            if total_size > 0:
                t.total = total_size
            t.update(count * block_size - t.n)
        urllib.request.urlretrieve(url, dest, reporthook=hook)


def run_prepare(images_dir, output_path, scale, eval_mode):
    """Run prepare.py. Returns True on success, False on failure."""
    cmd = [
        sys.executable, "prepare.py",
        "--images-dir",  images_dir,
        "--output-path", output_path,
        "--scale",       str(scale),
    ]
    if eval_mode:
        cmd.append("--eval")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  [FAILED] prepare.py exited with code {}".format(result.returncode))
        # Remove partial output so a retry works cleanly
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    return True


def ensure_hdf5(filename, dest, scale, local_images_dir, eval_mode, fallback_url):
    """
    Ensure `dest` exists. Priority:
      1. Skip if already exists.
      2. Generate from local_images_dir via prepare.py (if dir exists).
      3. Download from fallback_url (if provided).
    """
    if os.path.exists(dest):
        print("  [skip] {} already exists".format(filename))
        return

    # Try local images first
    if local_images_dir and os.path.isdir(local_images_dir):
        print("  Generating {} from {} (scale x{}) ...".format(filename, local_images_dir, scale))
        if run_prepare(local_images_dir, dest, scale, eval_mode):
            return
        print("  prepare.py failed, falling back to internet download.")

    # Fall back to internet
    if fallback_url:
        print("  Downloading {} from internet ...".format(filename))
        download_file(fallback_url, dest)
    else:
        print("  [SKIP] No local images found at '{}' and no internet mirror available.".format(
            local_images_dir or "(none)"))


def main():
    parser = argparse.ArgumentParser(description="Prepare SRCNN datasets and weights.")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4],
                        choices=[2, 3, 4], metavar="N",
                        help="Upscaling factors to prepare (default: 2 3 4)")
    parser.add_argument("--no-datasets", action="store_true",
                        help="Skip HDF5 dataset preparation entirely")
    parser.add_argument("--no-weights",  action="store_true",
                        help="Skip pre-trained weight download")
    args = parser.parse_args()

    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR,  exist_ok=True)

    # HDF5 datasets
    if not args.no_datasets:
        for scale in args.scales:
            print("\n=== Scale x{} ===".format(scale))

            train_filename, train_url = DATASET_URLS[scale]["train"]
            eval_filename,  eval_url  = DATASET_URLS[scale]["eval"]

            # 91-image train set
            ensure_hdf5(
                filename       = train_filename,
                dest           = os.path.join(DATASETS_DIR, train_filename),
                scale          = scale,
                local_images_dir = LOCAL_TRAIN,
                eval_mode      = False,
                fallback_url   = train_url,
            )

            # Set5 eval set
            ensure_hdf5(
                filename       = eval_filename,
                dest           = os.path.join(DATASETS_DIR, eval_filename),
                scale          = scale,
                local_images_dir = LOCAL_SET5,
                eval_mode      = True,
                fallback_url   = eval_url,
            )

            # Set14 eval set (no Dropbox mirror, local only)
            set14_filename = "Set14_x{}.h5".format(scale)
            ensure_hdf5(
                filename       = set14_filename,
                dest           = os.path.join(DATASETS_DIR, set14_filename),
                scale          = scale,
                local_images_dir = LOCAL_SET14,
                eval_mode      = True,
                fallback_url   = None,
            )

    # Pre-trained weights
    if not args.no_weights:
        print("\n=== Pre-trained weights ===")
        for scale in args.scales:
            filename, url = WEIGHTS_URLS[scale]
            dest = os.path.join(WEIGHTS_DIR, filename)
            if os.path.exists(dest):
                print("  [skip] {} already exists".format(filename))
            else:
                print("  Downloading {} ...".format(filename))
                download_file(url, dest)

    print("\nDone.")
    print("  Datasets : {}/".format(DATASETS_DIR))
    print("  Weights  : {}/".format(WEIGHTS_DIR))


if __name__ == "__main__":
    main()
