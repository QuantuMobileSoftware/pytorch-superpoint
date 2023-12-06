from pathlib import Path
import shutil

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import cv2
import os

from matching.bruteforce import BruteForceMatching
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()


def read_jpeg(path: Path) -> np.ndarray:
    """Faster than cv2 for jpegs"""
    with open(path, "rb") as f:
        img = jpeg.decode(f.read())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}")
    matching = BruteForceMatching("/home/topkech/work/pytorch-superpoint/logs/cross_domain/checkpoints/superPointNet_10000_checkpoint.pth.tar", use_TTA=False).eval().to(device)

    ext = ".jpg"
    folder = Path("./baseline")

    shutil.rmtree(folder, ignore_errors=True)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    base_path = Path("/home/topkech/work/sat_datasets/cross-domain-compressed")
    split = pd.read_csv(base_path/"split.csv").query("split == 'test'")

    for file in tqdm(split.itertuples(), total=split.shape[0]):
        lr = read_jpeg((base_path/file.lr_file).with_suffix(ext))
        hr = cv2.resize(read_jpeg((base_path/file.hr_file).with_suffix(ext)), (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # TODO: sliding window
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)
        lr = cv2.resize(lr, (240, 320), interpolation=cv2.INTER_NEAREST)
        hr = cv2.resize(hr, (240, 320), interpolation=cv2.INTER_NEAREST)

        lr = lr.reshape((1, 1, 240, 320)) / 255
        lr = torch.tensor(lr, device=device, dtype=torch.float32)
        hr = hr.reshape((1, 1, 240, 320)) / 255
        hr = torch.tensor(hr, device=device, dtype=torch.float32)

        pred = matching({"image0": lr, "image1": hr})

        kpts0, kpts1 = pred['pts_int0'], pred['pts_int1']
        matches = pred['matches0']

        to_save_dict = {
            'kpts0': kpts0, 'kpts1': kpts1,
            'desc0': pred['pts_desc0'], 'desc1': pred['pts_desc1'],
            'matches': matches,
        }
        np.savez(folder/f"{file.stack_name}.npz", **to_save_dict)
