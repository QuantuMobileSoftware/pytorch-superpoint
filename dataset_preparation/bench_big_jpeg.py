# On my machine I get ~0.3s
# Should be ok with multiple workers
# This [https://gist.github.com/ntamvl/84d234a48fa22a83449b3c1e39db06b8] reports I got 4.2GB/s read speeds
# The image we're testing is 30MB. Most delay is likely decoding the jpeg?
# Got 8 cores. At 1 worker per core we can get 8*3=24 images/s. Expected batch is 16 images.

# Looks a bit worrying so far. Need extra time for augs and such.

import time
from pathlib import Path
from utils.io import read_jpeg

# file = Path("/home/topkech/work/sat_datasets/cross-domain-compressed/openaerialmap/mosaic/0012_oam_2021-08-27.jpg")
file = Path("/home/topkech/work/sat_datasets/cross-domain-compressed/maxar/maxar/ard_37_031133012132_2022-10-04_104001007E34C000-visual.jpg")


for i in range(100):
    start = time.time()
    big_boi = read_jpeg(file)
    taken = time.time() - start
    print(taken, big_boi.shape)
