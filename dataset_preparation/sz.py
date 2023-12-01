import matplotlib.pyplot as plt
import pandas as pd
from common import SPLIT_FILE, _COMPRESSED_DATA_DIR
from utils.io import read_jpeg



if __name__ == "__main__":
    split = pd.read_csv(SPLIT_FILE).query("split == 'train'")
    hs = []
    ws = []
    for ex in split.itertuples():
        f = (_COMPRESSED_DATA_DIR/ex.lr_file).with_suffix(".jpg")
        im = read_jpeg(f)
        hs.append(im.shape[0])
        ws.append(im.shape[1])

    plt.figure(figsize=(10,10))
    # plt.hist2d(ws, hs, bins=100, vmin=0, vmax=100)
    plt.scatter(ws, hs, s=1)
    plt.savefig("sizes.png")
