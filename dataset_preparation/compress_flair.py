import numpy as np
from tqdm import tqdm
from rasterio.merge import merge
from rasterio.enums import Resampling

from utils.io import write_image, read_json

from common import FLAIRSettings, Subset


def build_tci(sat):
    """
    returned tci is in bgr format
    3558 is max saturation for TCI in L1C product. L2C max saturation of 2000 oversaturates this dataset.
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/definitions
    """
    sat = np.clip(sat[:, [0,1,2], ...], 0, 3558)
    sat = sat.astype(np.float32) / 3558
    sat = (sat * 255).astype(np.uint8)
    return sat


def calc_highres_bounds(img2centroid, superarea, raw_aerial_path):
    aerial_path = raw_aerial_path/superarea.replace("-", "/")
    aerial_images = aerial_path.glob("img/*.tif")
    img_centroids = np.array([img2centroid[im.name] for im in aerial_images])
    (miny_c, minx_c), (maxy_c, maxx_c) = img_centroids.min(axis=0), img_centroids.max(axis=0)
    return miny_c, minx_c, maxy_c, maxx_c


if __name__ == "__main__":
    img2centroid = read_json(FLAIRSettings.img2centroid_path)
    ss = Subset()

    group_num = 0
    for sendir, aerialdir in [(FLAIRSettings.raw_sen_train_dir, FLAIRSettings.raw_aerial_train_dir), (FLAIRSettings.raw_sen_test_dir, FLAIRSettings.raw_aerial_test_dir)]:
        split = "train" if sendir == FLAIRSettings.raw_sen_train_dir else "test"

        for sa_path in tqdm(list(aerialdir.glob("*/*"))):
            superarea = str(sa_path.relative_to(aerialdir)).replace("/", "-")

            images = list(sa_path.glob("img/*.tif"))
            merged, _ = merge(images, indexes=(1, 2, 3), resampling=Resampling.nearest)  # rio indexes start with 1 !!!
            merged = merged.transpose(1,2,0)
            # crop on edge tiles' centroids. It is the last known point to be (almost) pixel-to-pixel aligned between aerial and sen
            offset = FLAIRSettings.aerial_tile_size // 2
            merged = merged[offset:-offset, offset:-offset, :]

            miny, minx, maxy, maxx = calc_highres_bounds(img2centroid, superarea, aerialdir)

            # TODO: check cloud cover
            sen_path = sendir/superarea.replace("-", "/")/"sen"
            sen_mask_path = list(sen_path.glob("*masks.npy"))[0]
            mask = np.load(sen_mask_path)[:,1,...]  # only cloud mask
            cloud_ratio = (mask > 80).sum(axis=(1,2)) / np.prod(mask.shape[-2:])
            top3_cloudless = cloud_ratio.argsort()[:3]

            sen_path = list(sen_path.glob("*data.npy"))[0]
            sat = np.load(sen_path)[top3_cloudless]
            tci = build_tci(sat)
            tci = tci[:, :, miny:maxy, minx:maxx]

            for i, tci_at_time in enumerate(tci):
                stack_name = f"{superarea}_{i}"
                ss.add_item(split, stack_name, group_num)
                write_image(FLAIRSettings.aerial_dir/f"{stack_name}.jpg", merged)
                write_image(FLAIRSettings.sen_dir/f"{stack_name}.jpg", tci_at_time.transpose(1,2,0), rgb=False)
            group_num += 1
    ss.save(FLAIRSettings.subset_file)
