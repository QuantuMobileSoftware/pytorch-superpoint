from itertools import chain

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.windows import get_data_window
import numpy as np
from tqdm import tqdm
import rioxarray
from shapely.geometry import box
from pyproj import Transformer

from common import SatSettings, Subset
from utils.io import write_image, write_geom

class YAWindow():
    def __init__(self, top, left, bottom, right, transfrom, crs) -> None:
        self.top_px,self.left_px,self.bottom_px,self.right_px = top,left,bottom,right
        self.transform = transfrom
        self.crs = crs

    def bounds_geo(self):
        # flip here cause rastio transforms give inv points or something
        (left_crs, top_crs) = self.transform * (self.left_px, self.top_px)
        (right_crs, bottom_crs) = self.transform * (self.right_px, self.bottom_px)
        return left_crs, bottom_crs, right_crs, top_crs

    def __repr__(self) -> str:
        return f"{(self.top_px, self.left_px, self.bottom_px, self.right_px)} in {self.crs}"


def get_raster_bounds(path):
    with rio.open(path, "r") as r:
        to_wgs = Transformer.from_crs(r.crs, "epsg:4326")
        b = r.bounds
        l,b,r,t = to_wgs.transform_bounds(b.left, b.bottom, b.right, b.top)
    return box(l,b,r,t)


if __name__ == "__main__":
    ss = Subset()

    sentinel_paths = list(SatSettings.sentinel_dir.rglob("*TCI.jp2"))
    planetscope_paths = list(SatSettings.planet_dir.rglob("*.tif"))
    other_paths = list(chain(sentinel_paths, planetscope_paths))

    skysat_paths = list(SatSettings.skysat_dir.rglob("*.tif"))

    skysat_paths = [s for s in skysat_paths if s.stem not in SatSettings.bad_skysats]

    other_bounds = list(map(get_raster_bounds, other_paths))
    skysat_bounds = list(map(get_raster_bounds, skysat_paths))

    pairs = {}
    for skysat_path, skysat_bound in zip(skysat_paths, skysat_bounds):
        for other_path, other_bound in zip(other_paths, other_bounds):
            if skysat_bound.intersects(other_bound):
                others = pairs.get(skysat_path, [])
                others.append(other_path)
                pairs[skysat_path] = others

    group_num = 0
    for skysat_path, other_paths in tqdm(pairs.items()):
        with rio.open(skysat_path, "r") as skysat_ds:
            assert not skysat_ds.crs.is_geographic
            window = get_data_window(skysat_ds.read((1,2,3), masked=True))
            skysat_img = skysat_ds.read((1,2,3), window=window)
            skysat_img = reshape_as_image(skysat_img)
            skysat_transform = skysat_ds.window_transform(window)
            skysat_crs = skysat_ds.crs

        windows_skysat = []
        top, bottom = 0, SatSettings.skysat_tile_sz_px
        while top < skysat_img.shape[0]:
            left, right = 0, SatSettings.skysat_tile_sz_px
            while left < skysat_img.shape[1]:
                bounds = top, left, min(bottom, skysat_img.shape[0]), min(right, skysat_img.shape[1])
                w = YAWindow(*bounds, skysat_transform, skysat_crs)
                windows_skysat.append(w)

                left = right
                right += SatSettings.skysat_tile_sz_px
            top = bottom
            bottom += SatSettings.skysat_tile_sz_px

        for other_path in other_paths:
            stack_name = f"{skysat_path.parent.parent.stem}-{other_path.stem}"
            print(stack_name)
            otherx = rioxarray.open_rasterio(other_path)
            otherx_utm = otherx.rio.reproject(skysat_crs, esampling=Resampling.lanczos)

            for num, mw in enumerate(windows_skysat):
                h, w = mw.bottom_px - mw.top_px, mw.right_px - mw.left_px
                aspect_ratio = min(h, w) / max(h, w)

                # Skip before crops to avoid cropping empty arrays and rio raising errors
                if (aspect_ratio <= SatSettings.min_aspect_ratio):
                    continue

                skysat_crop = skysat_img[mw.top_px:mw.bottom_px, mw.left_px:mw.right_px]
                try:
                    other_crop = reshape_as_image(otherx_utm.rio.clip_box(*mw.bounds_geo()).to_numpy())
                except rioxarray.exceptions.NoDataInBounds:
                    print("No data")
                    continue
                empty_ratio_skysat = (np.logical_or.reduce(skysat_crop==0, axis=-1)).sum() / np.prod(skysat_crop.shape[:2])
                empty_ratio_other = (np.logical_or.reduce(other_crop==0, axis=-1)).sum() / np.prod(other_crop.shape[:2])
                if (empty_ratio_skysat >= SatSettings.max_empty_ratio) or (empty_ratio_other >= SatSettings.max_empty_ratio):
                    continue

                name = f"{stack_name}_{num}"
                write_image(SatSettings.compressed_skysat_dir/f"{name}.jpg", skysat_crop)
                write_image(SatSettings.compressed_other_dir/f"{name}.jpg", other_crop)
                ss.add_item(None, name, group_num)
        group_num += 1
    ss.save(SatSettings.subset_file)
