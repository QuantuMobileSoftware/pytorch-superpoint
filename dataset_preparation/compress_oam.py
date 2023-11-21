from functools import partial

import rasterio as rio
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject
from rasterio.windows import get_data_window, Window
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import rioxarray
from pyproj import Transformer

from common import OAMSettings
from utils.io import write_image

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


if __name__ == "__main__":
    mosaics = set(map(lambda x: x.stem, OAMSettings.mosaic_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, OAMSettings.basemap_dir.glob("*.tif")))
    stacks = mosaics and basemaps
    for stack_name in tqdm(stacks):
        print(stack_name)
        mosaic_path = OAMSettings.mosaic_dir/f"{stack_name}.tif"
        basemap_path = OAMSettings.basemap_dir/f"{stack_name}.tif"
        mosaic = rioxarray.open_rasterio(mosaic_path)
        mosaic_utm = mosaic.rio.reproject(mosaic.rio.estimate_utm_crs(), resolution=OAMSettings.target_mosaic_gsd, resampling=Resampling.lanczos, nodata=0)
        with rio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff', count=int(mosaic_utm.rio.count), width=int(mosaic_utm.rio.width), height=int(mosaic_utm.rio.height),
                dtype=str(mosaic_utm.dtype), crs=mosaic_utm.rio.crs, transform=mosaic_utm.rio.transform(recalc=True), nodata=mosaic_utm.rio.nodata
                ) as mosaic_ds:
                mosaic_ds.write(mosaic_utm.to_numpy())

                window = get_data_window(mosaic_ds.read((1,2,3), masked=True))
                mosaic_img = mosaic_ds.read((1,2,3), window=window)
                mosaic_img = reshape_as_image(mosaic_img)
                mosaic_transform = mosaic_ds.window_transform(window)
                mosaic_crs = mosaic_ds.crs

        windows_mosaic = []

        left, right, top, bottom = 0, OAMSettings.mosaic_tile_sz_px, 0, OAMSettings.mosaic_tile_sz_px
        while top < mosaic_img.shape[0]:
            while left < mosaic_img.shape[1]:
                bounds = top, left, min(bottom, mosaic_img.shape[0]), min(right, mosaic_img.shape[1])
                w = YAWindow(*bounds, mosaic_transform, mosaic_crs)
                windows_mosaic.append(w)

                left = right
                right += OAMSettings.mosaic_tile_sz_px
            top = bottom
            bottom += OAMSettings.mosaic_tile_sz_px

        basemapx = rioxarray.open_rasterio(basemap_path)
        basemapx_utm = basemapx.rio.reproject(mosaic_crs, esampling=Resampling.lanczos)

        for num, mw in enumerate(windows_mosaic):
            h, w = mw.bottom_px - mw.top_px, mw.right_px - mw.left_px
            aspect_ratio = min(h, w) / max(h, w)

            # Skip before crops to avoid cropping empty arrays and rio raising errors
            if (aspect_ratio <= OAMSettings.min_aspect_ratio):
                continue

            mosaic_crop = mosaic_img[mw.top_px:mw.bottom_px, mw.left_px:mw.right_px]
            basemap_crop = reshape_as_image(basemapx_utm.rio.clip_box(*mw.bounds_geo()).to_numpy())
            empty_ratio = (np.logical_or.reduce(mosaic_crop==0, axis=-1)).sum() / np.prod(mosaic_crop.shape[:2])

            if (empty_ratio >= OAMSettings.max_empty_ratio):
                continue

            write_image(OAMSettings.compressed_mosaic_dir/f"{stack_name}_{num}.jpg", mosaic_crop)
            write_image(OAMSettings.compressed_basemap_dir/f"{stack_name}_{num}.jpg", basemap_crop)
