import cv2
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.windows import get_data_window, Window
import numpy as np
from tqdm import tqdm
import rioxarray
from affine import Affine

from common import OAMSettings, Subset
from utils.io import write_image
from utils.geo import point_wgs2utm

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
    ss = Subset()
    mosaics = set(map(lambda x: x.stem, OAMSettings.mosaic_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, OAMSettings.basemap_dir.glob("*.tif")))
    stacks = mosaics and basemaps

    group_num = 0
    for stack_name in tqdm(stacks):
        mosaic_path = OAMSettings.mosaic_dir/f"{stack_name}.tif"
        basemap_path = OAMSettings.basemap_dir/f"{stack_name}.tif"

        with rio.open(mosaic_path, "r") as _md:
            coef = OAMSettings.target_mosaic_gsd / _md.res[0]
            h, w = int(_md.height/coef), int(_md.width/coef)
            # TODO: calc crs
            utm_crs = point_wgs2utm(*_md.lnglat())
            with WarpedVRT(_md, crs=utm_crs, warp_mem_limit=50000, warp_extras={'NUM_THREADS':7}, resampling=Resampling.lanczos, add_alpha=True) as mosaic_ds:
                window = get_data_window(mosaic_ds.read((1,2,3), out_shape=(h,w), masked=True))
                res_window = Window(window.col_off*coef, window.row_off*coef, window.width*coef, window.height*coef)
                mosaic_img = mosaic_ds.read((1,2,3,4), out_shape=(h,w), window=res_window)
                mosaic_img = reshape_as_image(mosaic_img)
                mosaic_transform = mosaic_ds.window_transform(res_window)
                mosaic_transform = Affine(OAMSettings.target_mosaic_gsd, mosaic_transform.b, mosaic_transform.c, mosaic_transform.d, -OAMSettings.target_mosaic_gsd, mosaic_transform.f)
                mosaic_crs = mosaic_ds.crs
        windows_mosaic = []

        top, bottom = 0, OAMSettings.mosaic_tile_sz_px
        while top < mosaic_img.shape[0]:
            left, right = 0, OAMSettings.mosaic_tile_sz_px
            while left < mosaic_img.shape[1]:
                bounds = top, left, min(bottom, mosaic_img.shape[0]), min(right, mosaic_img.shape[1])
                w = YAWindow(*bounds, mosaic_transform, mosaic_crs)
                windows_mosaic.append(w)

                left = right
                right += OAMSettings.mosaic_tile_sz_px
            top = bottom
            bottom += OAMSettings.mosaic_tile_sz_px

        basemapx = rioxarray.open_rasterio(basemap_path, cache=False)
        basemapx_utm = basemapx.rio.reproject(mosaic_crs, resampling=Resampling.nearest)

        for num, mw in enumerate(windows_mosaic):
            h, w = mw.bottom_px - mw.top_px, mw.right_px - mw.left_px
            aspect_ratio = min(h, w) / max(h, w)

            # Skip before crops to avoid cropping empty arrays and rio raising errors
            if (aspect_ratio <= OAMSettings.min_aspect_ratio):
                continue

            mosaic_crop = mosaic_img[mw.top_px:mw.bottom_px, mw.left_px:mw.right_px]
            try:
                basemap_crop = reshape_as_image(basemapx_utm.rio.clip_box(*mw.bounds_geo()).to_numpy())
            except rioxarray.exceptions.NoDataInBounds:
                print("No data")
                continue

            mosaic_crop = cv2.resize(mosaic_crop, (basemap_crop.shape[1], basemap_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            empty_mosaic = mosaic_crop[...,3] == 0  # empty from alpha band
            empty_basemap = np.zeros_like(empty_mosaic)  # these are all full
            empty = np.logical_or(empty_mosaic, empty_basemap)
            empty_ratio = empty.sum() / np.prod(basemap_crop.shape[:2])
            if (empty_ratio >= OAMSettings.max_empty_ratio):
                continue

            mosiac_crop = mosaic_crop * ~empty[...,np.newaxis]
            basemap_crop = basemap_crop * ~empty[...,np.newaxis]
            name = f"{stack_name}_{num}"
            write_image(OAMSettings.compressed_mosaic_dir/f"{name}.jpg", mosaic_crop)
            write_image(OAMSettings.compressed_basemap_dir/f"{name}.jpg", basemap_crop)
            np.save(OAMSettings.valid_mask_dir/f"{name}.npy", ~empty)
            ss.add_item(None, name, group_num)

        group_num += 1

    ss.save(OAMSettings.subset_file)
