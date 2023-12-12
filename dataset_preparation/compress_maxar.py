import cv2
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.windows import get_data_window
import numpy as np
from tqdm import tqdm
import rioxarray

from common import MaxarSettings, Subset
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
    ss = Subset()

    mosaics = set(map(lambda x: x.stem, MaxarSettings.maxar_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, MaxarSettings.planet_dir.glob("*.tif")))
    stacks = mosaics and basemaps

    group_num = 0
    for stack_name in tqdm(stacks):
        print(stack_name)
        maxar_path = MaxarSettings.maxar_dir/f"{stack_name}.tif"
        planet_path = MaxarSettings.planet_dir/f"{stack_name}.tif"

        with rio.open(maxar_path, "r", add_alpha=True) as maxar_ds:
            window = get_data_window(maxar_ds.read((1,2,3), masked=True))
            maxar_img = maxar_ds.read((1,2,3), window=window, masked=True)
            mask = np.logical_or.reduce(maxar_img.mask == True, axis=0, keepdims=True)
            maxar_img = np.concatenate([maxar_img.base, mask], axis=0)
            maxar_img = reshape_as_image(maxar_img)
            maxar_transform = maxar_ds.window_transform(window)
            maxar_crs = maxar_ds.crs

        windows_maxar = []
        top, bottom = 0, MaxarSettings.maxar_tile_sz_px
        while top < maxar_img.shape[0]:
            left, right = 0, MaxarSettings.maxar_tile_sz_px
            while left < maxar_img.shape[1]:
                bounds = top, left, min(bottom, maxar_img.shape[0]), min(right, maxar_img.shape[1])
                w = YAWindow(*bounds, maxar_transform, maxar_crs)
                windows_maxar.append(w)

                left = right
                right += MaxarSettings.maxar_tile_sz_px
            top = bottom
            bottom += MaxarSettings.maxar_tile_sz_px

        planetx = rioxarray.open_rasterio(planet_path)
        planetx_utm = planetx.rio.reproject(maxar_crs, esampling=Resampling.lanczos)

        for num, mw in enumerate(windows_maxar):
            h, w = mw.bottom_px - mw.top_px, mw.right_px - mw.left_px
            aspect_ratio = min(h, w) / max(h, w)

            # Skip before crops to avoid cropping empty arrays and rio raising errors
            if (aspect_ratio <= MaxarSettings.min_aspect_ratio):
                continue

            maxar_crop = maxar_img[mw.top_px:mw.bottom_px, mw.left_px:mw.right_px]
            try:
                planet_crop = reshape_as_image(planetx_utm.rio.clip_box(*mw.bounds_geo()).to_numpy())
            except rioxarray.exceptions.NoDataInBounds:
                print("No data")
                continue

            maxar_crop = cv2.resize(maxar_crop, (planet_crop.shape[1], planet_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            empty_maxar = maxar_crop[...,3]
            empty_planet = np.zeros_like(empty_maxar)
            empty = np.logical_or(empty_maxar, empty_planet)
            empty_ratio = empty.sum() / np.prod(planet_crop.shape[:2])
            if (empty_ratio >= MaxarSettings.max_empty_ratio):
                continue

            maxar_crop = maxar_crop * ~empty[...,np.newaxis]
            planet_crop = planet_crop* ~empty[...,np.newaxis]
            name = f"{stack_name}_{num}"
            write_image(MaxarSettings.compressed_maxar_dir/f"{name}.jpg", maxar_crop)
            write_image(MaxarSettings.compressed_planet_dir/f"{name}.jpg", planet_crop)
            np.save(MaxarSettings.valid_mask_dir/f"{name}.npy", ~empty)
            ss.add_item(None, name, group_num)
        group_num += 1

    ss.save(MaxarSettings.subset_file)
