import rasterio as rio
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.windows import get_data_window
import numpy as np
from tqdm import tqdm
import rioxarray

from common import MaxarSettings
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
    mosaics = set(map(lambda x: x.stem, MaxarSettings.maxar_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, MaxarSettings.planet_dir.glob("*.tif")))
    stacks = mosaics and basemaps
    for stack_name in tqdm(stacks):
        print(stack_name)
        maxar_path = MaxarSettings.maxar_dir/f"{stack_name}.tif"
        planet_path = MaxarSettings.planet_dir/f"{stack_name}.tif"

        with rio.open(maxar_path, "r") as maxar_ds:
            window = get_data_window(maxar_ds.read((1,2,3), masked=True))
            maxar_img = maxar_ds.read((1,2,3), window=window)
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

            empty_ratio_maxar = (np.logical_or.reduce(maxar_crop==0, axis=-1)).sum() / np.prod(maxar_crop.shape[:2])
            empty_ratio_planet = (np.logical_or.reduce(planet_crop==0, axis=-1)).sum() / np.prod(planet_crop.shape[:2])
            if (empty_ratio_maxar >= MaxarSettings.max_empty_ratio) or (empty_ratio_planet >= MaxarSettings.max_empty_ratio):
                continue

            write_image(MaxarSettings.compressed_maxar_dir/f"{stack_name}_{num}.jpg", maxar_crop)
            write_image(MaxarSettings.compressed_planet_dir/f"{stack_name}_{num}.jpg", planet_crop)
