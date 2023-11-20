from functools import partial

import rasterio as rio
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject
import geopandas as gpd
from tqdm import tqdm

from common import MaxarSettings
from utils.io import write_image


# def tile_pairs(lr, hr):
#     # read big raster
#     # read small raster
#     # crop both rasters to overlap only
#     # walk maxar with 4096 size, crop corresponding planet


def renorm(num, from_max, to_max): return num / from_max * to_max



if __name__ == "__main__":
    mosaics = set(map(lambda x: x.stem, MaxarSettings.maxar_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, MaxarSettings.planet_dir.glob("*.tif")))
    stacks = mosaics and basemaps
    for stack_name in tqdm(stacks):
        print(stack_name)
        maxar_path = MaxarSettings.maxar_dir/f"{stack_name}.tif"
        planet_path = MaxarSettings.planet_dir/f"{stack_name}.tif"

        with rio.open(maxar_path, "r") as maxar_ds:
            bounds = maxar_ds.bounds
            maxar_img = maxar_ds.read()
            maxar_crs = maxar_ds.crs
        aoi_polygon = Polygon([
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top)
        ])

        with rio.open(planet_path, "r") as planet_ds:
            if maxar_crs != planet_ds.crs:
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs=maxar_crs)
                aoi_gdf.to_crs(planet_ds.crs, inplace=True)
                aoi_polygon = aoi_gdf["geometry"].iloc[0]

            planet_in_bounds, _ = mask(planet_ds, [aoi_polygon], indexes=[1,2,3], crop=True, nodata=planet_ds.nodata, all_touched=True)  # 1 based index!!!

        maxar_img = reshape_as_image(maxar_img)
        planet_in_bounds = reshape_as_image(planet_in_bounds)

        renorm_lr = partial(renorm, from_max=maxar_img.shape[1], to_max=planet_in_bounds.shape[1])
        renorm_tb = partial(renorm, from_max=maxar_img.shape[0], to_max=planet_in_bounds.shape[0])

        num = 0
        left, right, top, bottom = 0, MaxarSettings.maxar_tile_sz_px, 0, MaxarSettings.maxar_tile_sz_px
        while bottom < maxar_img.shape[0]:
            while right < maxar_img.shape[1]:
                maxar_crop = maxar_img[top:bottom, left:right]

                # TODO: just int like this misaligns images. need to resample to have topright points on the same geo coord
                pt, pb, pl, pr = int(renorm_tb(top)), int(renorm_tb(bottom)), int(renorm_lr(left)), int(renorm_lr(right))
                planet_crop = planet_in_bounds[pt:pb, pl:pr]

                write_image(MaxarSettings.compressed_maxar_dir/f"{stack_name}_{num}.jpg", maxar_crop)
                write_image(MaxarSettings.compressed_planet_dir/f"{stack_name}_{num}.jpg", planet_crop)

                left = right
                right += MaxarSettings.maxar_tile_sz_px
                num += 1
            top = bottom
            bottom += MaxarSettings.maxar_tile_sz_px
