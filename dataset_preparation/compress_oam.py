from functools import partial

from tqdm import tqdm
import rasterio as rio
from shapely.geometry import Polygon
import rioxarray
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
import geopandas as gpd

from common import OAMSettings
from utils.io import write_image


def renorm(num, from_max, to_max): return num / from_max * to_max

# TODO: throw away thin long strip mosaics
# TODO: split/throw away mosaics with multiple disconnected clusters
if __name__ == "__main__":
    mosaics = set(map(lambda x: x.stem, OAMSettings.mosaic_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, OAMSettings.basemap_dir.glob("*.tif")))

    oam_stacks = mosaics and basemaps
    for stack_name in tqdm(oam_stacks):
        print(f"{stack_name} started")
        mosaic_path = OAMSettings.mosaic_dir/f"{stack_name}.tif"
        basemap_path = OAMSettings.basemap_dir/f"{stack_name}.tif"

        rds = rioxarray.open_rasterio(mosaic_path)
        rds_utm = rds.rio.reproject(rds.rio.estimate_utm_crs(), resolution=OAMSettings.target_mosaic_gsd, resampling=Resampling.bilinear, nodata=0)
        left, bottom, right, top = rds_utm.rio.bounds()
        mosaic_crs = rds_utm.rio.crs
        mosaic_img = rds_utm.to_numpy()
        print(f"{stack_name} reprojected")
        print(f"{stack_name} resized to {mosaic_img.shape}")

        aoi_polygon = Polygon([
            (left, bottom),
            (right, bottom),
            (right, top),
            (left, top)
        ])

        with rio.open(basemap_path, "r") as basemap_ds:
            if mosaic_crs != basemap_ds.crs:
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs=mosaic_crs)
                aoi_gdf.to_crs(basemap_ds.crs, inplace=True)
                aoi_polygon = aoi_gdf["geometry"].iloc[0]

            raster_in_bounds, _ = mask(basemap_ds, [aoi_polygon], indexes=[1,2,3], crop=True, nodata=basemap_ds.nodata, all_touched=True)  # 1 based index!!!

        print(f"{stack_name} masked")

        mosaic_img = reshape_as_image(mosaic_img)
        raster_in_bounds = reshape_as_image(raster_in_bounds)

        renorm_lr = partial(renorm, from_max=mosaic_img.shape[1], to_max=raster_in_bounds.shape[1])
        renorm_tb = partial(renorm, from_max=mosaic_img.shape[0], to_max=raster_in_bounds.shape[0])

        num = 0
        left, right, top, bottom = 0, OAMSettings.mosaic_tile_sz_px, 0, OAMSettings.mosaic_tile_sz_px
        while top < mosaic_img.shape[0]:
            while left < mosaic_img.shape[1]:
                print(f"{stack_name} cutting n{num} b{bottom} r{right}")
                mosaic_crop = mosaic_img[top:bottom, left:right]

                # TODO: just int like this misaligns images. need to resample to have topright points on the same geo coord
                pt, pb, pl, pr = int(renorm_tb(top)), int(renorm_tb(bottom)), int(renorm_lr(left)), int(renorm_lr(right))
                raster_crop = raster_in_bounds[pt:pb, pl:pr]

                write_image(OAMSettings.compressed_mosaic_dir/f"{stack_name}_{num}.jpg", mosaic_crop)
                write_image(OAMSettings.compressed_basemap_dir/f"{stack_name}_{num}.jpg", raster_crop)

                left = right
                right += OAMSettings.mosaic_tile_sz_px
                num += 1
            top = bottom
            bottom += OAMSettings.mosaic_tile_sz_px

        if (top == 0) or (left == 0): print(f"{stack_name} SKIPPED")
