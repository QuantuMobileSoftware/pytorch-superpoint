import rasterio as rio
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject
import geopandas as gpd

from common import OAMSettings
from utils.io import write_image


if __name__ == "__main__":
    mosaics = set(map(lambda x: x.stem, OAMSettings.mosaic_dir.glob("*.tif")))
    basemaps = set(map(lambda x: x.stem, OAMSettings.basemap_dir.glob("*.tif")))

    oam_stacks = mosaics and basemaps
    for stack_name in oam_stacks:
        mosaic_path = OAMSettings.mosaic_dir/f"{stack_name}.tif"
        basemap_path = OAMSettings.basemap_dir/f"{stack_name}.tif"

        with rio.open(mosaic_path, "r") as mosaic_ds:
            bounds = mosaic_ds.bounds
            # TODO: scale based on gsds, not straight /4
            # TODO: throw away thin long strip mosaics
            # TODO: split/throw away mosaics with multiple disconnected clusters
            mosaic_img = mosaic_ds.read(
                out_shape=(mosaic_ds.count, int(mosaic_ds.height / 4), int(mosaic_ds.width / 4)),
                resampling=Resampling.lanczos
            )
            mosaic_crs = mosaic_ds.crs
        aoi_polygon = Polygon([
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top)
        ])

        with rio.open(basemap_path, "r") as basemap_ds:
            if mosaic_crs != basemap_ds.crs:
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs=mosaic_crs)
                aoi_gdf.to_crs(basemap_ds.crs, inplace=True)
                aoi_polygon = aoi_gdf["geometry"].iloc[0]

            raster_crop, _ = mask(basemap_ds, [aoi_polygon], indexes=[1,2,3], crop=True, nodata=basemap_ds.nodata, all_touched=True)  # 1 based index!!!

        mosaic_img = reshape_as_image(mosaic_img)
        raster_crop = reshape_as_image(raster_crop)

        print(mosaic_img.shape)
        write_image(OAMSettings.compressed_mosaic_dir/f"{stack_name}.jpg", mosaic_img)
        write_image(OAMSettings.compressed_basemap_dir/f"{stack_name}.jpg", raster_crop)
