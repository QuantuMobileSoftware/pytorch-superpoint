import pandas as pd
import os
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
from tqdm import tqdm
from shapely.geometry import Polygon
import geopandas as gpd

from common import GeorefSettings
from utils.io import write_image


def assign_area_by_dataset(dataset_name):
    if '-frames' in dataset_name:
        return 'luftronix'
    elif 'djimini' in dataset_name and dataset_name != 'djimini2_04_12_2022':  # djimini2_04_12_2022 is in other place, not needed
        return 'dji'
    elif 'matrice' in dataset_name:
        return 'matrice'
    else:
        return None


# TODO: just rglob and join?
def find_file(root_dir, filename):
    if not filename.endswith('tif'):
        filename = filename + '.tif'

    root_dir = os.path.abspath(root_dir)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if 'modified' not in fname:
                continue
            check_dupl = fname.replace('_modified', '_modified_modified')
            if '_modified' in fname and os.path.exists(os.path.join(dirpath, check_dupl)):
                fname = check_dupl

            if fname.replace('_modified', '') == filename:
                return os.path.join(dirpath, fname)
    return None


if __name__ == "__main__":
    georef_progress = pd.read_csv(GeorefSettings.progress_file)
    georef_progress = georef_progress.query("Progress & Passed")
    georef_progress["area"] = georef_progress["Dataset"].apply(assign_area_by_dataset)
    georef_progress = georef_progress[georef_progress["area"].notna()]
    georef_progress["drone_path"] = georef_progress.apply(lambda x: find_file(GeorefSettings.drone_dir/x["Dataset"], x["Frame"]), axis=1)
    assert georef_progress[georef_progress["drone_path"].isna()].empty

    for area_name, sat_paths in tqdm(GeorefSettings.src2satfile.items(), desc="area"):
        georef_progress_area = georef_progress.query("area == @area_name")
        for path in tqdm(sat_paths, desc="sat", leave=False):
            with rasterio.open(path, "r") as sat_ds:
                for _, row in tqdm(georef_progress_area.iterrows(), desc="drone", leave=False):
                    pair_name = f"{row['area']}_{row['Dataset']}_{row['Frame'][:-4]}"
                    sat_dir = GeorefSettings.compressed_sat_dir/pair_name
                    sat_dir.mkdir(parents=True, exist_ok=True)

                    with rasterio.open(row["drone_path"], "r") as drone_ds:
                        bounds = drone_ds.bounds
                        drone_img = drone_ds.read().squeeze()

                    aoi_polygon = Polygon([
                        (bounds.left, bounds.bottom),
                        (bounds.right, bounds.bottom),
                        (bounds.right, bounds.top),
                        (bounds.left, bounds.top)
                    ])
                    if drone_ds.crs != sat_ds.crs:
                        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs=drone_ds.crs)
                        aoi_gdf.to_crs(sat_ds.crs, inplace=True)
                        aoi_polygon = aoi_gdf["geometry"].iloc[0]

                    raster_crop, _ = mask(sat_ds, [aoi_polygon], indexes=[1,2,3], crop=True, nodata=sat_ds.nodata, all_touched=True)  # 1 based index!!!
                    raster_crop = reshape_as_image(raster_crop)

                    write_image(GeorefSettings.compressed_drone_dir/f"{pair_name}.jpg", drone_img)
                    write_image(sat_dir/f"{path.stem}.jpg", raster_crop)
