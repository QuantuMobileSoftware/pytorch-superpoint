from typing import Tuple, List, Any
import os
import warnings
from pathlib import Path

import utm
from pyproj import CRS
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rioxarray
import rasterio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from rasterio.io import DatasetReader


from superpoint.superpointnet import SuperPointNet
import logging
from settings import (
    SUPERPOINT_INPUT_SHAPE, SUPERPOINT_PATH,
    KP_KEY, DESC_KEY, GSD_KEY,
    GEOTIF_EXT, UTM_SUFFIX, WINDOW_ASPECT_RATIO,
    MIN_SATELLITE_WINDOW_SIZE, MAX_SATELLITE_WINDOW_SIZE
)

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_window_size(satellite_resolution: float, window_px_width: float) -> float:
    assert window_px_width > 0, f"window_px_width must be more than 0. Got {window_px_width}"
    window_size = satellite_resolution * window_px_width
    return window_size


class KeyPointsExtractor():
    def __init__(self,
                 superpoints_model_path: Path = SUPERPOINT_PATH,
                 satellite_windows_overlap_ratio: float = 0.):
        """
        Args:
            superpoints_model_path )
            satellite_windows_overlap_ratio (float, optional):
                Ratio of overlap between 2 neighboor windows. Defaults to 0.0.
        """
        self.overlap_ratio = satellite_windows_overlap_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.superpoint = SuperPointNet(superpoints_model_path).to(self.device).eval()

    def get_points_and_descriptors_from_img(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers SuperPoint on image

        Args:
            image (np.ndarray): image array. 1-chanel or 3-chanel (RGB)

        Returns:
            Tuple[np.ndarray, np.ndarray]: keypoints [N, 2] and descriptors (N, 256)
        """
        if len(image.shape) ==3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        original_shape = image.shape
        image = cv2.resize(image, SUPERPOINT_INPUT_SHAPE)

        img1 = image.reshape((1, 1, *SUPERPOINT_INPUT_SHAPE[::-1]))
        img1 = img1 / 255
        img1 = torch.from_numpy(img1)
        img1 = img1.float().to(self.device)
        semi, desc = self.superpoint(img1)
        outs = self.superpoint.postprocess(semi, desc)

        kp: torch.Tensor; descs: torch.Tensor
        kp, descs = outs['pts_int'][0], outs['pts_desc'][0]

        kp = kp.detach().cpu().resolve_conj().resolve_neg().numpy()
        descs = descs.detach().cpu().resolve_conj().resolve_neg().numpy()

        kp_original_scale = kp.copy()

        kp_original_scale[:, 0] = kp[:, 0] / SUPERPOINT_INPUT_SHAPE[0] * original_shape[1]
        kp_original_scale[:, 1] = kp[:, 1] / SUPERPOINT_INPUT_SHAPE[1] * original_shape[0]

        return kp, kp_original_scale, descs

    @staticmethod
    def get_raster_crs(raster_path: os.PathLike):
        """returns crs of raster from raster_path"""
        with rasterio.open(raster_path) as src:
            return src.crs

    @staticmethod
    def get_raster_transform_and_resolution(raster_path: os.PathLike):
        """returns transformation and resolution (in px/m) of raster from raster_path"""
        with rasterio.open(raster_path) as src:
            return src.transform, src.res[0]

    def _get_windows_mosaic(self, polygon: Polygon, transform: np.ndarray, window_size_px) -> List[Window]:
        """
        Generates mosaic polygons within the given polygon.

        Args:
            polygon (Polygon): The input UTM polygon in which to generate square windows.
            transform: (np.ndarray): The transformation matrix of the raster
            window_size (float): width of window in meters
            raster_resolution: (float): Raster resolution, m / px

        Returns:
            List[Window]: A list of windows as rasterio.windows.Window objects.
        """

        x_coordinates = polygon.exterior.coords.xy[0]
        y_coordinates = polygon.exterior.coords.xy[1]
        x_px, y_px = rasterio.transform.rowcol(transform, x_coordinates, y_coordinates)

        polygon_px = Polygon(list(zip(x_px, y_px)))

        minx, miny, maxx, maxy = polygon_px.bounds


        def get_bounds_vector(min_coord, max_coord, window_len, overlap_ratio):
            row_start = row_end = min_coord
            rows_bounds = []
            while row_end < max_coord:
                row_start = row_end - window_len * overlap_ratio
                row_end = row_start + window_len
                rows_bounds.append((row_start, row_end))
            return rows_bounds

        rows_bounds = get_bounds_vector(miny, maxy, window_size_px, self.overlap_ratio)

        metadata = []
        for row_bound in rows_bounds:
            col_bounds = get_bounds_vector(minx, maxx, int(window_size_px * WINDOW_ASPECT_RATIO), self.overlap_ratio)
            for col_bound in col_bounds:
                x1, x2 = col_bound
                y1, y2 = row_bound
                metadata.append({'x0': x1, 'x1': x2, 'y0': y1, 'y1': y2})

        df = pd.DataFrame(metadata)

        x0_px, y0_px = df['x0'], df['y0']
        x1_px, y1_px = df['x1'], df['y1']

        windows = []
        for x0, y0, x1, y1 in zip(x0_px, y0_px, x1_px, y1_px):
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            width = x1-x0
            height = y1-y0
            windows.append(Window(y0, x0, height, width))
        return windows

    def read_raster_by_windows(self, windows: List[Window], dataset: DatasetReader) -> Tuple[np.array, np.array]:
        """
        Crop a raster by window.
        Args:
            windows (List[Window]): List with windows used for cropping.
            dataset: (DatasetReader): rasterio dataset to read from

        Returns:
            Tuple[np.array, np.array]: cropped image and transformation
        """
        for window in windows:
            out_image = dataset.read(window=window, boundless=True)
            out_transform = rasterio.windows.transform(window, dataset.transform)

            out_image = out_image[:3, :, :]
            yield reshape_as_image(out_image), out_transform

    @staticmethod
    def convert_raster_to_utm(satellite_path: os.PathLike, satellite_aoi_path: os.PathLike) -> Tuple[str, Any]:
        """
        Converts raster to utm crs
        Args:
            satellite_path (os.PathLike): path to raster in epsg
            satellite_path (os.PathLike): path to raster aoi

        Returns:
            Tuple[str, Any]: path to new file and utm crs
        """

        satellite_path_utm = str(satellite_path).replace(GEOTIF_EXT, UTM_SUFFIX+GEOTIF_EXT)

        satellite_aoi = gpd.read_file(satellite_aoi_path).to_crs("EPSG:4326")['geometry'].iloc[0]

        lon, lat = np.array(satellite_aoi.centroid.coords).squeeze()
        _, _, utm_number, utm_letter = utm.from_latlon(latitude=lat, longitude=lon)
        utm_crs = CRS.from_dict({"proj": "utm", "zone": utm_number, "south": utm_letter not in "XWVUTSRQPN"})

        rds = rioxarray.open_rasterio(satellite_path)
        rds_utm = rds.rio.reproject(utm_crs, resampling=Resampling.lanczos)
        rds_utm.rio.to_raster(satellite_path_utm)
        return satellite_path_utm, utm_crs


    def from_satellite(self, satellite_path: os.PathLike, aoi_path: os.PathLike) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Extract keypoints and descriptors from satellite image

        Args:
            satellite_path (os.PathLike): path to satellite tile
            aoi_pat'models/superPointNet_97000_checkpoint.pth.tar'h (os.PathLike): path to shapefile with AOI

        Returns:
            Tuple[gpd.GeoDataFrame, np.ndarray]: gpd frame with keypoints in raster crs, and array with descriptors
        """
        satellite_crs = self.get_raster_crs(satellite_path)
        is_raster_in_utm = satellite_crs.to_epsg() != 4326
        if not is_raster_in_utm:
            satellite_path, satellite_crs = KeyPointsExtractor.convert_raster_to_utm(satellite_path, aoi_path)

        satellite_aoi = gpd.read_file(aoi_path)
        satellite_aoi_utm = satellite_aoi.to_crs(satellite_crs)['geometry'].iloc[0]

        area_str_format = "{:,.2f}".format(satellite_aoi_utm.area/1e6).replace(",", " ")
        logging.info(f"Got AOI. Area: {area_str_format} square km")

        satellite_transform, satellite_resolution = self.get_raster_transform_and_resolution(satellite_path)

        window_size_m = get_window_size(satellite_resolution, SUPERPOINT_INPUT_SHAPE[0])
        window_size_m = max(window_size_m, MIN_SATELLITE_WINDOW_SIZE)
        window_size_m = min(window_size_m, MAX_SATELLITE_WINDOW_SIZE)
        window_size_px = window_size_m / satellite_resolution
        windows = self._get_windows_mosaic(satellite_aoi_utm, satellite_transform, window_size_px)

        geo_points_multiwindow = []
        key_points_multiwindow = []
        descriptors_multiwindow = []
        crops_w = []
        crops_h = []
        with rasterio.open(satellite_path) as src:
            crops_iterator = self.read_raster_by_windows(windows, src)

            for image_crop, transform in tqdm(crops_iterator, total=len(windows), desc='running superpoint on windows:'):
                if image_crop.shape[0] < 1 or image_crop.shape[1] < 1:
                    continue
                keypoints_in_superpoint_scale, keypoints_original_scale, descriptors = self.get_points_and_descriptors_from_img(image_crop)

                cols = keypoints_original_scale[:, 0]
                rows = keypoints_original_scale[:, 1]

                xs, ys = rasterio.transform.xy(transform, rows, cols)
                geo_kps = [Point(x, y) for x, y in zip(xs, ys)]

                geo_points_multiwindow.extend(geo_kps)
                crops_w.extend([image_crop.shape[1]] * len(geo_kps))
                crops_h.extend([image_crop.shape[0]] * len(geo_kps))

                key_points_multiwindow.append(keypoints_in_superpoint_scale)
                descriptors_multiwindow.append(descriptors)

        descriptors_multiwindow = np.concatenate(descriptors_multiwindow, axis=0)
        key_points_multiwindow = np.concatenate(key_points_multiwindow, axis=0)

        geo_points_multiwindow = gpd.GeoDataFrame(geometry=geo_points_multiwindow, crs=satellite_crs)
        geo_points_multiwindow.to_crs("EPSG:4326", inplace=True)
        geo_points_multiwindow['src_image'] = os.path.basename(satellite_path)
        wgs_points = np.array([geo_points_multiwindow.geometry.y, geo_points_multiwindow.geometry.x]).T

        logging.info(f"{len(geo_points_multiwindow)} pts retrieved, descriptors size: {descriptors_multiwindow.shape}")

        satellite_aoi = satellite_aoi['geometry'].iloc[0]
        is_in_aoi = geo_points_multiwindow.within(satellite_aoi).values
        keypoints_descriptors = {
            KP_KEY: wgs_points[is_in_aoi],
            DESC_KEY: descriptors_multiwindow[is_in_aoi],
            GSD_KEY: window_size_m / SUPERPOINT_INPUT_SHAPE[0],
        }
        return geo_points_multiwindow[is_in_aoi], keypoints_descriptors
