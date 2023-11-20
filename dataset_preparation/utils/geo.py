import math
import warnings
import geopandas as gpd
from shapely.geometry import MultiPoint


def point_wgs2utm(lon, lat):
    utm_band = "{:02d}".format(math.floor((lon + 180) / 6) % 60 + 1)
    crs = "epsg:326" + utm_band if lat >= 0 else "epsg:327" + utm_band
    return crs


def wgs2utm(gdf: gpd.GeoDataFrame):
    """
    10x faster than GeoDataFrame.estimate_utm_crs()
    """
    if gdf.crs.to_epsg() != 4326:
        raise RuntimeError(
            "Converting GeoDataFrame projection from wgs to utm. "
            "Input GeoDataFrame is not in wgs. crs is {}".format(gdf.crs.to_epsg())
        )
    # We get a centroid in ellipsoidal coordinates
    # this causes a UserWarning, because this is inaccurate
    # for our purpose it is good enough, so we ignore it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_centeroid = MultiPoint(gdf.geometry.centroid.tolist()).centroid
    crs = point_wgs2utm(gdf_centeroid.x, gdf_centeroid.y)
    return gdf.to_crs(crs=crs)
