import pickle
from pathlib import Path
import orjson
import numpy as np
import shapely
import geopandas as gpd
import cv2
from turbojpeg import TurboJPEG


jpeg = TurboJPEG()


def default(obj):
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    else:
        raise ValueError


def read_json(path):
    with open(path, "rb") as f:
        contents = orjson.loads(f.read())
    return contents


def write_json(path, contents):
    with open(path, "wb") as f:
        try:
            f.write(
                orjson.dumps(
                    contents,
                    default=default,
                    option=orjson.OPT_SERIALIZE_NUMPY,
                )
            )
        except TypeError:
            # json creates empty file on bad write
            path.unlink()
            raise


def read_geojson(path):
    """
    Fast geojson read
    4x speedup over gpd.read_file
    why? due to WGS84 only?
    """
    path = str(path)
    with open(path, "rb") as f:
        features = orjson.loads(f.read())
    # TODO: support non WGS84 inputs
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    return gdf


def read_image(path, rgb=True):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: f"Read failed: {path}"
    if rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def write_image(path, im, rgb=True):
    if rgb and len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    res = cv2.imwrite(str(path), im)
    if not res: raise ValueError(f"Cannot write to {path}")


def read_jpeg(path: Path) -> np.ndarray:
    """Faster than cv2 for jpegs"""
    with open(path, "rb") as f:
        img = jpeg.decode(f.read())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_pickle(path: Path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(path: Path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res

def read_geom(path: Path):
    return read_geojson(path).loc[0, "geometry"]
