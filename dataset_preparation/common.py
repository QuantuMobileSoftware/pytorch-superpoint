from pathlib import Path
from dataclasses import dataclass

_COMPRESSED_DATA_DIR = Path("/home/topkech/work/sat_datasets/cross-domain-compressed")
_RAW_DATA_DIR = Path("/larg/cross-domain-raw")
SEED = 424242
SPLIT_FILE = _COMPRESSED_DATA_DIR/"split.csv"


@dataclass(frozen=True)
class OAMSettings:
    _RAW_DIR = _RAW_DATA_DIR/"openaerialmap"
    _COMP_DIR = _COMPRESSED_DATA_DIR/"openaerialmap"

    mosaic_tile_sz_px = 4096
    target_mosaic_gsd = 0.3  # m/px
    min_aspect_ratio = 0.3
    max_empty_ratio = 0.5

    mosaic_dir = _RAW_DIR/"mosaics"
    basemap_dir = _RAW_DIR/"basemaps"

    compressed_mosaic_dir = _COMP_DIR/"mosaic"
    compressed_basemap_dir = _COMP_DIR/"basemap"
    compressed_mosaic_dir.mkdir(parents=True, exist_ok=True)
    compressed_basemap_dir.mkdir(parents=True, exist_ok=True)

    subset_file = _COMP_DIR/"subset.csv"


@dataclass(frozen=True)
class MaxarSettings:
    _RAW_DIR = _RAW_DATA_DIR/"maxar"
    _COMP_DIR = _COMPRESSED_DATA_DIR/"maxar"

    maxar_tile_sz_px = 4096
    min_aspect_ratio = 0.3
    max_empty_ratio = 0.5

    maxar_dir = _RAW_DIR/"maxar"
    planet_dir = _RAW_DIR/"planet"

    compressed_maxar_dir = _COMP_DIR/"maxar"
    compressed_planet_dir = _COMP_DIR/"planet"
    compressed_maxar_dir.mkdir(parents=True, exist_ok=True)
    compressed_planet_dir.mkdir(parents=True, exist_ok=True)

    subset_file = _COMP_DIR/"subset.csv"

@dataclass(frozen=True)
class SatSettings:
    _RAW_DIR = _RAW_DATA_DIR/"satellites"
    _COMP_DIR = _COMPRESSED_DATA_DIR/"satellites"

    bad_skysats = ["20231030_065840_ssc4_u0001_visual"]

    skysat_tile_sz_px = 4096
    min_aspect_ratio = 0.3
    max_empty_ratio = 0.5

    skysat_dir = _RAW_DIR/"SkySat"
    planet_dir = _RAW_DIR/"PlanetScope"
    sentinel_dir = _RAW_DIR/"Sentinel2"

    compressed_skysat_dir = _COMP_DIR/"skysat"
    compressed_other_dir = _COMP_DIR/"other"
    valid_mask_dir = _COMP_DIR/"valid_mask"
    compressed_skysat_dir.mkdir(parents=True, exist_ok=True)
    compressed_other_dir.mkdir(parents=True, exist_ok=True)
    valid_mask_dir.mkdir(parents=True, exist_ok=True)

    subset_file = _COMP_DIR/"subset.csv"


@dataclass(frozen=True)
class FLAIRSettings:
    _RAW_DIR = _RAW_DATA_DIR/"flair"
    _COMP_DIR = _COMPRESSED_DATA_DIR/"flair"

    aerial_tile_size = 512
    img2centroid_path = _RAW_DIR/"flair-2_centroids_sp_to_patch.json"

    raw_aerial_train_dir = _RAW_DIR/"flair_aerial_train"
    raw_aerial_test_dir = _RAW_DIR/"flair_2_aerial_test"
    raw_sen_train_dir = _RAW_DIR/"flair_sen_train"
    raw_sen_test_dir = _RAW_DIR/"flair_2_sen_test"

    aerial_dir = _COMP_DIR/"aerial"
    sen_dir = _COMP_DIR/"sen"
    aerial_dir.mkdir(parents=True, exist_ok=True)
    sen_dir.mkdir(parents=True, exist_ok=True)

    subset_file = _COMP_DIR/"subset.csv"



# add flair here
# redo flair compression to output pairs

# each compression outputs a csv file with: split ([none,train,val,test] for data comes pre-split like FLAIR), stack_name, group_num (1 highres raster is 1 group, avoids tiles from the same raster and same tile with different lowres counterparts leaking between sets)


import pandas as pd
class Subset:
    def __init__(self) -> None:
        self._data = []
    def add_item(self, split, stack_name, group_num):
        self._data.append({"split": split, "stack_name": stack_name, "group_num": group_num})
    def save(self, path):
        pd.DataFrame(self._data).to_csv(path)
