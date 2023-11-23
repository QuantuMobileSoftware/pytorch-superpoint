from pathlib import Path
from dataclasses import dataclass

_COMPRESSED_DATA_DIR = Path("/home/topkech/work/sat_datasets/cross-domain-compressed")
_RAW_DATA_DIR = Path("/home/topkech/work/sat_datasets/cross-domain-raw")

@dataclass(frozen=True)
class GeorefSettings:
    _RAW_DIR = _RAW_DATA_DIR/"manual_georeference"
    _COMP_DIR = _COMPRESSED_DATA_DIR/"manual_georeference"

    progress_file = _RAW_DIR/"Manual_Georeferencing_Progress.csv"
    sat_dir = _RAW_DIR/"sat_crops"
    drone_dir = _RAW_DIR/"referenced_drone"

    src2satfile = {
        "luftronix": [
            sat_dir/"S2A_MSIL1C_20221109T092211_N0400_R093_T34UGA_20221109T112315.SAFE/GRANULE/L1C_T34UGA_A038557_20221109T092343/IMG_DATA/T34UGA_20221109T092211_TCI.jp2",
            sat_dir/"SatNav_Lviv_2022-12-28_PS_psscene_visual/files/PSScene/20221228_085656_24_2475/visual/20221228_085656_24_2475_3B_Visual_modified.tif"
        ],
        "dji": [sat_dir/"djimini2_23_12_2022_large/djimini2_23_12_2022_large.tif"],
        "matrice": [sat_dir/"matrice_300_session_2/matrice_300_session_2.tif"]
    }

    compressed_sat_dir = _COMP_DIR/"sat"
    compressed_drone_dir = _COMP_DIR/"drone"
    compressed_sat_dir.mkdir(parents=True, exist_ok=True)
    compressed_drone_dir.mkdir(parents=True, exist_ok=True)


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
    compressed_skysat_dir.mkdir(parents=True, exist_ok=True)
    compressed_other_dir.mkdir(parents=True, exist_ok=True)
