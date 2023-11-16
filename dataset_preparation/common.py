from pathlib import Path
from dataclasses import dataclass

_COMPRESSED_DATA_DIR = Path("/home/topkech/work/sat_datasets/cross-domain-compressed")
_RAW_DATA_DIR = Path("/home/topkech/work/sat_datasets/cross-domain-raw")

@dataclass(frozen=True)
class GeorefSettings:
    _GEOREF_DIR = _RAW_DATA_DIR/"manual_georeference"
    _COMP_GEOREF_DIR = _COMPRESSED_DATA_DIR/"manual_georeference"

    progress_file = _GEOREF_DIR/"Manual_Georeferencing_Progress.csv"
    sat_dir = _GEOREF_DIR/"sat_crops"
    drone_dir = _GEOREF_DIR/"referenced_drone"

    src2satfile = {
        "luftronix": [
            sat_dir/"S2A_MSIL1C_20221109T092211_N0400_R093_T34UGA_20221109T112315.SAFE/GRANULE/L1C_T34UGA_A038557_20221109T092343/IMG_DATA/T34UGA_20221109T092211_TCI.jp2",
            sat_dir/"SatNav_Lviv_2022-12-28_PS_psscene_visual/files/PSScene/20221228_085656_24_2475/visual/20221228_085656_24_2475_3B_Visual_modified.tif"
        ],
        "dji": [sat_dir/"djimini2_23_12_2022_large/djimini2_23_12_2022_large.tif"],
        "matrice": [sat_dir/"matrice_300_session_2/matrice_300_session_2.tif"]
    }

    compressed_sat_dir = _COMP_GEOREF_DIR/"sat"
    compressed_drone_dir = _COMP_GEOREF_DIR/"drone"
    compressed_sat_dir.mkdir(parents=True, exist_ok=True)
    compressed_drone_dir.mkdir(parents=True, exist_ok=True)
