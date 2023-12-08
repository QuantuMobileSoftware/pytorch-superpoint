import pandas as pd
from common import FLAIRSettings, MaxarSettings, OAMSettings, SatSettings, SEED, _COMPRESSED_DATA_DIR, SPLIT_FILE

def split_groups(df, frac):
    groups = pd.DataFrame(df["group_num"].value_counts()).reset_index()
    first = set(groups.sample(frac=frac, weights="count", random_state=SEED)["group_num"])
    second = set(groups["group_num"]) - first
    assert len(first.intersection(second)) == 0
    return first, second

if __name__ == "__main__":
    # OAM
    oam_ss = pd.read_csv(OAMSettings.subset_file, index_col=0)
    oam_ss["hr_file"] = oam_ss["stack_name"].apply(lambda x: f"{(OAMSettings.compressed_mosaic_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")
    oam_ss["lr_file"] = oam_ss["stack_name"].apply(lambda x: f"{(OAMSettings.compressed_basemap_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")
    trainval_g, test_g = split_groups(oam_ss, 0.8)

    trainval_ss = oam_ss.query("group_num in @trainval_g")
    train_g, val_g = split_groups(trainval_ss, 0.8)

    oam_ss.loc[oam_ss["group_num"].isin(train_g), "split"] = "train"
    oam_ss.loc[oam_ss["group_num"].isin(val_g), "split"] = "val"
    oam_ss.loc[oam_ss["group_num"].isin(test_g), "split"] = "test"
    print("OAM SPLIT")
    print(oam_ss["split"].value_counts())

    # MAXAR
    maxar_ss = pd.read_csv(MaxarSettings.subset_file, index_col=0)
    maxar_ss["hr_file"] = maxar_ss["stack_name"].apply(lambda x: f"{(MaxarSettings.compressed_maxar_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")
    maxar_ss["lr_file"] = maxar_ss["stack_name"].apply(lambda x: f"{(MaxarSettings.compressed_planet_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")

    trainval_g, test_g = split_groups(maxar_ss, 0.8)
    trainval_ss = maxar_ss.query("group_num in @trainval_g")
    train_g, val_g = split_groups(trainval_ss, 0.8)
    maxar_ss.loc[maxar_ss["group_num"].isin(train_g), "split"] = "train"
    maxar_ss.loc[maxar_ss["group_num"].isin(val_g), "split"] = "val"
    maxar_ss.loc[maxar_ss["group_num"].isin(test_g), "split"] = "test"
    print("MAXAR SPLIT")
    print(maxar_ss["split"].value_counts())

    # SAT
    sat_ss = pd.read_csv(SatSettings.subset_file, index_col=0)
    sat_ss["hr_file"] = sat_ss["stack_name"].apply(lambda x: f"{(SatSettings.compressed_skysat_dir/f'{x}.jpg').relative_to(_COMPRESSED_DATA_DIR)}")
    sat_ss["lr_file"] = sat_ss["stack_name"].apply(lambda x: f"{(SatSettings.compressed_other_dir/f'{x}.jpg').relative_to(_COMPRESSED_DATA_DIR)}")
    sat_ss["valid_mask_file"] = sat_ss["stack_name"].apply(lambda x: f"{(SatSettings.valid_mask_dir/f'{x}.npz').relative_to(_COMPRESSED_DATA_DIR)}")

    trainval_g, test_g = split_groups(sat_ss, 0.8)
    trainval_ss = sat_ss.query("group_num in @trainval_g")
    train_g, val_g = split_groups(trainval_ss, 0.8)
    sat_ss.loc[sat_ss["group_num"].isin(train_g), "split"] = "train"
    sat_ss.loc[sat_ss["group_num"].isin(val_g), "split"] = "val"
    sat_ss.loc[sat_ss["group_num"].isin(test_g), "split"] = "test"
    print("SAT SPLIT")
    print(sat_ss["split"].value_counts())

    # FLAIR
    flair_ss = pd.read_csv(FLAIRSettings.subset_file, index_col=0)
    flair_ss["hr_file"] = flair_ss["stack_name"].apply(lambda x: f"{(FLAIRSettings.aerial_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")
    flair_ss["lr_file"] = flair_ss["stack_name"].apply(lambda x: f"{(FLAIRSettings.sen_dir/x).relative_to(_COMPRESSED_DATA_DIR)}")

    trainval_ss = flair_ss.query("split == 'train'")
    train_g, val_g = split_groups(trainval_ss, 0.8)
    flair_ss.loc[flair_ss["group_num"].isin(train_g), "split"] = "train"
    flair_ss.loc[flair_ss["group_num"].isin(val_g), "split"] = "val"
    print("FLAIR SPLIT")
    print(flair_ss["split"].value_counts())

    total = pd.concat([oam_ss, maxar_ss, sat_ss, flair_ss])
    print("TOTAL SPLIT")
    print(total["split"].value_counts())
    total.to_csv(SPLIT_FILE, index=False)
