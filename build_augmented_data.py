from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.name_origin import add_name_origin_features
from src.spatial import add_spatial_features


DATA_DIR = Path("data")
DERIVED_DIR = DATA_DIR / "derived"


def load_available_data() -> pd.DataFrame:
    frames = []
    for path in [DATA_DIR / "train.csv", DATA_DIR / "test.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No data/train.csv or data/test.csv found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    all_passengers = load_available_data()

    spatial = add_spatial_features(all_passengers)
    spatial.to_csv(DERIVED_DIR / "spatial_features.csv", index=False)

    name_origin = add_name_origin_features(all_passengers)
    name_origin.to_csv(DERIVED_DIR / "name_origin_features.csv", index=False)

    print(f"Spatial features saved to: {DERIVED_DIR / 'spatial_features.csv'}")
    print(f"Name origin features saved to: {DERIVED_DIR / 'name_origin_features.csv'}")


if __name__ == "__main__":
    main()
