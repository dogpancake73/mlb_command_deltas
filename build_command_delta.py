from pybaseball import statcast
import pandas as pd
import numpy as np
from datetime import date

# -----------------------------
# Settings
# -----------------------------

# No heatmap settings needed

# -----------------------------
# Seasons (2026 auto-updates)
# -----------------------------

seasons = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", str(date.today()))
}

# -----------------------------
# Pitch family mapping
# -----------------------------

pitch_map = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "FC": "Cutter",

    "SL": "Slider",
    "ST": "Sweeper",

    "KC": "Curveball",
    "CU": "Curveball",
    "CS": "Curveball",
    "SV": "Curveball",

    "CH": "Offspeed",
    "FS": "Offspeed",
    "FO": "Offspeed",
    "SC": "Offspeed"
}

all_results = []

# -----------------------------
# Main loop
# -----------------------------

for year, dates in seasons.items():

    print(f"Downloading {year}...")

    data = statcast(start_dt=dates[0], end_dt=dates[1])

    df = data[
        ["pitch_type", "player_name", "plate_x", "plate_z"]
    ].dropna()

    df["season"] = year
    df["pitch_group"] = df["pitch_type"].replace(pitch_map)

    df = df[df["pitch_group"].isin(pitch_map.values())]

    # -----------------------------
    # Command calculations
    # -----------------------------

    grouped = df.groupby(["player_name", "pitch_group"], observed=True)

    result = grouped.agg(
        pitch_count=("plate_x", "count"),
        plate_x_std=("plate_x", "std"),
        plate_z_std=("plate_z", "std")
    ).reset_index()

    result = result[result["pitch_count"] >= 100]

    result["command_delta"] = np.sqrt(
        result["plate_x_std"] ** 2 +
        result["plate_z_std"] ** 2
    )

    result["season"] = year

    pitch_means = result.groupby("pitch_group")["command_delta"].mean()
    pitch_stds = result.groupby("pitch_group")["command_delta"].std()

    def calc_z(row):
        std = pitch_stds[row["pitch_group"]]
        mean = pitch_means[row["pitch_group"]]
        if pd.isna(std) or std == 0:
            return 0
        return (row["command_delta"] - mean) / std

    result["z_score"] = result.apply(calc_z, axis=1)
    result["z_score"] = pd.to_numeric(result["z_score"], errors="coerce").fillna(0)

    result["command_grade"] = (50 - result["z_score"] * 10).round()
    result["command_grade"] = result["command_grade"].clip(20, 80)

    all_results.append(result)

# -----------------------------
# Output
# -----------------------------

command_df = pd.concat(all_results, ignore_index=True)

command_df.to_csv("command_deltas_processed.csv", index=False)

print("Command dataset created successfully (no heatmaps)")