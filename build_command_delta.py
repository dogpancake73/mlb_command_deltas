from pybaseball import statcast
import pandas as pd
import numpy as np

# -----------------------------
# Seasons
# -----------------------------

seasons = {
    2023: ("2023-03-30","2023-10-01"),
    2024: ("2024-03-28","2024-09-29"),
    2025: ("2025-03-27","2025-09-28")
}

# -----------------------------
# Pitch family mapping
# -----------------------------

pitch_map = {

    "FF":"Four-Seam",
    "SI":"Sinker",
    "FC":"Cutter",

    "SL":"Slider",

    "ST":"Sweeper",

    "KC":"Curveball",
    "CU":"Curveball",
    "CS":"Curveball",
    "SV":"Curveball",

    "CH":"Offspeed",
    "FS":"Offspeed",
    "FO":"Offspeed",
    "SC":"Offspeed"
}

all_results = []

# -----------------------------
# Download seasons
# -----------------------------

for year,dates in seasons.items():

    print("Downloading",year)

    data = statcast(start_dt=dates[0], end_dt=dates[1])

    df = data[
        [
            "pitch_type",
            "player_name",
            "plate_x",
            "plate_z"
        ]
    ].dropna()

    df["pitch_group"] = df["pitch_type"].replace(pitch_map)

    df = df[df["pitch_group"].isin(pitch_map.values())]

    grouped = df.groupby(["player_name","pitch_group"])

    result = grouped.agg(

        pitch_count=("plate_x","count"),
        plate_x_std=("plate_x","std"),
        plate_z_std=("plate_z","std")

    ).reset_index()

    # Minimum pitch threshold
    result = result[result["pitch_count"] >= 100]

    # -----------------------------
    # Command dispersion
    # -----------------------------

    result["command_delta"] = np.sqrt(
        result["plate_x_std"]**2 + result["plate_z_std"]**2
    )

    result["season"] = year

    # -----------------------------
    # Z-score within pitch family
    # -----------------------------

    pitch_means = result.groupby("pitch_group")["command_delta"].mean()
    pitch_stds = result.groupby("pitch_group")["command_delta"].std()

    def calc_z(row):

        std = pitch_stds[row["pitch_group"]]
        mean = pitch_means[row["pitch_group"]]

        if pd.isna(std) or std == 0:
            return 0

        return (row["command_delta"] - mean) / std

    result["z_score"] = result.apply(calc_z,axis=1)

    result["z_score"] = pd.to_numeric(result["z_score"],errors="coerce").fillna(0)

    # -----------------------------
    # Convert to 20–80 scale
    # -----------------------------

    result["command_grade"] = (50 - result["z_score"] * 10).round()

    result["command_grade"] = result["command_grade"].clip(20,80)

    all_results.append(result)

# -----------------------------
# Combine seasons
# -----------------------------

final = pd.concat(all_results)

final = final.sort_values(["season","command_grade"],ascending=[True,False])

final.to_csv("command_deltas_processed.csv",index=False)

print("Dataset complete")