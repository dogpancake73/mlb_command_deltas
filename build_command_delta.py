from pybaseball import statcast
import pandas as pd
import numpy as np

# -----------------------------
# Settings
# -----------------------------

MAX_HEATMAP_PITCHES = 250

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
heatmap_rows = []

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

    df["season"] = year
    df["pitch_group"] = df["pitch_type"].replace(pitch_map)

    df = df[df["pitch_group"].isin(pitch_map.values())]

    # -----------------------------
    # Command calculations
    # -----------------------------

    grouped = df.groupby(["player_name","pitch_group"])

    result = grouped.agg(

        pitch_count=("plate_x","count"),
        plate_x_std=("plate_x","std"),
        plate_z_std=("plate_z","std")

    ).reset_index()

    result = result[result["pitch_count"] >= 100]

    result["command_delta"] = np.sqrt(
        result["plate_x_std"]**2 +
        result["plate_z_std"]**2
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

    result["z_score"] = result.apply(calc_z,axis=1)

    result["z_score"] = pd.to_numeric(result["z_score"],errors="coerce").fillna(0)

    result["command_grade"] = (50 - result["z_score"] * 10).round()
    result["command_grade"] = result["command_grade"].clip(20,80)

    all_results.append(result)

    # -----------------------------
    # Filter pitchers for heatmaps
    # -----------------------------

    qualified = result[["player_name","pitch_group"]]

    df_heatmap = df.merge(
        qualified,
        on=["player_name","pitch_group"]
    )

    # -----------------------------
    # Downsample pitches
    # -----------------------------

    df_heatmap = (
        df_heatmap
        .groupby(["player_name","pitch_group","season"])
        .apply(
            lambda x: x.sample(
                min(len(x), MAX_HEATMAP_PITCHES),
                random_state=1
            )
        )
        .reset_index(drop=True)
    )

    df_heatmap = df_heatmap[
        [
            "player_name",
            "pitch_group",
            "plate_x",
            "plate_z",
            "season"
        ]
    ]

    heatmap_rows.append(df_heatmap)

command_df = pd.concat(all_results)
heatmap_df = pd.concat(heatmap_rows)

command_df.to_csv(
    "command_deltas_processed.csv",
    index=False
)

heatmap_df.to_csv(
    "pitch_locations.csv",
    index=False
)

print("Datasets created successfully")