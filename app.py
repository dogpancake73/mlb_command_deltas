import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse

st.set_page_config(page_title="MLB Command Delta Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("command_deltas_processed.csv")
pitches = pd.read_csv("pitch_locations.csv.gz")

# -----------------------------
# Sidebar
# -----------------------------

page = st.sidebar.selectbox("Select Page", ["Leaderboard","Pitcher View","Command Delta Primer"])

min_pitches = st.sidebar.slider(
    "Minimum Pitch Count",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)

# -----------------------------
# Leaderboard
# -----------------------------

if page == "Leaderboard":

    st.title("MLB Command Delta Leaderboard")

    season = st.selectbox(
        "Season",
        sorted(df["season"].unique())
    )

    pitch = st.selectbox(
        "Pitch Type",
        sorted(df["pitch_group"].unique())
    )

    filtered = df[
        (df["season"] == season) &
        (df["pitch_group"] == pitch) &
        (df["pitch_count"] >= min_pitches)
    ]

    filtered = filtered.sort_values(
        "command_grade",
        ascending=False
    )

    st.dataframe(
        filtered[
            [
                "player_name",
                "pitch_count",
                "command_delta",
                "command_grade",
                "z_score"
            ]
        ].reset_index(drop=True)
    )

# -----------------------------
# Pitcher View
# -----------------------------

elif page == "Pitcher View":

    st.title("Pitcher Command Analysis")

    pitcher = st.selectbox(
        "Pitcher",
        sorted(df["player_name"].unique())
    )

    season = st.selectbox(
        "Season",
        sorted(df["season"].unique())
    )

    # -----------------------------
    # Command table
    # -----------------------------

    pitcher_table = df[
        (df["player_name"] == pitcher) &
        (df["season"] == season) &
        (df["pitch_count"] >= min_pitches)
    ]

    st.subheader("Command Grades")

    st.dataframe(
        pitcher_table[
            [
                "pitch_group",
                "pitch_count",
                "command_delta",
                "command_grade",
                "z_score"
            ]
        ].reset_index(drop=True)
    )

    # -----------------------------
    # Command grade trend chart
    # -----------------------------

    st.subheader("Command Grade by Pitch Type Over Time")

    trend_data = df[
        (df["player_name"] == pitcher) &
        (df["pitch_count"] >= min_pitches)
    ]

    if not trend_data.empty:

        fig, ax = plt.subplots(figsize=(8,5))

        for pitch_type in trend_data["pitch_group"].unique():

            pitch_df = trend_data[
                trend_data["pitch_group"] == pitch_type
            ].sort_values("season")

            ax.plot(
                pitch_df["season"],
                pitch_df["command_grade"],
                marker="o",
                label=pitch_type
            )

        ax.set_xlabel("Season")
        ax.set_ylabel("Command Grade")

        ax.set_ylim(20,80)

        ax.axhline(50, linestyle="--", linewidth=1)

        ax.set_xticks(sorted(df["season"].unique()))

        ax.legend()

        st.pyplot(fig)

    # -----------------------------
    # Heatmap filters
    # -----------------------------

    st.subheader("Command Heatmap")

    pitch = st.selectbox(
        "Select Pitch Type for Heatmap",
        sorted(df["pitch_group"].unique())
    )

    heatmap_data = pitches[
        (pitches["player_name"] == pitcher) &
        (pitches["pitch_group"] == pitch) &
        (pitches["season"] == season)
    ]

    if len(heatmap_data) < 100:

        st.warning("Not enough pitches for heatmap (100 required)")

    else:

        x = heatmap_data["plate_x"]
        z = heatmap_data["plate_z"]

        xy = np.vstack([x,z])

        kde = gaussian_kde(xy)

        xgrid = np.linspace(-2,2,200)
        zgrid = np.linspace(0,5,200)

        X,Z = np.meshgrid(xgrid,zgrid)

        coords = np.vstack([X.ravel(),Z.ravel()])

        density = kde(coords).reshape(X.shape)

        fig, ax = plt.subplots(figsize=(6,7))

        ax.contourf(
            X,
            Z,
            density,
            levels=20,
            cmap="inferno"
        )

        # strike zone

        ax.plot(
            [-0.83,0.83,0.83,-0.83,-0.83],
            [1.5,1.5,3.5,3.5,1.5],
            color="white",
            linewidth=2
        )

        # -----------------------------
        # 1σ ellipse
        # -----------------------------

        cov = np.cov(x,z)

        vals,vecs = np.linalg.eigh(cov)

        order = vals.argsort()[::-1]

        vals = vals[order]
        vecs = vecs[:,order]

        theta = np.degrees(
            np.arctan2(*vecs[:,0][::-1])
        )

        width,height = 2 * np.sqrt(vals)

        ellipse = Ellipse(
            xy=(np.mean(x),np.mean(z)),
            width=width,
            height=height,
            angle=theta,
            edgecolor="cyan",
            fc="None",
            lw=2
        )

        ax.add_patch(ellipse)

        ax.set_xlim(-2,2)
        ax.set_ylim(0,5)

        ax.set_xlabel("Plate X")
        ax.set_ylabel("Plate Z")

        ax.set_title(
            f"{pitch} Command Heatmap\n{pitcher} ({season})"
        )

        st.pyplot(fig)
# -----------------------------
# Command Delta Primer Page
# -----------------------------

elif page == "Command Delta Primer":

    st.title("Command Delta: A Primer")

    st.markdown("""
### What is Command Delta?

Command Delta is a metric designed to measure pitch location consistency using pitch location dispersion.

Rather than evaluating whether a pitch is in the strike zone or not, the metric measures how tightly clustered a pitcher's locations are for a given pitch type.

In other words:

How consistently can a pitcher locate a pitch?

---

### How It Works

For each pitcher and pitch type, we measure the spread of pitch locations using the standard deviation of:

* Horizontal location (`plate_x`)
* Vertical location (`plate_z`)

We combine those into a single dispersion metric:

Command Delta = √(σx² + σz²)

Lower values indicate tighter pitch clusters and better command.

---

### Converting to Scouting Grades

To make the metric easier to interpret, Command Delta is converted into a 20–80 scouting scale.

This is done by calculating a z-score within each pitch family and mapping it to:

* **80** – Elite command  
* **70** – Plus-plus command  
* **60** – Plus command  
* **50** – MLB average  
* **40** – Below average  
* **30** – Poor command  
* **20** – Very poor command  

---

### Why Pitch Families Are Used

Different pitch types naturally have different movement and command profiles.

To account for this, Command Delta compares pitchers within pitch families:

* Four-Seam Fastballs  
* Sinkers  
* Cutters  
* Sliders  
* Sweepers  
* Curveballs  
* Offspeed pitches  

This ensures that pitchers are evaluated against comparable pitch types and their shapes.

---

### Interpreting the Heatmaps

The command heatmaps show where a pitcher actually located a given pitch.

The visualizations include:

* Kernel density heatmap – shows where pitches are most frequently located
* 1 standard deviation ellipse – represents the pitcher's command footprint

A smaller ellipse indicates tighter location consistency.

---

### Data Source

All pitch data comes from Statcast via the Python library:

`pybaseball`

The analysis currently includes the 2023–2025 MLB seasons.

---

### Why This Metric Matters

Command is one of the most important and difficult pitching skills to measure.

Traditional statistics often capture results, not process.

Command Delta focuses specifically on the consistency of pitch location, providing a clearer look at a pitcher's ability to execute.

Command Deltas should not be used in isolation to evaluate command, but rather in conjunction with things like strike and zone rates as well as heat maps.               

A potential limitation of this metric is that it may assign a lower command grade to a pitch if there are multiple intended locations, so be mindful of that when using referring to Command Deltas.                

---

### Explore the Dashboard

Use the pages in the sidebar to:

* View command leaderboards
* Explore pitcher arsenals
* Analyze pitch command heatmaps
""")