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
pitches = pd.read_csv("pitch_locations.csv")

# -----------------------------
# Sidebar
# -----------------------------

page = st.sidebar.selectbox("Select Page", ["Leaderboard","Pitcher View"])

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