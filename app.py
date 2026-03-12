import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLB Command Delta Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("command_deltas_processed.csv")

df = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------

page = st.sidebar.selectbox(
    "Select Page",
    ["Leaderboard", "Pitcher View"]
)

min_pitches = st.sidebar.slider(
    "Minimum Pitch Count",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)

# -----------------------------
# Leaderboard page
# -----------------------------

if page == "Leaderboard":

    st.title("MLB Command Delta Leaderboard")

    season = st.selectbox(
        "Season",
        sorted(df["season"].unique())
    )

    pitch = st.selectbox(
        "Pitch Family",
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
                "z_score",
                "command_grade"
            ]
        ].reset_index(drop=True)
    )

# -----------------------------
# Pitcher view page
# -----------------------------

elif page == "Pitcher View":

    st.title("Pitcher Command Profiles")

    pitchers = df[df["pitch_count"] >= min_pitches]["player_name"].unique()

    pitcher = st.selectbox(
        "Pitcher",
        np.sort(pitchers)
    )

    pitcher_df = df[
        (df["player_name"] == pitcher) &
        (df["pitch_count"] >= min_pitches)
    ].sort_values(["season", "pitch_group"])

    st.dataframe(
        pitcher_df[
            [
                "season",
                "pitch_group",
                "pitch_count",
                "command_delta",
                "z_score",
                "command_grade"
            ]
        ].reset_index(drop=True)
    )

    st.subheader("Command Grade by Pitch Family")

    fig, ax = plt.subplots()

    # Plot each pitch family
    for pitch in pitcher_df["pitch_group"].unique():

        subset = pitcher_df[pitcher_df["pitch_group"] == pitch]

        ax.plot(
            subset["season"],
            subset["command_grade"],
            marker="o",
            label=pitch
        )

    # Force full year labels
    seasons = sorted(pitcher_df["season"].unique())
    ax.set_xticks(seasons)
    ax.set_xticklabels(seasons)

    # Axis labels
    ax.set_xlabel("Season")
    ax.set_ylabel("Command Grade (20–80)")

    # Grade range
    ax.set_ylim(20, 80)

    # Horizontal reference lines
    for grade in [20, 30, 40, 50, 60, 70, 80]:
        ax.axhline(
            y=grade,
            linestyle="--",
            linewidth=0.8,
            alpha=0.4
        )

    ax.legend()

    st.pyplot(fig)