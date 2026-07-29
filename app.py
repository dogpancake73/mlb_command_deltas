import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLB Command Delta Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("command_deltas_processed.csv")

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
# Command Delta Primer Page
# -----------------------------

elif page == "Command Delta Primer":

    st.title("Command Delta: A Primer")

    st.markdown("""
### What is Command Delta?

Command Delta measures pitch location consistency using dispersion of plate_x and plate_z.

Lower values = tighter clusters = better command.

---

### How It Works

Command Delta = √(σx² + σz²)

Converted to a 20–80 scouting scale using pitch-family z‑scores.

---

### Data Source

Statcast pitch-by-pitch data via `pybaseball`.

Covers **2023–2026** MLB seasons.

---

### Explore the Dashboard

Use the sidebar to:

* View command leaderboards
* Explore pitcher command trends
""")
