# command_delta_dashboard_all_seasons.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MLB Command Delta Dashboard", layout="wide")

# --- Load processed data ---
df = pd.read_csv("command_deltas_processed.csv")

# --- Sidebar navigation ---
page = st.sidebar.selectbox("Select Page", ["Leaderboard", "Pitcher View"])

# --- Common filter ---
min_pitches = st.sidebar.slider(
    "Minimum Pitch Count",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)

# --- Leaderboard page ---
if page == "Leaderboard":
    st.title("MLB Command Delta Leaderboard")

    # Season filter
    season = st.selectbox(
        "Select Season",
        sorted(df["season"].unique())
    )

    # Pitch type filter
    pitch = st.selectbox(
        "Select Pitch Type",
        sorted(df["pitch_type"].unique())
    )

    filtered = df[
        (df["season"] == season) &
        (df["pitch_type"] == pitch) &
        (df["pitch_count"] >= min_pitches)
    ]

    filtered = filtered.sort_values("command_delta", ascending=False)

    st.subheader(f"Top Command Grades for {pitch} ({season})")
    st.dataframe(
        filtered[["player_name", "pitch_count", "command_delta", "command_grade", "z_score"]]
        .reset_index(drop=True)
    )

    # Optional CSV download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV for this Pitch Type & Season",
        data=csv,
        file_name=f"command_delta_{pitch}_{season}.csv",
        mime="text/csv"
    )

# --- Pitcher view page ---
elif page == "Pitcher View":
    st.title("MLB Command Delta - Pitcher View (All Seasons)")

    # Dropdown of all pitchers who meet min_pitches in at least one season
    available_pitchers = df[df["pitch_count"] >= min_pitches]["player_name"].unique()
    available_pitchers = np.sort(available_pitchers)

    pitcher_name = st.selectbox("Select Pitcher", available_pitchers)

    # Filter for selected pitcher across all seasons
    pitcher_df = df[
        (df["player_name"] == pitcher_name) &
        (df["pitch_count"] >= min_pitches)
    ].sort_values(["season", "pitch_type"])

    if pitcher_df.empty:
        st.warning(f"No data found for {pitcher_name} with at least {min_pitches} pitches.")
    else:
        st.subheader(f"{pitcher_name} Command Grades Across Seasons")
        st.dataframe(
            pitcher_df[["season", "pitch_type", "pitch_count", "command_delta", "command_grade", "z_score"]]
            .reset_index(drop=True)
        )

        # Optional CSV download
        csv = pitcher_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV for this Pitcher",
            data=csv,
            file_name=f"command_delta_{pitcher_name.replace(' ','_')}_all_seasons.csv",
            mime="text/csv"
        )