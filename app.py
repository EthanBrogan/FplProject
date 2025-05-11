import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Data & Models ---
@st.cache_data
def load_player_data():
    # Read precomputed CSVs
    dfs = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        dfs[pos] = pd.read_csv(f"output/{pos}_players.csv")
    return dfs

@st.cache_resource
def load_scaler_and_model():
    scaler = joblib.load("scaler.pkl")
    # If you saved your model: model = tf.keras.models.load_model("model.h5")
    # Otherwise, predictions are already baked into CSVs.
    return scaler

player_data = load_player_data()
scaler = load_scaler_and_model()

st.title("⚽️ FPL Predicted Points Demo")

# --- Sidebar options ---
st.sidebar.header("Fill Recommendations")
mode = st.sidebar.selectbox("Mode", ["Single Position", "Full Squad"])
if mode == "Single Position":
    position = st.sidebar.selectbox("Position", ["GK","DEF","MID","FWD"])
    n_spots = st.sidebar.slider("Number of players to pick", 1, 5, 2)

# --- Main App ---
if mode == "Single Position":
    st.header(f"Top {n_spots} {position} Recommendations")
    df = player_data[position]
    # Filter by budget if you want; otherwise take top N
    topn = df.nlargest(n_spots, "Predicted Points")[["name","team","value","Predicted Points"]]
    st.table(topn.style.format({"value":"£{:.1f}","Predicted Points":"{:.1f}"}))

elif mode == "Full Squad":
    st.header("Current Squad Selection")
    # Allow selection of current players
    current = {}
    budget = st.sidebar.number_input("Remaining Budget (£m)", 0.0, 100.0, 10.0, 0.1)
    for pos, max_sel in [("GK",2),("DEF",5),("MID",5),("FWD",3)]:
        current[pos] = st.multiselect(f"Current {pos}s", options=player_data[pos]["name"], default=[])
    # Compute open slots
    open_slots = {pos: max_sel - len(current[pos]) for pos,max_sel in [("GK",2),("DEF",5),("MID",5),("FWD",3)]}
    st.write("### Open Slots", open_slots)

    # Recommend fills
    all_recs = []
    for pos, slots in open_slots.items():
        if slots > 0:
            df = player_data[pos]
            # Filter out already selected and over-budget
            df = df[~df["name"].isin(current[pos])]
            df = df[df["value"] <= budget]  # simple budget filter
            picks = df.nlargest(slots, "Predicted Points")
            picks["position"] = pos
            all_recs.append(picks)
            st.subheader(f"Fill {slots} {pos}(s)")
            st.table(picks[["name","team","value","Predicted Points"]]
                     .style.format({"value":"£{:.1f}","Predicted Points":"{:.1f}"}))
            budget -= picks["value"].sum()
    st.write(f"### Final Remaining Budget: £{budget:.1f}")
