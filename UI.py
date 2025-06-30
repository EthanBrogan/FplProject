import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# --- Load Data & Models ---
@st.cache_data
def load_player_data():
    dfs = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        dfs[pos] = pd.read_csv(f"output/{pos}_players_Mdl_2024_25.csv")
    return dfs

@st.cache_resource
def load_scaler_and_model():
    scaler = joblib.load("scaler.pkl")
    return scaler

player_data = load_player_data()
scaler = load_scaler_and_model()

# --- Sidebar: Page selector ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Recommendations", "News & Stats", "Transfers & Prices", "Comparison"])

# --- Page: Recommendations ---
if page == "Recommendations":
    st.title("FPL Team Optimisation Tool")

    mode = st.sidebar.selectbox("Mode", ["Single Position", "Full Squad"])
    if mode == "Single Position":
        position = st.sidebar.selectbox("Position", ["GK","DEF","MID","FWD"])
        n_spots = st.sidebar.slider("Number of players to pick", 1, 5, 2)

        st.header(f"Top {n_spots} {position} Recommendations")
        df = player_data[position]
        topn = df.nlargest(n_spots, "Predicted Points")[["name","team","value","Predicted Points"]]
        st.table(topn.style.format({"value":"£{:.1f}","Predicted Points":"{:.1f}"}))

    else:  # Full Squad
        st.header("Current Squad Selection")

        # Position limits
        position_limits = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        current = {}

        # 1) Select current squad with names + prices in dropdown
        for pos in position_limits:
            df_pos = player_data[pos]
            options_with_price = [f"{row['name']} (£{row['value']:.1f}m)" for _, row in df_pos.iterrows()]
            # Map displayed name to actual player name for internal use
            name_map = {f"{row['name']} (£{row['value']:.1f}m)": row['name'] for _, row in df_pos.iterrows()}

            selected = st.multiselect(
                f"Select your current {pos}s",
                options=options_with_price,
                default=[]
            )
            # Convert back to player names only
            current[pos] = [name_map[name] for name in selected]

        # 2) Validate
        invalid = False
        for pos, limit in position_limits.items():
            if len(current[pos]) > limit:
                st.error(f"❌ Too many {pos}s! Max is {limit}.")
                invalid = True
        if invalid:
            st.stop()

        # 3) Calculate remaining budget
        initial_budget = 100.0
        spent = sum(
            player_data[p].set_index('name').loc[current[p], 'value'].sum()
            for p in current
        )
        budget = initial_budget - spent
        st.sidebar.write(f"Remaining Budget: £{budget:.1f}m")

        # 4) Show open slots
        open_slots = {
            pos: position_limits[pos] - len(current[pos])
            for pos in position_limits
        }
        st.write("### Open Slots", open_slots)

        # 5) Recommend fills
        all_recs = []  # store each position’s recommendations
        for pos, slots in open_slots.items():
            if slots <= 0:
                continue
            df = player_data[pos]
            df = df[~df["name"].isin(current[pos])]
            df = df[df["value"] <= budget]
            picks = df.nlargest(slots, "Predicted Points").copy()
            picks["position"] = pos
            all_recs.append(picks)

            st.subheader(f"Suggested {pos} Fills ({slots})")
            st.table(
                picks[["name","team","value","Predicted Points"]]
                .style.format({"value":"£{:.1f}", "Predicted Points":"{:.1f}"})
            )
            budget -= picks["value"].sum()

        st.write(f"### Final Remaining Budget: £{budget:.1f}m")

        # 6) Squad Summary Table
        st.write("## Squad Summary")
        summary_records = []

        # a) Already-selected players
        for pos, names in current.items():
            df_pos = player_data[pos].set_index("name")
            for name in names:
                row = df_pos.loc[name]
                summary_records.append({
                    "Name": name,
                    "Position": pos,
                    "Team": row["team"],
                    "Type": "Selected",
                    "Price (£m)": row["value"],
                    "Predicted Points": row["Predicted Points"],
                })

        # b) Recommended fills
        for picks in all_recs:
            for _, row in picks.iterrows():
                summary_records.append({
                    "Name": row["name"],
                    "Position": row["position"],
                    "Team": row["team"],
                    "Type": "Recommended",
                    "Price (£m)": row["value"],
                    "Predicted Points": row["Predicted Points"],
                })

        summary_df = pd.DataFrame(summary_records)

        # Sort with custom order for positions so it appears in GK, DEF, MID, FWD order
        position_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        summary_df["PositionOrder"] = summary_df["Position"].map(position_order)
        summary_df = summary_df.sort_values(["Type", "PositionOrder"]).drop(columns=["PositionOrder"]).reset_index(drop=True)

        st.dataframe(summary_df, use_container_width=True)

# --- Page: News & Stats ---
elif page == "News & Stats":
    st.title(" FPL Player Stats")

    st.markdown("""
    Below is a live overview of player statistics from the official Fantasy Premier League API.
    You can toggle top-N by total points, goals, assists, etc.
    """)

    @st.cache_data(show_spinner=False)
    def fetch_bootstrap():
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        return r.json()

    data = fetch_bootstrap()
    players = pd.DataFrame(data["elements"])
    teams = {t["id"]: t["name"] for t in data["teams"]}

    players["team_name"] = players["team"].map(teams)
    stats = players[[
        "first_name", "second_name", "team_name",
        "total_points", "goals_scored", "assists",
        "clean_sheets", "minutes", "goals_conceded"
    ]].rename(columns={
        "first_name": "First",
        "second_name": "Last"
    })

    metric = st.selectbox(
        "Select statistic to view top players by",
        ["total_points", "goals_scored", "assists", "clean_sheets", "minutes"]
    )
    top_n = st.slider("Show top players", 5, 30, 10)

    top_players = stats.nlargest(top_n, metric)
    top_players["Name"] = top_players["First"] + " " + top_players["Last"]
    st.table(
        top_players[["Name","team_name", metric]]
        .reset_index(drop=True)
        .style.format({metric: "{:.0f}"})
    )

    st.markdown("*Data from the FPL public API — updates every game-week.*")

# --- Page: Transfers & Prices ---
elif page == "Transfers & Prices":
    st.title("Net Transfers & Price Changes")

    st.markdown("""
    This page shows how many managers have transferred each player in or out, plus their current price.
    """)

    @st.cache_data(ttl=300)
    def fetch_market_data():
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        return r.json()

    market = fetch_market_data()
    teams = {t["id"]: t["name"] for t in market["teams"]}
    elements = market["elements"]

    records = []
    for p in elements:
        name = f"{p['first_name']} {p['second_name']}"
        team = teams[p["team"]]
        price = p["now_cost"] / 10
        ins = p["transfers_in"]
        outs = p["transfers_out"]
        net = ins - outs
        records.append({
            "Player": name,
            "Team": team,
            "Price (£m)": price,
            "Transfers In": ins,
            "Transfers Out": outs,
            "Net Transfers": net
        })

    df_market = pd.DataFrame(records)
    df_sorted = df_market.sort_values("Net Transfers", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_sorted.style.format({
            "Price (£m)": "{:.1f}",
            "Transfers In": "{:,}",
            "Transfers Out": "{:,}",
            "Net Transfers": "{:,}"
        }),
        use_container_width=True
    )
    st.markdown("*Data via FPL public API — refreshes every 5 minutes.*")

# --- New Page: Comparison ---
else:
    st.title("Comparison: Predicted vs Actual Best Squads")

    # Load all four CSVs 
    fpl_best_2324 = pd.read_csv("output/fpl_best_2324.csv")
    fpl_best_2425 = pd.read_csv("output/fpl_best_2425.csv")
    team_mdl_2324 = pd.read_csv("output/Team_mdl_2023_24.csv")
    team_mdl_2425 = pd.read_csv("output/Team_mdl_2024_25.csv")

    st.markdown("### Actual Best Squad 2023/24 Season")
    st.dataframe(fpl_best_2324.style.format({"total_points": "{:.0f}"}), use_container_width=True)

    st.markdown("### Predicted Best Squad for 2023/24 Season (Model trained on 22/23)")
    st.dataframe(team_mdl_2324.style.format({"Predicted Points": "{:.1f}", "value": "£{:.1f}"}), use_container_width=True)

    st.markdown("### Actual Best Squad 2024/25 Season")
    st.dataframe(fpl_best_2425.style.format({"total_points": "{:.0f}"}), use_container_width=True)

    st.markdown("### Predicted Best Squad for 2024/25 Season (Model trained on 23/24)")
    st.dataframe(team_mdl_2425.style.format({"Predicted Points": "{:.1f}", "value": "£{:.1f}"}), use_container_width=True)








