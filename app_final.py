from pyexpat import features
import streamlit as st
import pandas as pd
import joblib
import base64
import json
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# === Page Configuration ===
st.set_page_config(
    page_title="1. FC Köln Transfer Dashboard",
    page_icon="⚽",
    layout="wide"
)

# === GLOBAL DESIGN STYLES ===
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 0rem;
    }

    /* Button style: Predict – visible, bold, gradient */
    .stButton > button:first-child {
        background: linear-gradient(to right, #ba0c2f, #a5002a);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        height: 80px;
        width: 100%;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(255, 0, 0, 0.4);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton > button:first-child:hover {
        background: linear-gradient(to right, #ff3344, #cc0022);
        box-shadow: 0 6px 20px rgba(255, 0, 0, 0.6);
        transform: scale(1.03);
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === BACKGROUND FUNCTION WITH OVERLAY ===
stadium_background = "stadium.jpg"
logo_fc = "1-fc-koln-logo-png_seeklogo-266469.png"
logo_uni = "Uni_blau2.png"

def set_bg_image_with_overlay(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()

    st.markdown(
    f"""
    <style>
    @keyframes fadeIn {{
        from {{opacity: 0;}}
        to {{opacity: 1;}}
    }}

    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #f9f9f9 !important;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-weight: 500;
    }}

    h1, h2, h3, h4, h5, p, label, .stSlider span {{
        color: #f9f9f9 !important;
        font-weight: 600 !important;
    }}

    .stSlider > div[data-baseweb='slider'] span {{
        background-color: transparent !important;
        font-weight: bold;
    }}

    .stSelectbox, .stNumberInput, .stTextInput, .stMarkdown {{
        font-weight: 500;
    }}

    .block-container {{
        max-width: 1100px;
        margin: auto;
        padding-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# === BACKGROUND INIT ===
set_bg_image_with_overlay(stadium_background)


##Loading Dataset
@st.cache_data
def load_player_reference_data():
    return pd.read_csv("final_dataset.csv")

reference_df = load_player_reference_data()


# === HELP ICON ===
st.markdown("""
<style>
.help-icon {
    display: inline-block;
    position: relative;
    cursor: pointer;
    margin-left: 8px;
    color: #FFD700;
    font-weight: bold;
}

.help-icon:hover .tooltip {
    display: block;
}

.tooltip {
    display: none;
    position: absolute;
    top: 20px;
    left: 0;
    background-color: #333;
    color: #fff;
    padding: 0.7rem;
    border-radius: 8px;
    font-size: 0.85rem;
    max-width: 240px;
    z-index: 1000;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)



# === HEADER SECTION WITH LOGOS AND TITLE ===
col_fc, col_title, col_uni = st.columns([2, 6, 2])

with col_title:
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0.3rem;'>1. FC Köln Transfer Success Predictor</h1>
        <p style='text-align: center; color: white; font-size: 0.9rem;'>
            Developed in cooperation with the University of Cologne – estimate expected playing percentage for potential transfers.
        </p>
    """, unsafe_allow_html=True)

with col_fc:
    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.image(logo_fc, width=90)
    st.markdown("</div>", unsafe_allow_html=True)

with col_uni:
    st.markdown(
        f"""
        <div style='text-align: right;'>
            <img src="data:image/png;base64,{base64.b64encode(open(logo_uni, "rb").read()).decode()}" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )

# === CARD STYLE HELPER ===
def card_start(title):
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.08); 
                padding: 1.2rem 1.5rem; 
                border-radius: 12px; 
                margin-bottom: 1rem;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    <h4 style="color: white;">{title}</h4>
    """, unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# === Load Model and Mappings ===
model = xgb.XGBRegressor()
model.load_model("model2.json")

@st.cache_data
def load_mapping():
    with open("category_mappings.json") as f:
        return json.load(f)
category_mappings = load_mapping()



valid_areas = category_mappings["from_competition_competition_area"]
valid_to_areas = category_mappings["to_competition_competition_area"]
valid_position_groups = category_mappings["positionGroup"]
valid_main_positions = category_mappings["mainPosition"]
valid_feet = category_mappings["foot"]
valid_clean_sheets = category_mappings.get("clean_sheets_before_grouped", ["0-1", "2-4", "5-9", "10-14", "15+"])
valid_scorer_groups = category_mappings.get("scorer_before_grouped_category", ["defender/goalkeeper", "0-3", "4-6", "7-10", "11-15", "16-20", "21-30", "30+"])
# Dynamic mapping from real data
position_group_to_main = pd.read_csv("xgboost_predictions_test.csv").groupby("positionGroup")["mainPosition"].unique().apply(list).to_dict()

area_to_levels = {
    'Austria': [1, 2], 'Belgium': [1, 2], 'Bosnia-Herzegovina': [1], 'Bulgaria': [1], 'Canada': [1],
    'Croatia': [1, 2], 'Czech Republic': [1], 'Denmark': [1, 2], 'England': [1, 2, 3, 4], 'Estonia': [1],
    'Finland': [1, 2], 'France': [1, 2, 3], 'Georgia': [1], 'Germany': [1, 2, 3, 4], 'Greece': [1],
    'Hungary': [1], 'Ireland': [1], 'Israel': [1], 'Italy': [1, 2], 'Japan': [1], 'Korea, South': [1],
    'Latvia': [1], 'Lithuania': [1], 'Luxembourg': [1], 'Malta': [1], 'Moldova': [1], 'Montenegro': [1],
    'Netherlands': [1, 2], 'Northern Ireland': [1], 'Norway': [1, 2], 'Poland': [1], 'Portugal': [1, 2],
    'Romania': [1], 'Russia': [1], 'Saudi Arabia': [1], 'Scotland': [1], 'Serbia': [1], 'Slovakia': [1],
    'Slovenia': [1], 'Spain': [1, 2], 'Sweden': [1, 2], 'Switzerland': [1, 2], 'Türkiye': [1],
    'Ukraine': [1], 'United States': [1, 2, 3], 'Wales': [1]
}


# === Inputs ===

def help_input(label, tooltip_text):
    st.markdown(f"""
    <style>
    .help-icon {{
        display: inline-block;
        position: relative;
        cursor: pointer;
        margin-left: 8px;
        color: #FFD700;
        font-weight: bold;
    }}

    .tooltip {{
        display: none;
        position: absolute;
        top: 22px;
        left: 0;
        width: 360px;
        background-color: #333;
        color: #fff;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        z-index: 1000;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        white-space: normal;
        line-height: 1.4;
    }}

    .help-icon:hover .tooltip {{
        display: block;
    }}
    </style>

    <div style='margin-bottom: -15px'>
        <label style='font-weight:600;'>{label}</label>
        <span class="help-icon">❓
            <span class="tooltip">{tooltip_text}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)



col1, col2 = st.columns(2)

with col1:
    card_start("🧍 Player Profile")

    help_input("Height (cm)", "Enter the player's height in centimeters. Taller players may perform better in aerial duels.")
    height = st.slider("", 150, 220, 180, key="height")

    help_input("Transfer Age", "Enter the player's age at the time of transfer. Important for assessing player development and experience.")
    transfer_age = st.slider("", 16, 40, 25, key="transfer_age")

    help_input("Position Group", "Select the player's position group. Important for tactical fit and team balance.")
    position_group = st.selectbox("", valid_position_groups, key="position_group")

    help_input("Main Position", "Select the player's main position. Important for tactical fit and team balance.")
    main_position = st.selectbox("", position_group_to_main.get(position_group, []), key="main_position")

    help_input("Preferred Foot", "Select the player's preferred foot. Important for assessing shooting and passing capabilities.")
    foot = st.selectbox("", valid_feet, key="preferred_foot")

    help_input("Player Market Value (€M)", "Estimated market value of the player in millions of euros. Important for assessing transfer budget and player quality.")
    market_value = st.number_input("", 0.0, 200.0, 15.0, key="market_value")

    card_end()

    card_start("📊 Performance Details")

    help_input("Playing % Before", "Percentage of games played in the last season. Important for assessing player fitness and reliability.")
    percentage_played_before = st.slider("", 0.0, 100.0, 50.0, key="percentage_played_before")

    if position_group.lower() in ['defender', 'goalkeeper']:
        scorer_raw = "defender/goalkeeper"
        st.markdown("**Scorer (Goals + Assists):** Automatically ignored for defenders and goalkeepers")
    else:
        help_input("Scorer Value (Goals + Assists)", "Total goals and assists scored by the player in the last season. Important for forwards and midfielders.")
        # Ohne "defender/goalkeeper" in der Auswahl
        scorer_options = [g for g in valid_scorer_groups if g != "defender/goalkeeper"]
        scorer_raw = st.selectbox("", scorer_options, key="scorer")

    help_input("Clean Sheets", "Number of clean sheets kept by the player in the last season. Important for goalkeepers and defenders.")
    clean_sheets_before = st.selectbox("", valid_clean_sheets, key="clean_sheets")

    card_end()

with col2:
    card_start("🔄 Transfer Details")

    help_input("From Team Market Value (€M)", "Market value of the team the player is transferring from. Important for assessing the player's previous club's financial strength and quality.")
    from_team_market_value = st.number_input("", 0.0, 1000.0, 61.7, key="from_team_market_value")

    help_input("To Team Market Value (€M)", "Market value of the team the player is transferring to. Important for assessing the player's new club's financial strength and quality.")
    to_team_market_value = st.number_input("", 0.0, 1000.0, 61.7, key="to_team_market_value")

    help_input("From Area", "Select the geographical area of the team the player is transferring from. Important for assessing league strength and player adaptation.")
    from_area = st.selectbox("", valid_areas, index=valid_areas.index("Germany") if "Germany" in valid_areas else 0, key="from_area")

    help_input("From Level", "Select the competition level of the team the player is transferring from. Important for assessing league strength and player adaptation.")
    from_level = st.selectbox("", area_to_levels.get(from_area, [1, 2, 3, 4]),
                              index=area_to_levels.get(from_area, [1, 2, 3, 4]).index(1) if 1 in area_to_levels.get(from_area, [1, 2, 3, 4]) else 0,
                              key="from_level")

    help_input("To Area", "Select the geographical area of the team the player is transferring to. Important for assessing league strength and player adaptation.")
    to_area = st.selectbox("", valid_to_areas, index=valid_to_areas.index("Germany") if "Germany" in valid_to_areas else 0, key="to_area")

    help_input("To Level", "Select the competition level of the team the player is transferring to. Important for assessing league strength and player adaptation.")
    to_level = st.selectbox("", area_to_levels.get(to_area, [1, 2, 3, 4]),
                            index=area_to_levels.get(to_area, [1, 2, 3, 4]).index(1) if 1 in area_to_levels.get(to_area, [1, 2, 3, 4]) else 0,
                            key="to_level")

    card_end()

    with st.expander("⚙️ Further Transfer Details"):
        help_input("Loan Transfer", "Check if the transfer is a loan. Important for assessing player commitment and future prospects.")
        isLoan = st.checkbox("Loan Transfer", key="is_loan")

        help_input("Was Loan Before", "Check if the player was previously on loan. Important for understanding the player's transfer history.")
        wasLoan = st.checkbox("Was Loan Before", key="was_loan")

        help_input("Was Joker Substitute", "Check if the player was used as a joker substitute. Important for assessing tactical versatility.")
        was_joker = st.checkbox("Was Joker Substitute", key="was_joker")


# === Foreign Transfer Logic ===
foreign_transfer = int((from_area != to_area))


data = {col: 0 for col in model.feature_names_in_}
data.update({
    'height': height,
    'transferAge': transfer_age,
    'isLoan': int(isLoan),
    'wasLoan': int(wasLoan),
    'was_joker': int(was_joker),
    'foreign_transfer': foreign_transfer,
    'percentage_played_before': percentage_played_before,
    'scorer_before_grouped_category': scorer_raw,
    'clean_sheets_before_grouped': clean_sheets_before,
    'fromTeam_marketValue': from_team_market_value,
    'toTeam_marketValue': to_team_market_value,
    'marketvalue_closest': market_value,
    'from_competition_competition_level': from_level,
    'to_competition_competition_level': to_level,
    'foot': foot,
    'mainPosition': main_position,
    'positionGroup': position_group,
    'from_competition_competition_area': from_area,
    'to_competition_competition_area': to_area,
    'value_per_age': market_value / transfer_age if transfer_age > 0 else 0,
    'value_age_product': transfer_age * market_value,
    'team_market_value_relation': to_team_market_value / from_team_market_value if from_team_market_value > 0 else 0
})


input_df = pd.DataFrame([data])

# Category typing
for col, cats in category_mappings.items():
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col], categories=cats)


# === ACTION BUTTONS & OUTPUT ===
col_l, col_m = st.columns([1, 6])

with col_l:
    predict_clicked = st.button("🔮 Predict")
   

# Hilfsfunktion, um HEX → RGBA umzuwandeln
def hex_to_rgba(hex_color, alpha=0.5):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

# Prediction
if predict_clicked:
    with st.spinner("Running prediction..."):
        pred = model.predict(input_df)[0]
                # ÄHNLICHKEITSBERECHNUNG
        input_query = {
            #"height": height,
            "mainPosition": main_position,
            #"positionGroup": position_group,
            #"foot": foot,
            "transferAge": transfer_age,
            "marketvalue_closest": market_value,
            #"toTeam_marketValue": to_team_market_value,
            #"fromTeam_marketValue": from_team_market_value,
            "percentage_played_before": percentage_played_before,
            "scorer_before_grouped_category": scorer_raw,
            #"clean_sheets_before": clean_sheets_before,  # Falls du das dynamisch brauchst, kannst du das noch einbauen
            #"value_age_product": transfer_age * market_value,
            #"value_per_age": market_value / transfer_age if transfer_age > 0 else 0,
            'from_competition_competition_area': from_area,
            'to_competition_competition_area': to_area,
            'from_competition_competition_level': from_level,
            'to_competition_competition_level': to_level,
            'team_market_value_relation': to_team_market_value / from_team_market_value if from_team_market_value > 0 else 0

        }

        # === ÄHNLICHE SPIELER FINDEN ===
        def find_similar_players(input_data, df, top_n=3):
            features = list(input_data.keys())
            id_cols = ['playerId', 'playerName', 'mainPosition', "percentage_played", 'season']

            # 👇 Neu: doppelte Spaltennamen vermeiden
            all_cols = list(dict.fromkeys(features + id_cols))

            df_subset = df[all_cols].dropna().copy()
            df_subset.reset_index(drop=True, inplace=True)

            for col in df_subset.select_dtypes(include='object').columns:
                df_subset[col] = df_subset[col].astype(str)
                if col in input_data:
                    input_data[col] = str(input_data[col])

            df_subset = df_subset[df_subset['mainPosition'] == input_data['mainPosition']].copy()
            df_subset = df_subset.sort_values("season", ascending=False).drop_duplicates("playerId", keep="first").reset_index(drop=True)

            df_encoded = pd.get_dummies(df_subset[features])
            input_encoded = pd.get_dummies(pd.DataFrame([input_data]))
            df_encoded, input_encoded = df_encoded.align(input_encoded, join="inner", axis=1)

            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_encoded)
            input_scaled = scaler.transform(input_encoded)

            df_subset["distance"] = cdist(df_scaled, input_scaled).flatten()
            return df_subset.nsmallest(top_n, "distance")[["playerName", "mainPosition", "season", "percentage_played", "distance"]]


        similar_players = find_similar_players(input_query, reference_df)


        if pred < 30:
            msg, color, emoji = "Not Recommended", "#FF4B4B", "🚫"
        elif pred < 50:
            msg, color, emoji = "Uncertain", "#FFA500", "⚠️"  
        elif pred < 65:
            msg, color, emoji = "Good Transfer", "#90EE90", "✅"
        elif pred < 80:
            msg, color, emoji = "Very Good Transfer", "#32CD32", "💎"
        else:
            msg, color, emoji = "Key Player", "#008000", "🌟"

        rgba_bg = hex_to_rgba(color, alpha=0.6)  # 0.6 ist die Transparenz

        with col_m:
            st.markdown(f"""
            <div style='
                background-color: {rgba_bg};
                height: 80px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                border-radius: 12px;
                text-align: center;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);'>
                <span style='color: white; font-size: 1.3rem; font-weight: 600;'>
                    {emoji} {msg} – Expected Playing Time: <strong>{pred:.2f}%</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 👥 Top 3 Similar Transfers")
            for _, row in similar_players.iterrows():
                st.markdown(f"- **{row['playerName']}** | Position: {row['mainPosition']} | Season: {row['season']} | Playing %: {row['percentage_played']}%")

        
if st.checkbox("Show feature vector"):
    st.write({k: v for k, v in data.items() if v != 0})


# === Feature Importances ===

import matplotlib.pyplot as plt
import numpy as np

with st.expander("📈 Show Feature Importances"):
    fig, ax = plt.subplots()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # top 10
    features = np.array(model.feature_names_in_)[indices]

    ax.barh(features[::-1], importances[indices][::-1])
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# === Footer Section ===
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #f9f9f9; font-size: 0.8rem;'>
        © 2025 Next11 in Cooperation with 1. FC Köln and University of Cologne – All rights reserved.
        <br>
        <a href='https://www.fc-koeln.de'>1. FC Köln</a> | <a href='https://www.uni-koeln.de'>University of Cologne</a>
    </div>
    """, unsafe_allow_html=True)

