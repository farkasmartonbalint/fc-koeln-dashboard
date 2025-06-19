import streamlit as st
import pandas as pd
import joblib
import base64
import json
from pathlib import Path
import xgboost as xgb

st.set_page_config(
    page_title="1.FC Köln Transfer Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Correct paths to images
stadium_background = "stadium.jpg"
logo_fc = "1-fc-koln-logo-png_seeklogo-266469.png"
logo_uni = "Uni_blau.png"

# Apply full background styling via base64 with overlay
def set_bg_image_with_overlay(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)),
                        url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .stSlider > div[data-baseweb='slider'] span {{
            background-color: transparent !important;
            color: #ba0c2f !important;
            font-weight: bold;
        }}
        .stButton button {{
            background-color: #ba0c2f;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
        }}
        .block-container p, .block-container h1, .block-container h2, .block-container h3, .block-container h4 {{
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Inject background with white transparency overlay
set_bg_image_with_overlay(stadium_background)

# Logos and title
col_fc, col_title, col_uni = st.columns([1, 5, 1])
with col_fc:
    st.image(logo_fc, width=130)
with col_title:
    st.markdown("<h1 style='text-align: center;'>1. FC Köln Transfer Success Predictor with Attackers</h1>", unsafe_allow_html=True)
with col_uni:
    st.image(logo_uni, width=500)

st.markdown("""
<p>
This tool, developed in cooperation with the University of Cologne, predicts the <strong>expected playing percentage</strong> for potential transfers. Provide key transfer details to simulate potential success.
</p>
""", unsafe_allow_html=True)



# === Load Model and Mappings ===
model = xgb.XGBRegressor()
model.load_model("model_attackers.json")

with open("category_mappings_attackers.json") as f:
    category_mappings = json.load(f)

valid_areas = category_mappings["from_competition_competition_area"]
valid_to_areas = category_mappings["to_competition_competition_area"]
valid_position_groups = category_mappings["positionGroup"]
valid_main_positions = category_mappings["mainPosition"]
valid_feet = category_mappings["foot"]
valid_transfer_age = category_mappings["transfer_age_grouped"]
valid_scorer_grouped = category_mappings["scorer_before_grouped"]
valid_clean_sheets = category_mappings["clean_sheets_before_grouped_new"]

# Dynamic mapping from real data
position_group_to_main = pd.read_csv("xgboost_predictions_test_attackers.csv").groupby("positionGroup")["mainPosition"].unique().apply(list).to_dict()

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
col1, col2 = st.columns(2)
with col1:
    height = st.slider("Height (cm)", 150, 220, 180)
    transfer_age_grouped = st.selectbox("Transfer Age Group", valid_transfer_age)
    isLoan = st.checkbox("Loan Transfer")
    wasLoan = st.checkbox("Was Loan Before")
    was_joker = st.checkbox("Was Joker Substitute")
    market_value = st.number_input("Player Market Value (€M)", 0.0, 200.0, 15.0)
    percentage_played_before = st.slider("Playing % Before", 0.0, 100.0, 50.0)
    scorer_before_grouped = st.selectbox("Scorer Grouped", valid_scorer_grouped)
    clean_sheets_grouped = st.selectbox("Clean Sheets Grouped", valid_clean_sheets)

with col2:
    from_team_market_value = st.number_input("From Team Market Value (€M)", 0.0, 1000.0, 50.0)
    to_team_market_value = st.number_input("To Team Market Value (€M)", 0.0, 1000.0, 80.0)
    from_area = st.selectbox("From Area", valid_areas)
    from_level = st.selectbox("From Level", area_to_levels.get(from_area, [1, 2, 3, 4]))
    to_area = st.selectbox("To Area", valid_to_areas)
    to_level = st.selectbox("To Level", area_to_levels.get(to_area, [1, 2, 3, 4]))
    position_group = st.selectbox("Position Group", valid_position_groups)
    main_position = st.selectbox("Main Position", position_group_to_main.get(position_group, []))
    foot = st.selectbox("Preferred Foot", valid_feet)

foreign_transfer = int((from_area != to_area))

# === Feature Vector ===
data = {col: 0 for col in model.feature_names_in_}
data['height'] = height
data['transfer_age_grouped'] = transfer_age_grouped
data['isLoan'] = int(isLoan)
data['wasLoan'] = int(wasLoan)
data['was_joker'] = bool(was_joker)
data['foreign_transfer'] = foreign_transfer
data['percentage_played_before'] = percentage_played_before
data['scorer_before_grouped'] = scorer_before_grouped
data['clean_sheets_before_grouped_new'] = clean_sheets_grouped
data['fromTeam_marketValue'] = from_team_market_value
data['toTeam_marketValue'] = to_team_market_value
data['marketvalue_closest'] = market_value
data['from_competition_competition_level'] = from_level
data['to_competition_competition_level'] = to_level
data['foot'] = foot
data['mainPosition'] = main_position
data['positionGroup'] = position_group
data['from_competition_competition_area'] = from_area
data['to_competition_competition_area'] = to_area

input_df = pd.DataFrame([data])

# Category typing
for col, cats in category_mappings.items():
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col], categories=cats)

# === Prediction ===
if st.button("Predict"):
    pred = model.predict(input_df)[0]

    if pred < 40:
        msg, color = "🚫 Not Recommended", "#FF4B4B"
    elif pred < 55:
        msg, color = "⚠️ Uncertain", "#FFA500"
    elif pred < 70:
        msg, color = "🤔 Worth a Try", "#FFD700"
    elif pred < 80:
        msg, color = "✅ Good Transfer", "#90EE90"
    elif pred < 90:
        msg, color = "💎 Very Good", "#32CD32"
    else:
        msg, color = "🌟 Superstar", "#008000"

    st.markdown(f"""
    <div style='background-color:{color}; padding:1rem; border-radius:10px;'>
        <h3 style='color:white; text-align:center;'>
         {msg}: {pred:.2f}%
        </h3>
    </div>
    """, unsafe_allow_html=True)

if st.checkbox("Show feature vector"):
    st.write({k: v for k, v in data.items() if v != 0})
