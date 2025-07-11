import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==============================
# Password Protection
# ==============================
def check_password():
    correct_password = "Rait@123"
    password = st.text_input("🔒 Enter Password to Access", type="password")
    if password == "":
        st.stop()
    elif password != correct_password:
        st.error("❌ Incorrect password. Please try again.")
        st.stop()

check_password()

# ==============================
# Page Setup
# ==============================
st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🍿", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #222 !important;
}
.stButton>button {
    background-color: #ffc220;
    color: black;
    font-weight: bold;
    border-radius: 6px;
    padding: 10px 25px;
}
.stButton>button:hover {
    background-color: #e6ac00;
    color: white;
}
.report-text {
    font-size: 16px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_csv("extended_distance_dataset.csv")  # REPLACED
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# ==============================
# Constants & Settings
# ==============================
PETROL_PRICE = 106
VEHICLE_MILEAGE = 15
CO2_PER_KM_DEFAULT = 0.15

REALISTIC_SPOILAGE_RATES = {
    "tomato": 0.01, "onion": 0.003, "potato": 0.004,
    "cabbage": 0.005, "spinach": 0.012, "cauliflower": 0.006,
    "carrot": 0.002, "banana": 0.018, "mango": 0.02,
    "grapes": 0.015, "almonds": 0.0005, "dry fruits": 0.0008,
    "default": 0.007
}

HIGH_SHELF_COMMODITIES = ["rice", "wheat", "dal", "pulses", "almonds", "dry fruits", "grains", "nuts"]

VEHICLE_EMISSIONS = {
    "Bike": 0.02,
    "Tempo": 0.1,
    "Mini Truck": 0.12,
    "Truck": 0.18
}

def assign_vehicle(weight_kg):
    if weight_kg <= 20:
        return "Bike"
    elif weight_kg <= 100:
        return "Tempo"
    elif weight_kg <= 300:
        return "Mini Truck"
    else:
        return "Truck"

# ==============================
# Preprocessing
# ==============================
# Merge distance and contextual attributes
context_cols = [
    'supplier_id', 'distance_from_inventory_km', 'road_type', 'traffic_condition',
    'avg_speed_kmph', 'fuel_efficiency_factor', 'weather_condition', 'region_type'
]
suppliers = suppliers.merge(distance_df[context_cols], on='supplier_id', how='left')

# Merge emissions
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')

# Fill missing values
suppliers.fillna({
    'fuel_cost_per_km': 0,
    'co2_per_km': CO2_PER_KM_DEFAULT,
    'spoilage_rate_per_km': 0.001,
    'distance_from_inventory_km': 50,
    'avg_speed_kmph': 30,
    'fuel_efficiency_factor': 1.0,
    'road_type': 'highway',
    'traffic_condition': 'moderate',
    'weather_condition': 'clear',
    'region_type': 'urban'
}, inplace=True)

# Adjust transport cost using efficiency factor
suppliers['transport_cost'] = ((suppliers['distance_from_inventory_km'] / VEHICLE_MILEAGE)
                               * PETROL_PRICE * suppliers['fuel_efficiency_factor'])

# Adjust emissions
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']

# Estimate shelf life
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['shelf_life_days'] = suppliers.apply(
    lambda row: 90 if row['commodity'].lower() in HIGH_SHELF_COMMODITIES else row['shelf_life_days'],
    axis=1
)

# Local score combines cost and emissions
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['transport_cost'] + suppliers['emissions_kg']
