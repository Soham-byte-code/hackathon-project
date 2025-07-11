import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ========================
# Password Protection
# ========================
def check_password():
    correct_password = "Rait@123"
    password = st.text_input("🔒 Enter Password to Access", type="password")
    if password == "":
        st.stop()
    elif password != correct_password:
        st.error("❌ Incorrect password. Please try again.")
        st.stop()

check_password()

# ========================
# Page Setup
# ========================
st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")
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

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# ========================
# Constants
# ========================
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
    "EV Van": 0.03,
    "Bike": 0.02,
    "Tempo": 0.1,
    "Mini Truck": 0.12,
    "Truck": 0.18
}


def assign_vehicle(weight_kg):
    if weight_kg <= 5:
        return "EV scooter"
    elif weight_kg <= 20:
        return "Bike"
    elif weight_kg <= 50:
        return "Tempo"
    else:
        return "Not Supported"
 
# ========================
# Preprocessing
# ========================
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({
    'fuel_cost_per_km': 0,
    'co2_per_km': CO2_PER_KM_DEFAULT,
    'spoilage_rate_per_km': 0.001,
    'distance_from_inventory_km': 50
}, inplace=True)

suppliers['transport_cost'] = (suppliers['distance_from_inventory_km'] / VEHICLE_MILEAGE) * PETROL_PRICE
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']

# Shelf life logic
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['shelf_life_days'] = suppliers.apply(
    lambda row: 90 if row['commodity'].lower() in HIGH_SHELF_COMMODITIES else row['shelf_life_days'],
    axis=1
)

suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['transport_cost'] + suppliers['emissions_kg']

# ========================
# Train AI Model
# ========================
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(1000, 2500, size=len(demand))
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0)

features = ['modal_price', 'distance_km', 'transport_cost', 'local_price', 'central_price']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[features], demand['decision'])

# ========================
# UI
# ========================
st.markdown("""
<div style='text-align: center; margin-top: 20px; margin-bottom: 30px;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png' width='160'/>
    <h2 style='color:#0071ce; margin-top: 10px;'>Walmart FreshRoute AI</h2>
    <p style='font-size: 16px; color: #ccc;'>Smarter sourcing, fresher produce, lower carbon footprint 🌿</p>
</div>
""", unsafe_allow_html=True)


commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
location = "Shanivar Peth"
qty_needed = st.number_input("🔢 Quantity Needed (in kg)", min_value=1, max_value=50, value=50)

# ========================
# Session State
# ========================
if "decision_done" not in st.session_state:
    st.session_state.decision_done = False
if "order_placed" not in st.session_state:
    st.session_state.order_placed = False

if st.button("🚀 Get AI Decision"):
    st.session_state.decision_done = True
    st.session_state.order_placed = False

if st.session_state.decision_done:
    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error("No suppliers found.")
        st.session_state.decision_done = False
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        if best['emissions_kg'] > 10:
            alt = matched.loc[matched['emissions_kg'].idxmin()]
            if alt['emissions_kg'] < best['emissions_kg'] * 0.8:
                best = alt
                st.info("♻️ Switched to lower CO₂ supplier.")

        central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
        central_emissions = round(150 * CO2_PER_KM_DEFAULT, 2)

        ai_input = pd.DataFrame([{
            'modal_price': best['price_per_unit'],
            'distance_km': best['distance_from_inventory_km'],
            'transport_cost': best['transport_cost'],
            'local_price': best['price_per_unit'],
            'central_price': central_price
        }])
        prediction = model.predict(ai_input)[0]
        confidence = model.predict_proba(ai_input)[0][prediction]
        decision = "✅ Source Locally" if prediction == 1 else "🚛 Use Central Warehouse"

        dist = best['distance_from_inventory_km']
        current_mode = assign_vehicle(qty_needed)
        best_mode = assign_vehicle(qty_needed)
        current_emission = round(dist * VEHICLE_EMISSIONS[current_mode], 2)

        spoilage_rate = REALISTIC_SPOILAGE_RATES.get(commodity.lower(), REALISTIC_SPOILAGE_RATES['default'])
        spoilage_kg = round(dist * spoilage_rate * qty_needed, 2)
        spoilage_pct = round((spoilage_kg / qty_needed) * 100, 2)

        travel_time = round(dist / 30, 2)
        total_cost = round(qty_needed * best['price_per_unit'], 2)
        final_cost = round(total_cost + best['transport_cost'], 2)
        route = f"{best.get('supply_region', 'Unknown')} (Pune) → Shanivar Peth (Pune)"

        if prediction == 0 and best['price_per_unit'] < central_price and current_emission < central_emissions:
            decision = "✅ Source Locally (Overridden by Sustainability)"

        # Show AI Report
        st.success("📦 AI Decision Generated")
        st.markdown(f"""<div class='report-text'>
        <strong>Commodity:</strong> {best['commodity']}<br>
        <strong>Supplier:</strong> {best['supplier_name']} (ID: {best['supplier_id']})<br>
        <strong>Available Qty:</strong> {int(best['available_quantity_kg'])} kg<br>
        <strong>Requested Qty:</strong> {qty_needed} kg<br>
        <strong>Local Price:</strong> ₹{best['price_per_unit']} per kg<br>
        <strong>Total Cost:</strong> ₹{total_cost}<br>
        <strong>Transport Cost:</strong> ₹{round(best['transport_cost'], 2)}<br>
        <strong>Final Cost:</strong> ₹{final_cost}<br>
        <strong>CO₂ (Local):</strong> {current_emission} kg<br>
        <strong>CO₂ (Central):</strong> {central_emissions} kg<br>
        <strong>Spoilage:</strong> {spoilage_kg} kg ({spoilage_pct}%)<br>
        <strong>Shelf Life:</strong> {int(best['shelf_life_days'])} days<br>
        <strong>AI Decision:</strong> {decision}<br>
        <strong>Confidence:</strong> {round(confidence * 100, 2)}%<br>
        <strong>Current Vehicle:</strong> {current_mode}<br>
        <strong>Recommended Vehicle:</strong> {best_mode}<br>
        <strong>Route:</strong> {route}<br>
        <strong>Estimated Travel Time:</strong> {travel_time} hrs<br>
        <strong>Decision Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        </div>""", unsafe_allow_html=True)

        if st.button("🛒 Place Order"):
            st.session_state.order_placed = True

if st.session_state.order_placed:
    st.markdown(f"""
    <div style='
        text-align: center;
        padding: 40px 30px;
        background-color: #1e1e1e;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #f0f0f0;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
        border: 1px solid #333;
    '>
        <div style='font-size: 24px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; justify-content: center; color: #00e676;'>
            ✅ <span style='margin-left: 10px;'>Order Placed Successfully!</span>
        </div>
        <p style='font-size: 18px;'>You have placed an order for <strong>{qty_needed} kg of {commodity}</strong>.</p>
        <p><strong>Supplier:</strong> {best['supplier_name']} <span style="color: #aaa;">(ID: {best['supplier_id']})</span></p>
        <p><strong>Delivery Route:</strong> {route}</p>
        <p><strong>Final Cost:</strong> ₹{final_cost}</p>
        <p><strong>ETA:</strong> {travel_time} hours</p>
        <hr style='margin: 20px 0; border: none; border-top: 1px solid #444;'>
        <p style='color: #8bc34a; font-weight: 600; font-size: 16px;'>
            🌱 Thanks for choosing sustainability with <span style="color: #4caf50;">Walmart FreshRoute AI</span>
        </p>
    </div>
    """, unsafe_allow_html=True)



