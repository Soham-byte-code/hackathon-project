import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# === Simple Password Protection ===
def check_password():
    correct_password = "Rait@123"
    password = st.text_input("🔒 Enter Password to Access", type="password")
    if password == "":
        st.stop()
    elif password != correct_password:
        st.error("❌ Incorrect password.")
        st.stop()

check_password()

# === Config ===
st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; color: #222 !important; }
.stButton>button {
    background-color: #ffc220; color: black; font-weight: bold;
    border-radius: 6px; padding: 10px 25px;
}
.stButton>button:hover { background-color: #e6ac00; color: white; }
.report-text { font-size: 16px; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)

# === Load Data ===
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# === Constants ===
PETROL_PRICE = 106
VEHICLE_MILEAGE = 15
CO2_PER_KM_DEFAULT = 0.15
FIXED_QTY = 50
VEHICLE_EMISSIONS = {
    'Bike': 0.02,
    'Tempo': 0.1,
    'Mini Truck': 0.12,
    'Truck': 0.18
}

# Spoilage Rates
REALISTIC_SPOILAGE_RATES = {
    "tomato": 0.01, "onion": 0.003, "potato": 0.004, "cabbage": 0.005,
    "spinach": 0.012, "cauliflower": 0.006, "carrot": 0.002,
    "banana": 0.018, "mango": 0.02, "grapes": 0.015,
    "almonds": 0.0005, "dry fruits": 0.0008, "default": 0.007
}

# Shelf Life Categories
GRAINS_AND_NUTS = ["almonds", "dry fruits", "pulses", "rice", "wheat"]
VEG_AND_FRUITS = ["tomato", "onion", "potato", "cabbage", "spinach", "banana", "mango", "grapes", "carrot", "cauliflower"]

# === Data Preprocessing ===
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
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['transport_cost'] + suppliers['emissions_kg']

# === Train AI Model ===
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(1000, 2500, size=len(demand))
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0)

features = ['modal_price', 'distance_km', 'transport_cost', 'local_price', 'central_price']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[features], demand['decision'])

# === UI ===
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png", width=160)
st.markdown("<h2 style='color:#0071ce;'>Walmart FreshRoute AI</h2>", unsafe_allow_html=True)
st.markdown("Smarter sourcing, fresher produce, lower carbon footprint 🌿")

commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
location = "Shanivar Peth"
qty_needed = FIXED_QTY
st.markdown(f"🔢 Quantity Fixed at: **{qty_needed} kg**")

# === Decision Button ===
if st.button("🚀 Get AI Decision"):
    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error("No suppliers found.")
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
        current_mode = 'Bike' if qty_needed <= 20 else 'Tempo' if qty_needed <= 60 else 'Mini Truck'
        current_emission = dist * VEHICLE_EMISSIONS[current_mode]

        spoilage_rate = REALISTIC_SPOILAGE_RATES.get(commodity.lower(), REALISTIC_SPOILAGE_RATES['default'])
        spoilage_kg = round(dist * spoilage_rate * qty_needed, 2)
        spoilage_pct = round((spoilage_kg / qty_needed) * 100, 2)

        if commodity.lower() in GRAINS_AND_NUTS:
            shelf_life = 60
        elif commodity.lower() in VEG_AND_FRUITS:
            shelf_life = 10
        else:
            shelf_life = 20

        travel_time = round(dist / 30, 2)
        total_cost = round(qty_needed * best['price_per_unit'], 2)
        final_cost = round(total_cost + best['transport_cost'], 2)
        route = f"{best.get('supply_region', 'Unknown')} → {location}"

        # === Final Report ===
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
        <strong>CO₂ (Local):</strong> {round(current_emission, 2)} kg<br>
        <strong>CO₂ (Central):</strong> {central_emissions} kg<br>
        <strong>Spoilage:</strong> {spoilage_kg} kg ({spoilage_pct}%)<br>
        <strong>Shelf Life:</strong> {shelf_life} days<br>
        <strong>AI Decision:</strong> {decision}<br>
        <strong>Confidence:</strong> {round(confidence * 100, 2)}%<br>
        <strong>Transport Vehicle:</strong> {current_mode}<br>
        <strong>Route:</strong> {route}<br>
        <strong>ETA:</strong> {travel_time} hrs<br>
        <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        </div>""", unsafe_allow_html=True)

        if st.button("🛒 Place Order"):
            st.balloons()
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: #e9f8e5; border-radius: 12px;'>
                <h3>✅ Order Placed Successfully!</h3>
                <p><strong>{qty_needed} kg of {best['commodity']}</strong> ordered from <strong>{best['supplier_name']}</strong> (ID: {best['supplier_id']}).</p>
                <p><strong>Route:</strong> {route}</p>
                <p><strong>Final Cost:</strong> ₹{final_cost}</p>
                <p><strong>ETA:</strong> {travel_time} hrs</p>
                <p style='color: green; font-weight: bold;'>Thank you for choosing Walmart FreshRoute AI 🌿</p>
            </div>
            """, unsafe_allow_html=True)

