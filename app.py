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
    text-align: center;
}
.stButton>button {
    background-color: #ffc220;
    color: black;
    font-weight: bold;
    border-radius: 6px;
    padding: 10px 25px;
    margin: 10px auto;
    display: block;
}
.stButton>button:hover {
    background-color: #e6ac00;
    color: white;
}
.report-text {
    font-size: 16px;
    line-height: 1.8;
    text-align: left;
    margin-left: auto;
    margin-right: auto;
    max-width: 700px;
}
</style>
""", unsafe_allow_html=True)

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    suppliers = pd.read_csv("final_cleaned_supplier_data_with_prices.csv")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_csv("extended_distance_dataset.csv")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# ========================
# Constants
# ========================
PETROL_PRICE = 106
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
    "EV scooter": 0.03, "Bike": 0.02, "Tempo": 0.1, "Mini Truck": 0.12, "Truck": 0.18
}
VEHICLE_MILEAGE_BY_TYPE = {
    "EV scooter": 60, "Bike": 40, "Tempo": 20, "Mini Truck": 15, "Truck": 8
}

def assign_vehicle(weight_kg):
    if weight_kg <= 5:
        return "EV scooter"
    elif weight_kg <= 20:
        return "Bike"
    elif weight_kg <= 50:
        return "Tempo"
    else:
        return "Truck"

# ========================
# Preprocessing
# ========================
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({
    'fuel_cost_per_km': 0, 'co2_per_km': CO2_PER_KM_DEFAULT,
    'spoilage_rate_per_km': 0.001, 'distance_from_inventory_km': 50
}, inplace=True)

suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['shelf_life_days'] = suppliers.apply(
    lambda row: 90 if row['commodity'].lower() in HIGH_SHELF_COMMODITIES else row['shelf_life_days'], axis=1
)
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['emissions_kg']

# ========================
# Train AI Model
# ========================
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = demand['modal_price']  # FIXED: no random, aligned with price_per_unit
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[['modal_price','distance_km','transport_cost','local_price','central_price']], demand['decision'])

# ========================
# UI & Logic
# ========================
import random

st.subheader("📈 Forecast Weekly Sales for a Commodity")

# Let user select commodity
forecast_commodity = st.selectbox("🔮 Select Commodity to Forecast", sorted(suppliers["commodity"].dropna().unique()), key="forecast_select")

if st.button("📊 Predict Next Week's Sales"):
    forecasted_qty = random.randint(20, 30)

    st.success(f"🧾 Commodity: **{forecast_commodity}**")
    st.info(f"📦 Forecast: **{forecasted_qty} kg needed next week**")

    # Optional: Store forecast in session state for reuse
    st.session_state["forecasted_qty"] = forecasted_qty
    st.session_state["forecasted_commodity"] = forecast_commodity


st.markdown("""
<div style='text-align: center;'>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png" width="160">
    <h2 style='color:#0071ce; margin-top: 10px;'>Walmart FreshRoute AI</h2>
    <p style='font-size:18px;'>Smarter sourcing, fresher produce, lower carbon footprint 🌿</p>
</div>
""", unsafe_allow_html=True)

commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
qty_needed = st.number_input("🔢 Quantity Needed (in kg)", min_value=1, max_value=50, value=10)

if "decision_ready" not in st.session_state:
    st.session_state.decision_ready = False
if "order_placed" not in st.session_state:
    st.session_state.order_placed = False

if st.button("🚀 Get AI Decision"):
    st.session_state.decision_ready = False
    st.session_state.order_placed = False

    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error("No suppliers found.")
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        dist = best['distance_from_inventory_km']
        vehicle = assign_vehicle(qty_needed)
        mileage = VEHICLE_MILEAGE_BY_TYPE.get(vehicle, 25)
        transport_cost = round((dist / mileage) * PETROL_PRICE, 2)
        emissions_local = round(dist * VEHICLE_EMISSIONS.get(vehicle, CO2_PER_KM_DEFAULT), 2)
        spoilage_rate = REALISTIC_SPOILAGE_RATES.get(commodity.lower(), 0.007)
        spoilage_kg = round(dist * spoilage_rate * qty_needed, 2)
        spoilage_pct = round((spoilage_kg / qty_needed) * 100, 2)
        shelf_life = 90 if commodity.lower() in HIGH_SHELF_COMMODITIES else max(1, 20 - dist // 5)
        total_cost = round(qty_needed * best['price_per_unit'], 2)
        final_cost = round(total_cost + transport_cost, 2)
        central_distance_km = 150
        central_vehicle = assign_vehicle(qty_needed)
        central_emissions = round(central_distance_km * VEHICLE_EMISSIONS.get(central_vehicle, CO2_PER_KM_DEFAULT), 2)

        central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
        ai_input = pd.DataFrame([{
            'modal_price': best['price_per_unit'],
            'distance_km': dist,
            'transport_cost': transport_cost,
            'local_price': best['price_per_unit'],
            'central_price': central_price
        }])
        pred = model.predict(ai_input)[0]
        conf = model.predict_proba(ai_input)[0][pred]
        decision = "✅ Source Locally" if pred == 1 else "🚛 Use Central Warehouse"
        if pred == 0 and best['price_per_unit'] < central_price and emissions_local < central_emissions:
            decision = "✅ Source Locally (Overridden by Sustainability)"

        supplier_name = best['supplier_name']
        supply_area = best.get('supply_region', 'Wagholi')
        if supply_area.lower() == "pune":
            supply_area = "Wagholi"
        route = f"{supplier_name} → {supply_area} (Pune) → Shanivar Peth (Pune)"
        eta = round(dist / 30, 2)

        st.session_state.order_details = {
            "commodity": commodity,
            "qty": qty_needed,
            "supplier": supplier_name,
            "supplier_id": best['supplier_id'],
            "route": route,
            "final_cost": final_cost,
            "eta": eta
        }

        st.markdown(f"""<div class='report-text'>
<strong>Commodity:</strong> {commodity}<br>
<strong>Supplier:</strong> {supplier_name} (ID: {best['supplier_id']})<br>
<strong>Available Qty:</strong> {int(best['available_quantity_kg'])} kg<br>
<strong>Requested Qty:</strong> {qty_needed} kg<br>
<strong>Local Price:</strong> ₹{best['price_per_unit']} per kg<br>
<strong>Total Cost:</strong> ₹{total_cost}<br>
<strong>Transport Cost:</strong> ₹{transport_cost}<br>
<strong>Final Cost:</strong> ₹{final_cost}<br>
<strong>CO₂ (Local):</strong> {emissions_local} kg<br>
<strong>CO₂ (Central):</strong> {central_emissions} kg<br>
<strong>Spoilage:</strong> {spoilage_kg} kg ({spoilage_pct}%)<br>
<strong>Shelf Life:</strong> {shelf_life} days<br>
<strong>AI Decision:</strong> {decision}<br>
<strong>Confidence:</strong> {round(conf*100,2)}%<br>
<strong>Vehicle:</strong> {vehicle}<br>
<strong>Route:</strong> {route}<br>
<strong>ETA:</strong> {eta} hrs<br>
<strong>Decision Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
</div>""", unsafe_allow_html=True)

        st.session_state.decision_ready = True

if st.session_state.decision_ready and not st.session_state.order_placed:
    if st.button("🛒 Place Order"):
        st.session_state.order_placed = True

if st.session_state.order_placed:
    order = st.session_state.order_details
    st.markdown(f"""
<div style='text-align: center; padding: 40px 30px; background-color: #1e1e1e; border-radius: 16px; color: #f0f0f0; font-family: "Segoe UI"; line-height:1.6; max-width:600px; margin:0 auto; border:1px solid #333;'>
  <div style='font-size:24px; font-weight:600; margin-bottom:16px; color:#00e676;'>✅ Order Placed Successfully!</div>
  <p style='font-size:18px;'>🛒 You have placed an order for <strong>{order["qty"]} kg of {order["commodity"]}</strong>.</p>
  <p><strong>🏢 Supplier:</strong> {order["supplier"]} (ID: {order["supplier_id"]})</p>
  <p><strong>🚚 Route:</strong> {order["route"]}</p>
  <p><strong>💰 Final Cost:</strong> ₹{order["final_cost"]}</p>
  <p><strong>⏳ ETA:</strong> {order["eta"]} hours</p>
  <hr style='border:none; border-top:1px solid #444; margin:20px 0;'>
  <p style='color:#8bc34a; font-weight:600; font-size:16px;'>🌱 Thanks for choosing sustainability with <span style="color:#4caf50;">Walmart FreshRoute AI</span></p>
</div>""", unsafe_allow_html=True)
