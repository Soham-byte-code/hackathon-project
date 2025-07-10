import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")

# === Styling ===
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

# === Prepare Data ===
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({'fuel_cost_per_km': 4.5, 'co2_per_km': 0.1, 'spoilage_rate_per_km': 0.0005, 'distance_from_inventory_km': 50}, inplace=True)

suppliers['transport_cost'] = suppliers['distance_from_inventory_km'] * suppliers['fuel_cost_per_km']
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
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

# === UI Header ===
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png", width=160)
st.markdown("<h2 style='color:#0071ce;'>Walmart FreshRoute AI</h2>", unsafe_allow_html=True)
st.markdown("Smarter sourcing, fresher produce, lower carbon footprint 🌿")

# === Inputs ===
commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
location = st.text_input("📍 Your Shop Location", placeholder="e.g. Wagholi, Pune")
qty_needed = st.number_input("🔢 Quantity Needed (in kg)", min_value=1, max_value=10000, value=50)

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
                st.info("♻️ Switched to supplier with lower CO₂ emissions.")

        # Simulate central supplier
        central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
        central_emissions = round(150 * 0.15, 2)

        # AI Prediction
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

        # Vehicle logic
        dist = best['distance_from_inventory_km']
        vehicle_emissions = {'EV Van': 0.03, 'Bike': 0.02, 'Tempo': 0.1, 'Mini Truck': 0.12, 'Truck': 0.18}
        current_mode = best.get('transport_mode', 'Tempo')
        current_emission = dist * vehicle_emissions.get(current_mode, 0.1)
        best_mode = min(vehicle_emissions, key=lambda m: dist * vehicle_emissions[m])
        best_emission = dist * vehicle_emissions[best_mode]

        # Extra metrics
        spoilage_rate_per_km = best.get('spoilage_rate_per_km', 0.001)
        spoilage_kg = dist * spoilage_rate_per_km * qty_needed  # FIXED to use requested quantity
        spoilage_pct = round((spoilage_kg / qty_needed) * 100, 2)
        travel_time = round(dist / 30, 2)
        route = f"{best.get('supply_region', 'Unknown')} → {location or 'Inventory'}"
        override = False
        if prediction == 0 and best['price_per_unit'] < central_price and current_emission < central_emissions and spoilage_pct < 10:
            decision = "✅ Source Locally (Overridden by Sustainability)"
            override = True

        total_cost = qty_needed * best['price_per_unit']

        # === Output ===
        st.success("📦 AI Decision Generated")
        st.markdown(f"""<div class='report-text'>
        <strong>Commodity:</strong> {best['commodity']}<br>
        <strong>Supplier:</strong> {best['supplier_name']} (ID: {best['supplier_id']})<br>
        <strong>Available Qty:</strong> {int(best['available_quantity_kg'])} kg<br>
        <strong>Requested Qty:</strong> {qty_needed} kg<br>
        <strong>Local Price:</strong> ₹{best['price_per_unit']} per kg<br>
        <strong>Total Cost:</strong> ₹{total_cost}<br>
        <strong>Transport Cost:</strong> ₹{round(best['transport_cost'], 2)}<br>
        <strong>CO₂ (Local):</strong> {round(current_emission, 2)} kg<br>
        <strong>CO₂ (Central):</strong> {central_emissions} kg<br>
        <strong>Spoilage:</strong> {round(spoilage_kg, 2)} kg ({spoilage_pct}%)<br>
        <strong>Shelf Life:</strong> {int(best['shelf_life_days'])} days<br>
        <strong>Override Applied:</strong> {override}<br>
        <strong>AI Decision:</strong> {decision}<br>
        <strong>Confidence:</strong> {round(confidence * 100, 2)}%<br>
        <strong>Current Vehicle:</strong> {current_mode}<br>
        <strong>Recommended Vehicle:</strong> {best_mode}<br>
        <strong>Emissions (Switched):</strong> {round(best_emission, 2)} kg<br>
        <strong>Route:</strong> {route}<br>
        <strong>Estimated Travel Time:</strong> {travel_time} hrs<br>
        <strong>Decision Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        </div>""", unsafe_allow_html=True)

        if st.button("🛒 Place Order"):
            st.balloons()
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: #e9f8e5; border-radius: 12px;'>
                <h3>✅ Order Placed Successfully!</h3>
                <p>You have placed an order for <strong>{qty_needed} kg of {best['commodity']}</strong>.</p>
                <p><strong>Supplier:</strong> {best['supplier_name']} (ID: {best['supplier_id']})</p>
                <p><strong>Delivery Route:</strong> {route}</p>
                <p><strong>Total Cost:</strong> ₹{total_cost}</p>
                <p><strong>ETA:</strong> {travel_time} hours</p>
                <p style='color: green; font-weight: bold;'>Thanks for choosing sustainability with Walmart FreshRoute AI 🌱</p>
            </div>
            """, unsafe_allow_html=True)
