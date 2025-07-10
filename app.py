import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")

# ==== Improved Styling for UI and Text Visibility ====
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
}
.block-container {
    background-color: #fefefe;
    border-radius: 12px;
    padding: 2rem;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: #222;
}
.report-text {
    font-size: 16px;
    line-height: 1.6;
    color: #333;
}
.stButton>button {
    background-color: #ffc220;
    color: black;
    font-weight: bold;
    padding: 10px 25px;
    border-radius: 6px;
}
.stButton>button:hover {
    background-color: #e6ac00;
}
</style>
""", unsafe_allow_html=True)

# ==== Load Data ====
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# ==== Data Preprocessing ====
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({'fuel_cost_per_km': 5, 'co2_per_km': 0.15, 'spoilage_rate_per_km': 0.001, 'distance_from_inventory_km': 50}, inplace=True)
suppliers['transport_cost'] = suppliers['distance_from_inventory_km'] * suppliers['fuel_cost_per_km']
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['spoilage_kg'] = suppliers['distance_from_inventory_km'] * suppliers['spoilage_rate_per_km'] * suppliers['available_quantity_kg']
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['transport_cost'] + suppliers['emissions_kg'] + suppliers['spoilage_kg']

# ==== Train Model ====
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(1000, 2500, size=len(demand))
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0)
features = ['modal_price', 'distance_km', 'transport_cost', 'local_price', 'central_price']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[features], demand['decision'])

# ==== UI ====
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png", width=150)
st.title("Walmart FreshRoute AI 🌿")
st.markdown("### Smarter sourcing, fresher produce, lower carbon footprint 🌎")

# === Inputs ===
commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
location = st.text_input("📍 Your Shop Location", placeholder="e.g. Wagholi, Pune")
qty_needed = st.number_input("🔢 Quantity Needed (in kg)", min_value=1, max_value=10000, value=50)

# === Main Logic ===
if st.button("🚀 Get AI Decision"):
    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error("No suppliers found.")
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        if best['emissions_kg'] > 10:
            lower_emission = matched.loc[matched['emissions_kg'].idxmin()]
            if lower_emission['emissions_kg'] < best['emissions_kg'] * 0.8:
                best = lower_emission
                st.info("♻️ Switched to lower CO₂ supplier.")

        central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
        central_emissions = round(150 * 0.15, 2)
        input_df = pd.DataFrame([{
            'modal_price': best['price_per_unit'],
            'distance_km': best['distance_from_inventory_km'],
            'transport_cost': best['transport_cost'],
            'local_price': best['price_per_unit'],
            'central_price': central_price
        }])
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]
        decision = "✅ Source Locally" if prediction == 1 else "🚛 Use Central Warehouse"

        vehicle_emissions = {'EV Van': 0.03, 'Bike': 0.02, 'Tempo': 0.1, 'Mini Truck': 0.12, 'Truck': 0.18}
        dist = best['distance_from_inventory_km']
        current_mode = best.get('transport_mode', 'Tempo')
        current_emission = dist * vehicle_emissions.get(current_mode, 0.15)
        best_mode = min(vehicle_emissions, key=lambda m: dist * vehicle_emissions[m])
        best_emission = dist * vehicle_emissions[best_mode]
        spoilage_pct = round((best['spoilage_kg'] / best['available_quantity_kg']) * 100, 2)
        travel_time = round(dist / 30, 2)
        route = f"{best.get('supply_region', 'Unknown')} → {location or 'Inventory'}"
        override = False
        if prediction == 0 and best['price_per_unit'] < central_price and current_emission < central_emissions:
            decision = "✅ Source Locally (Overridden by Sustainability)"
            override = True

        total_cost = qty_needed * best['price_per_unit']

        st.success("📦 AI Decision Generated")
        st.markdown(f"""
<div class="report-text">
<b>Commodity:</b> {best['commodity']}<br>
<b>Supplier:</b> {best['supplier_name']} (ID: {best['supplier_id']})<br>
<b>Available Qty:</b> {int(best['available_quantity_kg'])} kg<br>
<b>Requested Qty:</b> {qty_needed} kg<br>
<b>Local Price:</b> ₹{best['price_per_unit']} per kg<br>
<b>Total Cost:</b> ₹{total_cost}<br>
<b>Transport Cost:</b> ₹{round(best['transport_cost'], 2)}<br>
<b>CO₂ (Local):</b> {round(current_emission, 2)} kg<br>
<b>CO₂ (Central):</b> {central_emissions} kg<br>
<b>Spoilage:</b> {round(best['spoilage_kg'], 2)} kg ({spoilage_pct}%)<br>
<b>Shelf Life:</b> {int(best['shelf_life_days'])} days<br>
<b>Override Applied:</b> {override}<br>
<b>AI Decision:</b> {decision}<br>
<b>Confidence:</b> {round(confidence * 100, 2)}%<br>
<b>Current Vehicle:</b> {current_mode}<br>
<b>Recommended Vehicle:</b> {best_mode}<br>
<b>Emissions (Switched):</b> {round(best_emission, 2)} kg<br>
<b>Route:</b> {route}<br>
<b>Estimated Travel Time:</b> {travel_time} hrs<br>
<b>Decision Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
</div>
""", unsafe_allow_html=True)

        # === Place Order Button ===
        if st.button("🛒 Place Order"):
            st.balloons()
            st.markdown(f"""
## ✅ Order Placed!
Your order for <b>{qty_needed} kg of {best['commodity']}</b> has been placed at  
<b>🧑‍🌾 {best['supplier_name']}</b> (ID: {best['supplier_id']})  
📍 Route: {route}  
💸 <b>Total Cost:</b> ₹{total_cost}  
🚚 <b>ETA:</b> {travel_time} hours  
🌿 Thanks for choosing <b>Walmart FreshRoute AI</b>!
""", unsafe_allow_html=True)


