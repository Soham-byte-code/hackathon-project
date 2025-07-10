import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")

# Styling
st.markdown("""
<style>
html, body, [class*="css"] {font-family:'Segoe UI',sans-serif;color:#222!important;}
.stButton>button {background:#ffc220;color:black;font-weight:bold;padding:10px 25px;border-radius:6px;}
.stButton>button:hover {background:#e6ac00;color:white;}
.report-text {font-size:16px;line-height:1.8;}
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# Real petrol price
PETROL_PER_L = 103.50  # ₹/litre :contentReference[oaicite:8]{index=8}

# Merge data
suppliers = (suppliers
    .merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
    .merge(emissions[['supplier_id', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left'))

suppliers = suppliers.fillna({
    'co2_per_km': 0.12,
    'spoilage_rate_per_km': np.random.uniform(0.0002, 0.0005),
    'distance_from_inventory_km': 50
})

# Fuel costs per mode
FUEL_COST = {
    'Tempo': PETROL_PER_L / 8,
    'Mini Truck': PETROL_PER_L / 6,
    'Truck': PETROL_PER_L / 5,
    'EV Van': 0,
    'Bike': 0
}

# Compute logistics
suppliers['fuel_cost_per_km'] = suppliers['transport_mode'].map(FUEL_COST).fillna(PETROL_PER_L/7)
suppliers['transport_cost'] = (suppliers['distance_from_inventory_km'] * suppliers['fuel_cost_per_km']).clip(upper=2000)
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['spoilage_kg'] = (suppliers['distance_from_inventory_km'] * suppliers['spoilage_rate_per_km']
                            * suppliers['available_quantity_kg']).clip(upper=suppliers['available_quantity_kg'] * .2)
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['local_score'] = (suppliers['price_per_unit'] + suppliers['transport_cost']
                             + suppliers['emissions_kg'] + suppliers['spoilage_kg'])

# Train AI model
np.random.seed(42)
demand['distance_km'] = np.random.randint(10,150,size=len(demand))
demand['transport_cost'] = demand['distance_km'] * (PETROL_PER_L/7)
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(800,2200,size=len(demand))
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400),1,0)

features = ['modal_price','distance_km','transport_cost','local_price','central_price']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[features], demand['decision'])

# UI
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png",width=160)
st.markdown("<h2 style='color:#0071ce;'>Walmart FreshRoute AI 🌿</h2>", unsafe_allow_html=True)
commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
location = st.text_input("📍 Your Shop Location", placeholder="e.g. Wagholi, Pune")
qty = st.number_input("🔢 Quantity Needed (kg)", min_value=1, max_value=10000, value=50)

if st.button("🚀 Get AI Decision"):
    m = suppliers[suppliers['commodity'].str.lower()==commodity.lower()]
    if m.empty:
        st.error("No suppliers found.")
    else:
        best = m.loc[m['local_score'].idxmin()]
        if best['emissions_kg']>10:
            alt = m.loc[m['emissions_kg'].idxmin()]
            if alt['emissions_kg']<best['emissions_kg']*0.8:
                best = alt; st.info("♻️ Switched to lower CO₂ supplier.")

        central_price = round(best['price_per_unit'] * np.random.uniform(1.8,2.4),2)
        central_em = round(150*0.15,2)
        ai_input = pd.DataFrame([{
            'modal_price':best['price_per_unit'],
            'distance_km':best['distance_from_inventory_km'],
            'transport_cost':best['transport_cost'],
            'local_price':best['price_per_unit'],
            'central_price':central_price
        }])
        pred = model.predict(ai_input)[0]
        conf = model.predict_proba(ai_input)[0][pred]
        decision = "✅ Source Locally" if pred==1 else "🚛 Use Central Warehouse"
        dist = best['distance_from_inventory_km']
        veh_em = {'EV Van':0.03,'Bike':0.02,'Tempo':0.1,'Mini Truck':0.12,'Truck':0.18}
        curr = best.get('transport_mode','Tempo')
        curr_em = dist*veh_em.get(curr,0.15)
        best_mode = min(veh_em, key=lambda m:dist*veh_em[m])
        swap_em = dist*veh_em[best_mode]
        spoil_pct = round(best['spoilage_kg']/best['available_quantity_kg']*100,2)
        travel = round(dist/30,2)
        route = f"{best.get('supply_region','Unknown')} → {location or 'Inventory'}"
        override = False
        if pred==0 and best['price_per_unit']<central_price and curr_em<central_em:
            decision="✅ Source Locally (Overridden)"; override=True
        total = round(qty*best['price_per_unit'],2)

        st.success("📦 AI Decision")
        st.markdown(f"""<div class='report-text'>
        <b>Commodity:</b> {best['commodity']}<br>
        <b>Supplier:</b> {best['supplier_name']} (ID: {best['supplier_id']})<br>
        <b>Avail. Qty:</b> {int(best['available_quantity_kg'])} kg<br>
        <b>Requested:</b> {qty} kg<br>
        <b>Price per kg:</b> ₹{best['price_per_unit']}<br>
        <b>Total Cost:</b> ₹{total}<br>
        <b>Transport Cost:</b> ₹{round(best['transport_cost'],2)}<br>
        <b>CO₂ (Local):</b> {round(curr_em,2)} kg<br>
        <b>CO₂ (Central):</b> {central_em} kg<br>
        <b>Spoilage:</b> {round(best['spoilage_kg'],2)} kg ({spoil_pct}%)<br>
        <b>Shelf Life:</b> {int(best['shelf_life_days'])} days<br>
        <b>Override:</b> {override}<br>
        <b>Decision:</b> {decision} (Confidence: {round(conf*100,2)}%)<br>
        <b>Vehicle:</b> {curr} → {best_mode}<br>
        <b>Swap CO₂:</b> {round(swap_em,2)} kg<br>
        <b>Route:</b> {route}<br>
        <b>Travel Time:</b> {travel} hrs<br>
        <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        </div>""", unsafe_allow_html=True)

        if st.button("🛒 Place Order"):
            st.balloons()
            st.markdown(f"""<div style='text-align:center;padding:20px;background:#e9f8e5;border-radius:12px;'>
            <h3>✅ Order Placed!</h3>
            <p>{qty} kg of {best['commodity']} ordered from {best['supplier_name']}.</p>
            <p>Route: {route}</p><p>Total: ₹{total}</p><p>ETA: {travel} hrs</p>
            <p style='color:green;font-weight:bold;'>Thanks for choosing sustainability! 🌱</p>
            </div>""", unsafe_allow_html=True)
