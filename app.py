import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Walmart FreshRoute AI", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .main {
            background: url('https://source.unsplash.com/1600x900/?vegetables,market') no-repeat center center fixed;
            background-size: cover;
        }
        .block-container {
            padding: 2rem;
            background-color: rgba(255,255,255,0.95);
            border-radius: 10px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        }
        .stButton>button {
            background-color: #ffc220;
            color: #000;
            font-weight: 600;
            border-radius: 6px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #e6ac00;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align:center; padding: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png" width="180">
        <h1 style="margin-top: 10px; color:#0071ce;">Walmart FreshRoute AI</h1>
        <p style="font-size: 18px;">Smarter sourcing, fresher produce, lower carbon footprint 🌿</p>
    </div>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_data():
    suppliers = pd.read_excel("cleaned_supplier_data.xlsx")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    return suppliers, emissions, distance_df, inventory, demand

suppliers, emissions, distance_df, inventory, demand = load_data()

# Preprocess data
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({'fuel_cost_per_km': 5, 'co2_per_km': 0.15, 'spoilage_rate_per_km': 0.001, 'distance_from_inventory_km': 50}, inplace=True)
suppliers['transport_cost'] = suppliers['distance_from_inventory_km'] * suppliers['fuel_cost_per_km']
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['spoilage_kg'] = suppliers['distance_from_inventory_km'] * suppliers['spoilage_rate_per_km'] * suppliers['available_quantity_kg']
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['transport_cost'] + suppliers['emissions_kg'] + suppliers['spoilage_kg']

# Train model
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(1000, 2500, size=len(demand))
demand['decision'] = np.where((demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0)

features = ['modal_price', 'distance_km', 'transport_cost', 'local_price', 'central_price']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(demand[features], demand['decision'])

# UI Inputs
available_commodities = sorted(suppliers['commodity'].dropna().unique())
commodity = st.selectbox("🥦 Select a commodity:", available_commodities)
location = st.text_input("📍 Your Shop Location:", placeholder="e.g. Wagholi, Pune")

# Decision logic
if st.button("🚀 Get AI Decision"):
    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error(f"No suppliers found for '{commodity}'.")
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        co2_threshold = 10
        if best['emissions_kg'] > co2_threshold:
            low_emission_supplier = matched.loc[matched['emissions_kg'].idxmin()]
            if low_emission_supplier['emissions_kg'] < best['emissions_kg'] * 0.8:
                best = low_emission_supplier
                st.info("♻️ Supplier switched for lower CO₂ emissions.")

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

        override = False
        if prediction == 0 and best['price_per_unit'] < central_price and current_emission < central_emissions:
            decision = "✅ Source Locally (Overridden by Sustainability)"
            override = True

        spoilage_pct = round((best['spoilage_kg'] / best['available_quantity_kg']) * 100, 2)
        travel_time = round(dist / 30, 2)
        route = f"{best.get('supply_region', 'Unknown')} → {location or 'Inventory'}"

        st.success("📦 AI Decision Generated")
        st.markdown(f"""
        **Commodity:** {best['commodity']}  
        **Supplier:** {best['supplier_name']} (ID: {best['supplier_id']})  
        **Available Qty:** {int(best['available_quantity_kg'])} kg  
        **Distance to Shop:** {dist} km  
        **Local Price:** ₹{best['price_per_unit']}  
        **Central Price:** ₹{central_price}  
        **Transport Cost:** ₹{round(best['transport_cost'], 2)}  
        **CO₂ (Local):** {round(current_emission, 2)} kg  
        **CO₂ (Central):** {central_emissions} kg  
        **Spoilage:** {round(best['spoilage_kg'], 2)} kg ({spoilage_pct}%)  
        **Shelf Life:** {int(best['shelf_life_days'])} days  
        **Override Applied:** {override}  
        **AI Decision:** {decision}  
        **Confidence:** {round(confidence * 100, 2)}%  
        **Current Vehicle:** {current_mode}  
        **Recommended Vehicle:** {best_mode}  
        **Emissions (Switched):** {round(best_emission, 2)} kg  
        **Route:** {route}  
        **Estimated Travel Time:** {travel_time} hrs  
        **Decision Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        """)
