import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# === Load all datasets ===
@st.cache_data
def load_data():
    demand = pd.read_csv("demand.csv")
    suppliers = pd.read_csv("pune_supplier_dataset_100_entries.csv")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_excel("Distance Dataset.xlsx")
    inventory = pd.read_excel("inventory location.xlsx")
    return demand, suppliers, emissions, distance_df, inventory

demand, suppliers, emissions, distance_df, inventory = load_data()

# === Merge supplier info with distance + emissions ===
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')

suppliers['fuel_cost_per_km'].fillna(5, inplace=True)
suppliers['co2_per_km'].fillna(0.15, inplace=True)
suppliers['spoilage_rate_per_km'].fillna(0.001, inplace=True)
suppliers['distance_from_inventory_km'].fillna(50, inplace=True)

suppliers['transport_cost'] = suppliers['distance_from_inventory_km'] * suppliers['fuel_cost_per_km']
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['spoilage_kg'] = suppliers['distance_from_inventory_km'] * suppliers['spoilage_rate_per_km'] * suppliers['available_quantity_kg']
suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
suppliers['local_score'] = (
    suppliers['price_per_unit'] +
    suppliers['transport_cost'] +
    suppliers['emissions_kg'] +
    suppliers['spoilage_kg']
)

# === Train AI model ===
np.random.seed(42)
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
demand['local_price'] = np.random.randint(1000, 2500, size=len(demand))
demand['decision'] = np.where(
    (demand['local_price'] < demand['central_price']) & (demand['transport_cost'] < 400), 1, 0
)

features = ['modal_price', 'distance_km', 'transport_cost', 'local_price', 'central_price']
X = demand[features]
y = demand['decision']
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

# === Inventory location (e.g. Wagholi) ===
inventory_name = inventory.loc[0, "name"]

vehicle_emissions = {
    'EV Van': 0.03,
    'Bike': 0.02,
    'Tempo': 0.1,
    'Mini Truck': 0.12,
    'Truck': 0.18
}

# === Streamlit UI ===
st.title("FreshRoute AI Decision System")
user_input = st.text_input("Enter a commodity (e.g. Tomato, Onion):").strip().lower()

if user_input:
    matched = suppliers[suppliers['commodity'].str.lower() == user_input]

    if matched.empty:
        st.error(f"No suppliers found for '{user_input}'.")
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
        central_emissions = round(150 * 0.15, 2)

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

        # Vehicle & emissions
        distance = best['distance_from_inventory_km']
        current_mode = best.get('transport_mode', 'Tempo')
        current_factor = vehicle_emissions.get(current_mode, 0.15)
        current_emission = distance * current_factor

        best_mode = current_mode
        best_emission = current_emission
        for mode, factor in vehicle_emissions.items():
            est = distance * factor
            if est < best_emission:
                best_mode = mode
                best_emission = est

        co2_savings = round(current_emission - best_emission, 2)
        spoilage_pct = round((best['spoilage_kg'] / best['available_quantity_kg']) * 100, 2)
        travel_time_hr = round(distance / 30, 2)
        route = f"{best.get('supply_region', 'Unknown')} → {inventory_name}"

        # Override logic
        override = False
        if prediction == 0:
            if best['price_per_unit'] < central_price and current_emission < central_emissions and spoilage_pct < 10:
                decision = "✅ Source Locally (Overridden by Sustainability Rule)"
                override = True

        # === Final Output ===
        st.subheader("📊 AI Decision Report")
        st.markdown(f"""
        **Commodity**: {best['commodity']}  
        **Supplier**: {best['supplier_name']} (ID: {best['supplier_id']})  
        **Available Qty**: {int(best['available_quantity_kg'])} kg  
        **Distance to {inventory_name}**: {distance} km  
        **Local Price**: ₹{best['price_per_unit']}, **Central Price**: ₹{central_price}  
        **Local Transport Cost**: ₹{round(best['transport_cost'], 2)}  
        **CO₂ Emissions (Local)**: {round(current_emission, 2)} kg  
        **CO₂ Emissions (Central)**: {central_emissions} kg  
        **Spoilage Expected**: {round(best['spoilage_kg'], 2)} kg ({spoilage_pct}%)  
        **Estimated Shelf-Life**: {int(best['shelf_life_days'])} days  
        **Override Applied**: {override}  
        **AI Decision**: {decision}  
        **Confidence**: {round(confidence * 100, 2)}%  
        **Current Vehicle**: {current_mode}  
        **Recommended Vehicle**: {best_mode}  
        **Emissions If Switched**: {round(best_emission, 2)} kg  
        **CO₂ Savings If Switched**: {co2_savings} kg  
        **Suggested Route**: {route}  
        **Estimated Travel Time**: {travel_time_hr} hours  
        **Decision Time**: {datetime.now().isoformat()}  
        """)
