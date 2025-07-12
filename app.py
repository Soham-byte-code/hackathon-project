import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Prophet for forecasting
try:
    from prophet import Prophet
except ImportError:
    st.error("Please install Prophet: pip install prophet")
    st.stop()

# ======================== #
#  Password Protection     #
# ======================== #
def check_password():
    correct_password = "Rait@123"
    password = st.text_input("🔒 Enter Password to Access", type="password")
    if password == "":
        st.stop()
    elif password != correct_password:
        st.error("❌ Incorrect password. Please try again.")
        st.stop()

check_password()

# ======================== #
#      Load Data           #
# ======================== #
@st.cache_data
def load_data():
    suppliers = pd.read_csv("final_cleaned_supplier_data_with_prices.csv")
    emissions = pd.read_csv("transport_route_emissions.csv")
    distance_df = pd.read_csv("extended_distance_dataset.csv")
    inventory = pd.read_excel("inventory location.xlsx")
    demand = pd.read_csv("demand.csv")
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")
    return suppliers, emissions, distance_df, inventory, demand, train_df, test_df

suppliers, emissions, distance_df, inventory, demand, train_df, test_df = load_data()

# ======================== #
#  Constants & Settings    #
# ======================== #
PETROL_PRICE = 106
CO2_PER_KM_DEFAULT = 0.15
REALISTIC_SPOILAGE_RATES = {
    "tomato": 0.01, "onion": 0.003, "potato": 0.004,
    "cabbage": 0.005, "spinach": 0.012, "cauliflower": 0.006,
    "carrot": 0.002, "banana": 0.018, "mango": 0.02,
    "grapes": 0.015, "almonds": 0.0005, "dry fruits": 0.0008, "default": 0.007
}
HIGH_SHELF_COMMODITIES = ["rice", "wheat", "dal", "pulses", "almonds", "dry fruits", "grains", "nuts"]
VEHICLE_EMISSIONS = {
    "EV scooter": 0.03, "Bike": 0.02, "Tempo": 0.1, "Mini Truck": 0.12, "Truck": 0.18
}
VEHICLE_MILEAGE = {
    "EV scooter": 60, "Bike": 40, "Tempo": 20, "Mini Truck": 15, "Truck": 8
}

def assign_vehicle(weight):
    if weight <= 5: return "EV scooter"
    elif weight <= 20: return "Bike"
    elif weight <= 50: return "Tempo"
    else: return "Truck"

# ======================== #
#     Prophet Forecast     #
# ======================== #
def forecast_sales(commodity):
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    df = train_df[train_df["Product_Name"].str.lower() == commodity.lower()]
    if df.empty:
        return None
    weekly = df.set_index("Date")["Quantity_Sold"].resample("W-MON").sum()
    df_p = weekly.reset_index().rename(columns={"Date": "ds", "Quantity_Sold": "y"})
    model = Prophet(weekly_seasonality=True)
    model.fit(df_p)
    future = model.make_future_dataframe(periods=1, freq='W-MON')
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(1)

# ======================== #
#     Preprocessing        #
# ======================== #
suppliers = suppliers.merge(distance_df[['supplier_id', 'distance_from_inventory_km']], on='supplier_id', how='left')
suppliers = suppliers.merge(emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']], on='supplier_id', how='left')
suppliers.fillna({'fuel_cost_per_km': 0, 'co2_per_km': CO2_PER_KM_DEFAULT, 'spoilage_rate_per_km': 0.001, 'distance_from_inventory_km': 50}, inplace=True)
suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
suppliers['shelf_life_days'] = np.where(suppliers['commodity'].str.lower().isin(HIGH_SHELF_COMMODITIES), 90, np.maximum(1, 20 - suppliers['distance_from_inventory_km'] // 5))
suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['emissions_kg']

# ======================== #
#  Train RandomForest AI   #
# ======================== #
demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
demand['transport_cost'] = demand['distance_km'] * 4
demand['central_price'] = demand['modal_price'] + demand['transport_cost']
merged = demand.merge(suppliers[['commodity', 'price_per_unit']], on='commodity', how='left')
merged.rename(columns={'price_per_unit': 'local_price'}, inplace=True)
merged['decision'] = np.where((merged['local_price'] < merged['central_price']) & (merged['transport_cost'] < 400), 1, 0)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(merged[['modal_price','distance_km','transport_cost','local_price','central_price']], merged['decision'])

# ======================== #
#         UI               #
# ======================== #
st.markdown("<h2 style='color:#0071ce;'>Walmart FreshRoute AI</h2>", unsafe_allow_html=True)
commodity = st.selectbox("🥦 Select a commodity:", sorted(suppliers['commodity'].dropna().unique()))
qty_needed = st.number_input("🔢 Quantity Needed (in kg)", min_value=1, max_value=50, value=10)

# Show forecast
forecast = forecast_sales(commodity)
if forecast is not None:
    date = forecast['ds'].values[0]
    qty = forecast['yhat'].values[0]
    st.success(f"📦 Predicted Sales for next week ({str(date)[:10]}): **{int(qty)} units**")

if st.button("🚀 Get AI Decision"):
    matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
    if matched.empty:
        st.error("❌ No suppliers found.")
    else:
        best = matched.loc[matched['local_score'].idxmin()]
        dist = best['distance_from_inventory_km']
        vehicle = assign_vehicle(qty_needed)
        mileage = VEHICLE_MILEAGE[vehicle]
        transport_cost = round((dist / mileage) * PETROL_PRICE, 2)
        emissions_local = round(dist * VEHICLE_EMISSIONS[vehicle], 2)
        spoilage = REALISTIC_SPOILAGE_RATES.get(commodity.lower(), 0.007)
        spoilage_kg = round(dist * spoilage * qty_needed, 2)
        shelf = best['shelf_life_days']
        total = round(qty_needed * best['price_per_unit'], 2)
        final = round(total + transport_cost, 2)
        central_em = round(150 * CO2_PER_KM_DEFAULT, 2)

        central_price = round(best['price_per_unit'] * 2.2, 2)
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
        if pred == 0 and best['price_per_unit'] < central_price and emissions_local < central_em:
            decision = "✅ Source Locally (Overridden by Sustainability)"

        st.markdown(f"""
        <div style='text-align:left; font-size:17px; padding:15px; border:1px solid #ccc; border-radius:10px; max-width:720px; margin: auto;'>
        <b>Commodity:</b> {commodity}<br>
        <b>Supplier:</b> {best['supplier_name']} (ID: {best['supplier_id']})<br>
        <b>Available Qty:</b> {int(best['available_quantity_kg'])} kg<br>
        <b>Requested Qty:</b> {qty_needed} kg<br>
        <b>Local Price:</b> ₹{best['price_per_unit']} /kg<br>
        <b>Total Cost:</b> ₹{total}<br>
        <b>Transport Cost:</b> ₹{transport_cost}<br>
        <b>Final Cost:</b> ₹{final}<br>
        <b>CO₂ (Local):</b> {emissions_local} kg<br>
        <b>CO₂ (Central):</b> {central_em} kg<br>
        <b>Spoilage:</b> {spoilage_kg} kg<br>
        <b>Shelf Life:</b> {shelf} days<br>
        <b>AI Decision:</b> {decision}<br>
        <b>Confidence:</b> {round(conf*100,2)}%<br>
        <b>Vehicle:</b> {vehicle}<br>
        <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

