import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# Prophet import with installation check
try:
    from prophet import Prophet
except ImportError:
    st.error("Please install Prophet using: pip install prophet")
    st.stop()

# --- Password Protection ---
def check_password():
    """Password protection for the app"""
    correct_password = "Rait@123"
    password = st.text_input("🔒 Enter Password to Access", type="password")
    if password == "":
        st.warning("Please enter a password")
        st.stop()
    elif password != correct_password:
        st.error("❌ Incorrect password")
        st.stop()

check_password()

# --- Styling ---
st.set_page_config(
    page_title="Walmart FreshRoute AI",
    page_icon="🌿",
    layout="centered"
)

custom_css = """
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
        margin: 10px auto;
    }
    .stButton>button:hover {
        background-color: #e6ac00;
        color: white;
    }
    .report-box {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        suppliers = pd.read_csv("final_cleaned_supplier_data_with_prices.csv")
        emissions = pd.read_csv("transport_route_emissions.csv") 
        distance_df = pd.read_csv("extended_distance_dataset.csv")
        inventory = pd.read_excel("inventory location.xlsx")
        demand = pd.read_csv("demand.csv")
        train_df = pd.read_csv("train_data.csv")
        return suppliers, emissions, distance_df, inventory, demand, train_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

suppliers, emissions, distance_df, inventory, demand, train_df = load_data()

# --- Constants ---
PETROL_PRICE = 106
CO2_PER_KM_DEFAULT = 0.15
REALISTIC_SPOILAGE_RATES = {
    "tomato": 0.01, "onion": 0.003, "potato": 0.004,
    "cabbage": 0.005, "spinach": 0.012, "cauliflower": 0.006,
    "carrot": 0.002, "banana": 0.018, "mango": 0.02,
    "grapes": 0.015, "almonds": 0.0005, "dry fruits": 0.0008,
    "default": 0.007
}
HIGH_SHELF_COMMODITIES = ["rice", "wheat", "dal", "pulses", 
                         "almonds", "dry fruits", "grains", "nuts"]
VEHICLE_EMISSIONS = {
    "EV scooter": 0.03, "Bike": 0.02, "Tempo": 0.1,
    "Mini Truck": 0.12, "Truck": 0.18
}
VEHICLE_MILEAGE_BY_TYPE = {
    "EV scooter": 60, "Bike": 40, "Tempo": 20,
    "Mini Truck": 15, "Truck": 8
}

def assign_vehicle(weight_kg):
    """Assign appropriate vehicle based on weight"""
    if weight_kg <= 5:
        return "EV scooter"
    elif weight_kg <= 20:
        return "Bike"
    elif weight_kg <= 50:
        return "Tempo"
    else:
        return "Truck"

# --- Data Preprocessing ---
def preprocess_data():
    """Clean and merge supplier data"""
    try:
        suppliers = suppliers.merge(
            distance_df[['supplier_id', 'distance_from_inventory_km']], 
            on='supplier_id', how='left'
        )
        suppliers = suppliers.merge(
            emissions[['supplier_id', 'fuel_cost_per_km', 'co2_per_km', 'spoilage_rate_per_km']],
            on='supplier_id', how='left'
        )
        suppliers.fillna({
            'fuel_cost_per_km': 0,
            'co2_per_km': CO2_PER_KM_DEFAULT,
            'spoilage_rate_per_km': 0.001,
            'distance_from_inventory_km': 50
        }, inplace=True)
        
        suppliers['emissions_kg'] = suppliers['distance_from_inventory_km'] * suppliers['co2_per_km']
        suppliers['shelf_life_days'] = np.maximum(1, 20 - (suppliers['distance_from_inventory_km'] // 5))
        suppliers['shelf_life_days'] = suppliers.apply(
            lambda row: 90 if row['commodity'].lower() in HIGH_SHELF_COMMODITIES else row['shelf_life_days'],
            axis=1
        )
        suppliers['local_score'] = suppliers['price_per_unit'] + suppliers['emissions_kg']
        return suppliers
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.stop()

suppliers = preprocess_data()

# --- Dashboard Header ---
st.title("🌿 Walmart FreshRoute AI Dashboard")
st.markdown("---")

# --- Sales Forecasting Section ---
st.header("📈 Sales Forecasting")
with st.expander("Predict Product Demand"):
    try:
        product_options = sorted(train_df['Product_Name'].dropna().unique())
        selected_product = st.selectbox("Select Product for Forecast", product_options)
        
        if st.button("Generate Sales Forecast"):
            if not selected_product or selected_product not in train_df['Product_Name'].values:
                st.error("Please select a valid product")
            else:
                df_filtered = train_df[train_df["Product_Name"].str.lower() == selected_product.lower()].copy()
                df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")
                df_filtered.dropna(subset=["Date"], inplace=True)
                
                if df_filtered.empty:
                    st.error("No valid dates found for selected product")
                else:
                    df_resampled = df_filtered.set_index("Date")["Quantity_Sold"].resample("W-MON").sum()
                    df_p = df_resampled.reset_index().rename(columns={"Date": "ds", "Quantity_Sold": "y"})
                    
                    if len(df_p) < 5:
                        st.warning("Insufficient data points for accurate forecasting (minimum 5 weeks required)")
                    else:
                        model = Prophet(
                            weekly_seasonality=True,
                            yearly_seasonality=True,
                            uncertainty_samples=50
                        )
                        model.fit(df_p)
                        
                        future = model.make_future_dataframe(periods=1, freq="W-MON", include_history=False)
                        forecast = model.predict(future)
                        
                        if forecast.empty:
                            st.error("Failed to generate forecast - empty result returned")
                        else:
                            forecast_ds = forecast['ds'].iloc[0].strftime('%Y-%m-%d')
                            forecasted_qty = int(forecast['yhat'].iloc[0])
                            confidence_lower = int(forecast['yhat_lower'].iloc[0])
                            confidence_upper = int(forecast['yhat_upper'].iloc[0])
                            
                            st.success("📊 Forecast Generated Successfully")
                            st.markdown(
                                f"""
                                <div class='report-box'>
                                    <h4 style='text-align:center'>{selected_product} Forecast</h4>
                                    <p><strong>Week of:</strong> {forecast_ds}</p>
                                    <p><strong>Expected Sales:</strong> {forecasted_qty} units</p>
                                    <p><strong>Confidence Interval:</strong> {confidence_lower} - {confidence_upper} units</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")

# --- AI Sourcing Decision ---
st.header("🧠 AI Sourcing Recommendation")
with st.expander("Get Sourcing Decision"):
    try:
        commodity = st.selectbox(
            "Select Commodity",
            sorted(suppliers['commodity'].dropna().str.title().unique())
        )
        qty_needed = st.number_input(
            "Quantity Needed (kg)",
            min_value=1,
            max_value=500,
            value=10
        )
        
        if st.button("Generate Sourcing Plan"):
            if not commodity:
                st.error("Please select a commodity")
            else:
                matched = suppliers[suppliers['commodity'].str.lower() == commodity.lower()]
                
                if matched.empty:
                    st.error("No suppliers found for selected commodity")
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
                    
                    # Generate comparison with central warehouse
                    central_distance_km = 150
                    central_vehicle = assign_vehicle(qty_needed)
                    central_emissions = round(central_distance_km * VEHICLE_EMISSIONS.get(central_vehicle, CO2_PER_KM_DEFAULT), 2)
                    central_price = round(best['price_per_unit'] * np.random.uniform(1.8, 2.4), 2)
                    
                    # AI decision logic
                    ai_input = pd.DataFrame([{
                        'modal_price': best['price_per_unit'],
                        'distance_km': dist,
                        'transport_cost': transport_cost,
                        'local_price': best['price_per_unit'],
                        'central_price': central_price
                    }])
                    
                    decision_text = ""
                    confidence = 0
                    
                    try:
                        # Train model if not already trained
                        if 'model' not in st.session_state:
                            np.random.seed(42)
                            demand['distance_km'] = np.random.randint(10, 150, size=len(demand))
                            demand['transport_cost'] = demand['distance_km'] * 4
                            demand['central_price'] = demand['modal_price'] + demand['transport_cost']
                            demand = demand.merge(suppliers[['commodity', 'price_per_unit']], on='commodity', how='left')
                            demand.rename(columns={'price_per_unit': 'local_price'}, inplace=True)
                            demand['decision'] = np.where(
                                (demand['local_price'] < demand['central_price']) & 
                                (demand['transport_cost'] < 400), 1, 0
                            )
                            st.session_state.model = RandomForestClassifier(n_estimators=150, random_state=42)
                            st.session_state.model.fit(
                                demand[['modal_price','distance_km','transport_cost','local_price','central_price']],
                                demand['decision']
                            )
                        
                        # Get prediction
                        pred = st.session_state.model.predict(ai_input)[0]
                        conf = st.session_state.model.predict_proba(ai_input)[0][pred]
                        confidence = round(conf * 100, 2)
                        
                        if pred == 0 and best['price_per_unit'] < central_price and emissions_local < central_emissions:
                            decision_text = "✅ Source Locally (Overridden by Sustainability Check)"
                        else:
                            decision_text = "✅ Source Locally" if pred == 1 else "🚛 Use Central Warehouse"
                            
                    except Exception as e:
                        st.error(f"AI Decision Engine Error: {str(e)}")
                        decision_text = "⚠️ Manual Review Required"
                        
                    # Display results
                    supplier_name = best['supplier_name']
                    supply_area = best.get('supply_region', 'Unknown Region')
                    route = f"{supplier_name} → {supply_area} → Shanivar Peth"
                    eta = round(dist / 30, 2)
                    
                    st.markdown(
                        f"""
                        <div class='report-box'>
                            <h4 style='text-align:center'>Sourcing Recommendation</h4>
                            <p><strong>Commodity:</strong> {commodity.title()}</p>
                            <p><strong>Supplier:</strong> {supplier_name} (ID: {best['supplier_id']})</p>
                            <p><strong>Available Stock:</strong> {int(best['available_quantity_kg'])} kg</p>
                            
                            <h5>Financial Metrics</h5>
                            <p><strong>Local Price:</strong> ₹{best['price_per_unit']}/kg</p>
                            <p><strong>Transport Cost:</strong> ₹{transport_cost}</p>
                            <p><strong>Total Cost:</strong> ₹{final_cost}</p>
                            <p><strong>Spoilage Estimate:</strong> {spoilage_kg} kg ({spoilage_pct}%)</p>
                            
                            <h5>Environmental Impact</h5>
                            <p><strong>CO₂ Emissions (Local):</strong> {emissions_local} kg</p>
                            <p><strong>CO₂ Emissions (Central):</strong> {central_emissions} kg</p>
                            <p><strong>Estimated Shelf Life:</strong> {shelf_life} days</p>
                            
                            <h5>Operational Details</h5>
                            <p><strong>Vehicle Recommended:</strong> {vehicle}</p>
                            <p><strong>Route:</strong> {route}</p>
                            <p><strong>ETA:</strong> {eta} hours</p>
                            
                            <h5>AI Decision</h5>
                            <p><strong>Recommendation:</strong> {decision_text}</p>
                            <p><strong>Confidence:</strong> {confidence}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    except Exception as e:
        st.error(f"Sourcing recommendation error: {str(e)}")

# --- System Status ---
st.sidebar.header("System Status")
st.sidebar.write(f"Supplier Records: {len(suppliers)}")
st.sidebar.write(f"Products Tracked: {len(train_df['Product_Name'].unique())}")
st.sidebar.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
