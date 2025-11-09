import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Load model and expected columns ---
model = joblib.load('C:/Users/MSI/Desktop/renewable-energy-cost-predictor/models/pv_model.pkl')
model_columns = joblib.load('C:/Users/MSI/Desktop/renewable-energy-cost-predictor/models/model_columns.pkl')

st.title("PV Solar Production Dashboard")

# --- Sidebar: Energy Parameters ---
st.sidebar.header("Energy Parameters")
cooling = st.sidebar.number_input("Total Cooling (kW)", min_value=0, max_value=5000, value=0)
heating = st.sidebar.number_input("Total Heating (kW)", min_value=0, max_value=5000, value=0)
mechanical = st.sidebar.number_input("Total Mechanical (kW)", min_value=0, max_value=5000, value=0)
lighting = st.sidebar.number_input("Total Lighting (kW)", min_value=0, max_value=5000, value=0)
plug_loads = st.sidebar.number_input("Total Plug Loads (kW)", min_value=0, max_value=5000, value=0)
data_center = st.sidebar.number_input("Total Data Center (kW)", min_value=0, max_value=5000, value=0)
total_building = st.sidebar.number_input("Total Building (kW)", min_value=0, max_value=10000, value=0)
building_net = st.sidebar.number_input("Building Net (kW)", min_value=0, max_value=10000, value=0)

# --- Sidebar: Temporal Parameters ---
st.sidebar.header("Temporal Parameters")
selected_date = st.sidebar.date_input("Select Date")
selected_time = st.sidebar.time_input("Select Time")
dt = datetime.combine(selected_date, selected_time)
year = dt.year
month = dt.month
hour = dt.hour
weekday = dt.weekday()  # 0=Monday, 6=Sunday

# --- Prepare input DataFrame ---
X_input = pd.DataFrame([[0]*len(model_columns)], columns=model_columns)

# Fill energy values
X_input['Total Cooling (kW)'] = cooling
X_input['Total Heating (kW)'] = heating
X_input['Total Mechanical (kW)'] = mechanical
X_input['Total Lighting (kW)'] = lighting
X_input['Total Plug Loads (kW)'] = plug_loads
X_input['Total Data Center (kW)'] = data_center
X_input['Total Building (kW)'] = total_building
X_input['Building Net (kW)'] = building_net

# Fill temporal features
if 'year' in X_input.columns:
    X_input['year'] = year
if 'month' in X_input.columns:
    X_input['month'] = month
if 'hour' in X_input.columns:
    X_input['hour'] = hour
if 'weekday' in X_input.columns:
    X_input['weekday'] = weekday

# Fill one-hot day
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_name = days[weekday]
one_hot_col = f'Day of Week_{day_name}'
if one_hot_col in X_input.columns:
    X_input[one_hot_col] = 1

# Ensure 'Unnamed: 11' exists if required
if 'Unnamed: 11' in X_input.columns:
    X_input['Unnamed: 11'] = 0

# --- Make Prediction ---
prediction = model.predict(X_input)
predicted_pv = max(0, prediction[0])  # Ensure no negative PV values
st.subheader(f"Predicted PV Production: {predicted_pv:.2f} kW")

# --- Example Visualization ---
example_data = pd.DataFrame({
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'PV (kW)': [10, 15, 8, 12, 20]
})
st.line_chart(example_data.set_index('Day'))
