import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Data preparation
def load_data():
    # Load the CSV file provided by the user
    if os.path.exists('user_data.csv'):
        try:
            data = pd.read_csv('user_data.csv')
            if data.empty or data.shape[1] == 0:
                raise ValueError("Empty or corrupt CSV file")
        except Exception as e:
            print(f"Error loading data: {e}")
            data = pd.DataFrame()
    else:
        print("CSV file 'user_data.csv' not found.")
        data = pd.DataFrame()
    return data

data = load_data()

# Map the columns to internal variable names
data = data.rename(columns={
    'Yield (Bales Per Hectare)': 'BalesPerHectare',
    'Max Wrap and Ejection Process (km/h)': 'MaxWrapEjectionSpeed',
    'Max for Row Units (km/h)': 'PickerSpeed'
})

# Ensure correct data types
data['BalesPerHectare'] = pd.to_numeric(data['BalesPerHectare'], errors='coerce')
data['MaxWrapEjectionSpeed'] = pd.to_numeric(data['MaxWrapEjectionSpeed'], errors='coerce')
data['PickerSpeed'] = pd.to_numeric(data['PickerSpeed'], errors='coerce')

# Remove any rows with missing values
data = data.dropna(subset=['BalesPerHectare', 'MaxWrapEjectionSpeed', 'PickerSpeed'])

# Machine learning model
def train_model():
    # Prepare features and target
    X = data[['BalesPerHectare']]
    y = data[['PickerSpeed', 'MaxWrapEjectionSpeed']]

    # Train the model
    reg_model = MultiOutputRegressor(LinearRegression())
    reg_model.fit(X, y)

    # Save the model
    joblib.dump(reg_model, 'reg_model.pkl')
    return reg_model

reg_model = train_model()

# Calculate minimum and maximum speeds and yields
min_speed = round(data['PickerSpeed'].min(), 1)
max_speed = round(data['PickerSpeed'].max(), 1)

min_bales = round(data['BalesPerHectare'].min(), 1)
max_bales = round(data['BalesPerHectare'].max(), 1)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("PickerSpeedVisualizer"),
    dbc.Row([
        dbc.Col([
            html.Label('Yield (Bales per Hectare) Range'),
            dcc.RangeSlider(
                id='bales-range',
                min=min_bales,
                max=max_bales,
                step=0.1,
                value=[min_bales, max_bales],
                marks={i: f'{i}' for i in np.arange(min_bales, max_bales + 1, 1)},
            ),
            html.Br(),
            html.Label('Picker Speed Range (km/h)'),
            dcc.RangeSlider(
                id='speed-range',
                min=min_speed,
                max=max_speed,
                step=0.1,
                value=[min_speed, max_speed],
                marks={i: f'{i:.1f}' for i in np.arange(min_speed, max_speed + 0.5, 0.5)},
            ),
            html.Br(),
            html.H4("Predict Speeds for Desired Yield"),
            html.Label("Enter Desired Yield (Bales per Hectare):"),
            dcc.Input(
                id='input-yield',
                type='number',
                value=data['BalesPerHectare'].mean(),
                min=data['BalesPerHectare'].min(),
                max=data['BalesPerHectare'].max(),
                step=0.1,
            ),
            html.Div(id='prediction-output'),
            html.Br(),
            html.H4("Add New Data"),
            dcc.Input(id='input-bales-per-hectare', type='number', placeholder='Yield (Bales per Hectare)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-picker-speed', type='number', placeholder='Picker Speed (km/h)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-max-wrap-ejection-speed', type='number', placeholder='Max Wrap and Ejection Speed (km/h)', min=0, step=0.1),
            html.Br(),
            html.Button('Submit Data', id='submit-data-btn', n_clicks=0),
            html.Div(id='data-submit-output'),
        ], width=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Scatter Plot', tab_id='scatter-tab'),
                dbc.Tab(label='Histogram', tab_id='histogram-tab'),
      
