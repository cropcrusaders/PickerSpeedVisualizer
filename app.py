import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
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
    X = data[['PickerSpeed', 'MaxWrapEjectionSpeed']]
    y = data['BalesPerHectare']

    # Train the model
    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Save the model
    joblib.dump(reg_model, 'reg_model.pkl')
    return reg_model

reg_model = train_model()

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("PickerSpeedVisualizer"),
    dbc.Row([
        dbc.Col([
            html.Label('Picker Speed Range (km/h)'),
            dcc.RangeSlider(
                id='speed-range',
                min=round(data['PickerSpeed'].min(), 1),
                max=round(data['PickerSpeed'].max(), 1),
                step=0.1,
                value=[
                    round(data['PickerSpeed'].min(), 1),
                    round(data['PickerSpeed'].max(), 1)
                ],
                marks={i: f'{i}' for i in np.arange(
                    round(data['PickerSpeed'].min(), 1),
                    round(data['PickerSpeed'].max(), 1) + 0.5, 0.5
                )},
            ),
            html.Br(),
            html.Label('Bales per Hectare Range'),
            dcc.RangeSlider(
                id='bales-range',
                min=round(data['BalesPerHectare'].min(), 1),
                max=round(data['BalesPerHectare'].max(), 1),
                step=0.1,
                value=[
                    round(data['BalesPerHectare'].min(), 1),
                    round(data['BalesPerHectare'].max(), 1)
                ],
                marks={i: f'{i}' for i in np.arange(
                    round(data['BalesPerHectare'].min(), 1),
                    round(data['BalesPerHectare'].max(), 1) + 1, 1
                )},
            ),
            html.Br(),
            html.H4("Predict Bale Yield"),
            html.Label("Enter Picker Speed (km/h):"),
            dcc.Input(
                id='input-speed',
                type='number',
                value=data['PickerSpeed'].mean(),
                min=data['PickerSpeed'].min(),
                max=data['PickerSpeed'].max(),
                step=0.1,
            ),
            html.Br(),
            html.Label("Enter Max Wrap and Ejection Speed (km/h):"),
            dcc.Input(
                id='input-wrap-ejection-speed',
                type='number',
                value=data['MaxWrapEjectionSpeed'].mean(),
                min=data['MaxWrapEjectionSpeed'].min(),
                max=data['MaxWrapEjectionSpeed'].max(),
                step=0.1,
            ),
            html.Div(id='prediction-output'),
            html.Br(),
            html.H4("Add New Data"),
            dcc.Input(id='input-picker-speed', type='number', placeholder='Picker Speed (km/h)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-max-wrap-ejection-speed', type='number', placeholder='Max Wrap and Ejection Speed (km/h)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-bales-per-hectare', type='number', placeholder='Yield (Bales per Hectare)', min=0, step=0.1),
            html.Br(),
            html.Button('Submit Data', id='submit-data-btn', n_clicks=0),
            html.Div(id='data-submit-output'),
        ], width=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Scatter Plot', tab_id='scatter-tab'),
                dbc.Tab(label='Histogram', tab_id='histogram-tab'),
            ], id='tabs', active_tab='scatter-tab'),
            html.Div(id='tab-content'),
        ], width=9),
    ]),
], fluid=True)

# Callbacks

# Callback to render content based on active tab
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    if active_tab == 'scatter-tab':
        return dcc.Graph(id='scatter-plot')
    elif active_tab == 'histogram-tab':
        return dcc.Graph(id='histogram-plot')
    else:
        return html.Div("No tab selected")

# Callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('speed-range', 'value'),
        Input('bales-range', 'value'),
    ]
)
def update_scatter(speed_range, bales_range):
    filtered_data = data[
        (data['PickerSpeed'] >= speed_range[0]) &
        (data['PickerSpeed'] <= speed_range[1]) &
        (data['BalesPerHectare'] >= bales_range[0]) &
        (data['BalesPerHectare'] <= bales_range[1])
    ]

    # Create the figure
    fig = go.Figure()

    # Plot Picker Speed
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['PickerSpeed'],
        mode='markers',
        name='Picker Speed',
        marker=dict(color='blue'),
    ))

    # Plot Max Wrap and Ejection Speed
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['MaxWrapEjectionSpeed'],
        mode='markers',
        name='Max Wrap and Ejection Speed',
        marker=dict(color='red'),
    ))

    # Compute trend lines for each variable
    trend_variables = {
        'Picker Speed': ('PickerSpeed', 'blue'),
        'Max Wrap and Ejection Speed': ('MaxWrapEjectionSpeed', 'red'),
    }

    for name, (variable, color) in trend_variables.items():
        model = LinearRegression()
        X = filtered_data[['BalesPerHectare']]
        y = filtered_data[variable]
        model.fit(X, y)
        trend_x = np.linspace(X.min(), X.max(), 100)
        trend_y = model.predict(trend_x)
        fig.add_trace(go.Scatter(
            x=trend_x.flatten(),
            y=trend_y,
            mode='lines',
            name=f'{name} Trend Line',
            line=dict(color=color, dash='dash')
        ))

    fig.update_layout(
        title='Speeds vs. Yield (Bales per Hectare)',
        xaxis_title='Yield (Bales per Hectare)',
        yaxis_title='Speed (km/h)',
        hovermode='closest',
        legend=dict(x=0.01, y=0.99)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=0.5)
    return fig

# Callback to handle form data submission
@app.callback(
    Output('data-submit-output', 'children'),
    Input('submit-data-btn', 'n_clicks'),
    State('input-picker-speed', 'value'),
    State('input-max-wrap-ejection-speed', 'value'),
    State('input-bales-per-hectare', 'value')
)
def add_data_to_csv(n_clicks, picker_speed, max_wrap_ejection_speed, bales_per_hectare):
    if n_clicks > 0:
        try:
            if picker_speed is None or bales_per_hectare is None or max_wrap_ejection_speed is None:
                return 'Please provide Picker Speed, Max Wrap and Ejection Speed, and Yield.'

            # Create a new row of data
            new_data = pd.DataFrame({
                'BalesPerHectare': [bales_per_hectare],
                'MaxWrapEjectionSpeed': [max_wrap_ejection_speed],
                'PickerSpeed': [picker_speed],
            })

            # Append the new data to the CSV file
            new_data.to_csv('user_data.csv', mode='a', header=False, index=False)

            # Update the global data variable
            global data
            data = pd.concat([data, new_data], ignore_index=True)

            # Retrain the model with the new data
            train_model()

            return "Data submitted successfully."
        except Exception as e:
            return f"Error: {str(e)}"
    return ''

# Callback to update histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    []
)
def update_histogram():
    fig = go.Figure()
    # Plot histograms for available speed variables
    if 'PickerSpeed' in data.columns and data['PickerSpeed'].notnull().any():
        fig.add_trace(go.Histogram(
            x=data['PickerSpeed'],
            nbinsx=10,
            name='Picker Speed',
            marker_color='blue',
            opacity=0.75
        ))
    if 'MaxWrapEjectionSpeed' in data.columns and data['MaxWrapEjectionSpeed'].notnull().any():
        fig.add_trace(go.Histogram(
            x=data['MaxWrapEjectionSpeed'],
            nbinsx=10,
            name='Max Wrap and Ejection Speed',
            marker_color='red',
            opacity=0.75
        ))
    fig.update_layout(
        barmode='overlay',
        title='Distribution of Speeds',
        xaxis_title='Speed (km/h)',
        yaxis_title='Count',
        legend=dict(x=0.7, y=0.95)
    )
    return fig

# Callback for prediction output
@app.callback(
    Output('prediction-output', 'children'),
    [
        Input('input-speed', 'value'),
        Input('input-wrap-ejection-speed', 'value')
    ]
)
def predict_bale_yield(speed_value, wrap_ejection_speed):
    predicted_yield = reg_model.predict([[speed_value, wrap_ejection_speed]])[0]
    return html.Div([html.H5(f"Predicted Yield (Bales per Hectare): {predicted_yield:.2f}")])

if __name__ == '__main__':
    app.run_server(debug=True)
