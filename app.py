import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go  # Import graph_objects
from sklearn.linear_model import LinearRegression
import joblib
import base64
import io

# Data preparation
# Load or generate data
def load_or_generate_data():
    if os.path.exists('picker_data.csv'):
        try:
            data = pd.read_csv('picker_data.csv')
            if data.empty or data.shape[1] == 0:
                raise ValueError("Empty or corrupt CSV file")
            if 'MaxBaleEjectionSpeed' not in data.columns:
                print("Adding MaxBaleEjectionSpeed to existing data.")
                data['MaxBaleEjectionSpeed'] = data['PickerSpeed'] * np.random.uniform(1.1, 1.3, size=len(data))
                data.to_csv('picker_data.csv', index=False)
        except (pd.errors.EmptyDataError, ValueError):
            print("The existing 'picker_data.csv' is empty or corrupt. Generating new data.")
            data = generate_synthetic_data()
            data.to_csv('picker_data.csv', index=False)
    else:
        data = generate_synthetic_data()
        data.to_csv('picker_data.csv', index=False)
    return data

def generate_synthetic_data():
    np.random.seed(42)
    num_samples = 500
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=num_samples, freq='H'),
        'PickerSpeed': np.random.uniform(4.0, 6.0, size=num_samples),
        'BalesPerHectare': np.random.uniform(2.0, 4.0, size=num_samples),
    })
    # Simulate a relationship between PickerSpeed and BalesPerHectare
    data['BalesPerHectare'] += (
        (data['PickerSpeed'] - 5) * 0.5 + np.random.normal(0, 0.1, num_samples)
    )
    # Add MaxBaleEjectionSpeed
    data['MaxBaleEjectionSpeed'] = data['PickerSpeed'] * np.random.uniform(1.1, 1.3, size=num_samples)
    return data

data = load_or_generate_data()
data['Date'] = pd.to_datetime(data['Date'])

# Machine learning model
# Load or train model
def load_or_train_model():
    if os.path.exists('reg_model.pkl'):
        try:
            reg_model = joblib.load('reg_model.pkl')
        except (ModuleNotFoundError, EOFError, ImportError):
            print("Model file is corrupted or incompatible. Retraining the model...")
            reg_model = train_and_save_model()
    else:
        reg_model = train_and_save_model()
    return reg_model

def train_and_save_model():
    # Prepare features and target
    X = data[['PickerSpeed']]
    y = data['BalesPerHectare']

    # Train the model
    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Save the model
    joblib.dump(reg_model, 'reg_model.pkl')
    return reg_model

reg_model = load_or_train_model()

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
                    round(data['PickerSpeed'].max(), 1) + 0.2, 0.2
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
                    round(data['BalesPerHectare'].max(), 1) + 0.5, 0.5
                )},
            ),
            html.Br(),
            html.H4("Predict Bale Yield"),
            html.Label("Enter Picker Speed (km/h):"),
            dcc.Input(
                id='input-speed',
                type='number',
                value=5.0,
                min=4.0,
                max=6.0,
                step=0.1,
            ),
            html.Div(id='prediction-output'),
            html.Br(),
            html.H4("Add New Data"),
            dcc.Input(id='input-picker-speed', type='number', placeholder='Picker Speed (km/h)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-bales-per-hectare', type='number', placeholder='Bales per Hectare', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-date', type='text', placeholder='Date (YYYY-MM-DD HH:MM:SS)'),
            html.Br(),
            html.Button('Submit Data', id='submit-data-btn', n_clicks=0),
            html.Div(id='data-submit-output'),
        ], width=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Scatter Plot', tab_id='scatter-tab'),
                dbc.Tab(label='Histogram', tab_id='histogram-tab'),
                dbc.Tab(label='Heatmap', tab_id='heatmap-tab'),
            ], id='tabs', active_tab='scatter-tab'),
            html.Div(id='tab-content'),
        ], width=9),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # Update every 5 seconds
        n_intervals=0
    ),
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
    elif active_tab == 'heatmap-tab':
        return dcc.Graph(id='heatmap-plot')
    else:
        return html.Div("No tab selected")

# Callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('speed-range', 'value'),
        Input('bales-range', 'value'),
        Input('interval-component', 'n_intervals')
    ]
)
def update_scatter(speed_range, bales_range, n_intervals):
    global data
    data = pd.read_csv('picker_data.csv')  # Reload data to ensure new data is reflected
    data['Date'] = pd.to_datetime(data['Date'])

    # Append a new data point
    new_data_point = {
        'Date': data['Date'].max() + pd.Timedelta(hours=1),
        'PickerSpeed': np.random.uniform(4.0, 6.0),
        'BalesPerHectare': np.random.uniform(2.0, 4.0),
    }
    new_data_point['BalesPerHectare'] += (new_data_point['PickerSpeed'] - 5) * 0.5 + np.random.normal(0, 0.1)
    new_data_point['MaxBaleEjectionSpeed'] = new_data_point['PickerSpeed'] * np.random.uniform(1.1, 1.3)
    data = pd.concat([data, pd.DataFrame([new_data_point])], ignore_index=True)

    # Save the updated data
    data.to_csv('picker_data.csv', index=False)

    filtered_data = data[
        (data['PickerSpeed'] >= speed_range[0]) &
        (data['PickerSpeed'] <= speed_range[1]) &
        (data['BalesPerHectare'] >= bales_range[0]) &
        (data['BalesPerHectare'] <= bales_range[1])
    ]

    # Create the figure with two traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['PickerSpeed'],
        mode='markers',
        name='Picker Speed',
        marker=dict(color='blue'),
        hovertext=filtered_data['Date'].astype(str)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['MaxBaleEjectionSpeed'],
        mode='markers',
        name='Max Bale Ejection Speed',
        marker=dict(color='red'),
        hovertext=filtered_data['Date'].astype(str)
    ))

    # Add average lines
    avg_picker_speed = filtered_data['PickerSpeed'].mean()
    avg_max_bale_ejection_speed = filtered_data['MaxBaleEjectionSpeed'].mean()
    avg_bales_per_hectare = filtered_data['BalesPerHectare'].mean()

    fig.add_hline(y=avg_picker_speed, line_dash="dash", line_color='blue',
                  annotation_text='Avg Picker Speed', annotation_position="bottom right")
    fig.add_hline(y=avg_max_bale_ejection_speed, line_dash="dash", line_color='red',
                  annotation_text='Avg Max Bale Ejection Speed', annotation_position="top right")
    fig.add_vline(x=avg_bales_per_hectare, line_dash="dash", line_color='green',
                  annotation_text='Avg Bales per Hectare', annotation_position="top left")

    fig.update_layout(
        title='Picker Speed vs. Bales per Hectare with Max Bale Ejection Speed',
        xaxis_title='Bales per Hectare',
        yaxis_title='Speed (km/h)',
        hovermode='closest',
        legend=dict(x=0.01, y=0.99)
    )
    fig.update_xaxes(dtick=0.5)
    fig.update_yaxes(dtick=0.2)
    return fig

# Callback to handle form data submission
@app.callback(
    Output('data-submit-output', 'children'),
    Input('submit-data-btn', 'n_clicks'),
    State('input-picker-speed', 'value'),
    State('input-bales-per-hectare', 'value'),
    State('input-date', 'value')
)
def add_data_to_csv(n_clicks, picker_speed, bales_per_hectare, date):
    if n_clicks > 0:
        try:
            if picker_speed is None or bales_per_hectare is None or not date:
                return 'Please provide all inputs (Picker Speed, Bales per Hectare, and Date).'

            # Convert the input date to datetime
            parsed_date = pd.to_datetime(date)

            # Calculate MaxBaleEjectionSpeed
            max_bale_ejection_speed = picker_speed * np.random.uniform(1.1, 1.3)

            # Create a new row of data
            new_data = pd.DataFrame({
                'Date': [parsed_date],
                'PickerSpeed': [picker_speed],
                'BalesPerHectare': [bales_per_hectare],
                'MaxBaleEjectionSpeed': [max_bale_ejection_speed]
            })

            # Append the new data to the CSV file
            new_data.to_csv('picker_data.csv', mode='a', header=False, index=False)

            return f"Data submitted successfully: {picker_speed} km/h, {bales_per_hectare} bales/ha on {date}"
        except Exception as e:
            return f"Error: {str(e)}"
    return ''

# Callback to update histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_histogram(n_intervals):
    data = pd.read_csv('picker_data.csv')
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data['PickerSpeed'],
        nbinsx=20,
        name='Picker Speed',
        marker_color='blue',
        opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=data['MaxBaleEjectionSpeed'],
        nbinsx=20,
        name='Max Bale Ejection Speed',
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

# Callback to update heatmap
@app.callback(
    Output('heatmap-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_heatmap(n_intervals):
    data = pd.read_csv('picker_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    heatmap_data = data.pivot_table(
        index=data['Date'].dt.date,
        columns=data['Date'].dt.hour,
        values='BalesPerHectare',
        aggfunc='mean'
    )
    fig = px.imshow(
        heatmap_data,
        labels={'x': "Hour", 'y': "Date", 'color': "Bales per Hectare"}
    )
    fig.update_layout(title='Heatmap of Bale Yields Over Time')
    return fig

# Callback for prediction output
@app.callback(
    Output('prediction-output', 'children'),
    [Input('input-speed', 'value')]
)
def predict_bale_yield(speed_value):
    predicted_yield = reg_model.predict([[speed_value]])[0]
    return html.Div([html.H5(f"Predicted Bales per Hectare: {predicted_yield:.2f}")])

if __name__ == '__main__':
    app.run_server(debug=True)
