import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.linear_model import LinearRegression
import joblib
import os

# Data preparation
# Load or generate data
if os.path.exists('picker_data.csv'):
    data = pd.read_csv('picker_data.csv')
else:
    # Generate synthetic data
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

    # Save the dataset
    data.to_csv('picker_data.csv', index=False)

data['Date'] = pd.to_datetime(data['Date'])

# Machine learning model
# Load or train model
if os.path.exists('reg_model.pkl'):
    # Load the model
    reg_model = joblib.load('reg_model.pkl')
else:
    # Prepare features and target
    X = data[['PickerSpeed']]
    y = data['BalesPerHectare']

    # Train the model
    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Save the model
    joblib.dump(reg_model, 'reg_model.pkl')

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
                marks={
                    i: f'{i}' for i in np.arange(
                        round(data['PickerSpeed'].min(), 1),
                        round(data['PickerSpeed'].max(), 1) + 0.2, 0.2
                    )
                },
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
                marks={
                    i: f'{i}' for i in np.arange(
                        round(data['BalesPerHectare'].min(), 1),
                        round(data['BalesPerHectare'].max(), 1) + 0.5, 0.5
                    )
                },
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
    global data  # Declare 'data' as global at the beginning

    # Simulate real-time data update
    new_data_point = {
        'Date': data['Date'].max() + pd.Timedelta(hours=1),
        'PickerSpeed': np.random.uniform(4.0, 6.0),
        'BalesPerHectare': np.random.uniform(2.0, 4.0),
    }
    data = data.append(new_data_point, ignore_index=True)

    # Filter data based on slider inputs
    filtered_data = data[
        (data['PickerSpeed'] >= speed_range[0]) &
        (data['PickerSpeed'] <= speed_range[1]) &
        (data['BalesPerHectare'] >= bales_range[0]) &
        (data['BalesPerHectare'] <= bales_range[1])
    ]

    fig = px.scatter(
        filtered_data,
        x='BalesPerHectare',
        y='PickerSpeed',
        labels={
            'BalesPerHectare': 'Bales per Hectare',
            'PickerSpeed': 'Picker Speed (km/h)'
        },
        title='Picker Speed vs. Bales per Hectare',
        hover_data=['Date'],
    )
    fig.update_xaxes(dtick=0.5)
    fig.update_yaxes(dtick=0.2)
    return fig

# Callback to update histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_histogram(n_intervals):
    fig = px.histogram(data, x='PickerSpeed', nbins=20)
    fig.update_layout(title='Distribution of Picker Speeds')
    return fig

# Callback to update heatmap
@app.callback(
    Output('heatmap-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_heatmap(n_intervals):
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
    return html.Div([
        html.H5(f"Predicted Bales per Hectare: {predicted_yield:.2f}")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
