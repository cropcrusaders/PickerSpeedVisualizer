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
# Load or generate data
def load_or_generate_data():
    if os.path.exists('picker_data.csv'):
        try:
            data = pd.read_csv('picker_data.csv')
            if data.empty or data.shape[1] == 0:
                raise ValueError("Empty or corrupt CSV file")
            # Ensure all required columns are present
            required_columns = ['Date', 'PickerSpeed', 'BalesPerHectare', 'MaxBaleEjectionSpeed', 'MaxWrapSpeed']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = np.nan
            data.to_csv('picker_data.csv', index=False)
        except (pd.errors.EmptyDataError, ValueError):
            print("The existing 'picker_data.csv' is empty or corrupt. Generating new data.")
            data = pd.DataFrame(columns=['Date', 'PickerSpeed', 'BalesPerHectare', 'MaxBaleEjectionSpeed', 'MaxWrapSpeed'])
            data.to_csv('picker_data.csv', index=False)
    else:
        data = pd.DataFrame(columns=['Date', 'PickerSpeed', 'BalesPerHectare', 'MaxBaleEjectionSpeed', 'MaxWrapSpeed'])
        data.to_csv('picker_data.csv', index=False)
    return data

data = load_or_generate_data()
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Machine learning model
# Load or train model
def load_or_train_model():
    if os.path.exists('reg_model.pkl'):
        try:
            reg_model = joblib.load('reg_model.pkl')
        except (ModuleNotFoundError, EOFError, ImportError):
            print("Model file is corrupted or incompatible.")
            reg_model = None
    else:
        reg_model = None
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
                min=0,
                max=10,
                step=0.1,
                value=[0, 10],
                marks={i: f'{i}' for i in range(0, 11)},
            ),
            html.Br(),
            html.Label('Bales per Hectare Range'),
            dcc.RangeSlider(
                id='bales-range',
                min=0,
                max=10,
                step=0.1,
                value=[0, 10],
                marks={i: f'{i}' for i in range(0, 11)},
            ),
            html.Br(),
            html.H4("Predict Bale Yield"),
            html.Label("Enter Picker Speed (km/h):"),
            dcc.Input(
                id='input-speed',
                type='number',
                value=5.0,
                min=0,
                max=10,
                step=0.1,
            ),
            html.Div(id='prediction-output'),
            html.Br(),
            html.H4("Add New Data"),
            dcc.Input(id='input-picker-speed', type='number', placeholder='Picker Speed (km/h)', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-bales-per-hectare', type='number', placeholder='Bales per Hectare', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-max-bale-ejection-speed', type='number', placeholder='Max Bale Ejection Speed', min=0, step=0.1),
            html.Br(),
            dcc.Input(id='input-max-wrap-speed', type='number', placeholder='Max Wrap Speed', min=0, step=0.1),
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
    # Removed Interval component as we are no longer auto-updating
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
    ]
)
def update_scatter(speed_range, bales_range):
    data = pd.read_csv('picker_data.csv')  # Reload data to ensure new data is reflected
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    filtered_data = data.dropna(subset=['PickerSpeed', 'BalesPerHectare'])
    filtered_data = filtered_data[
        (filtered_data['PickerSpeed'] >= speed_range[0]) &
        (filtered_data['PickerSpeed'] <= speed_range[1]) &
        (filtered_data['BalesPerHectare'] >= bales_range[0]) &
        (filtered_data['BalesPerHectare'] <= bales_range[1])
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
        hovertext=filtered_data['Date'].astype(str)
    ))

    # Plot Max Bale Ejection Speed if available
    if 'MaxBaleEjectionSpeed' in filtered_data.columns and filtered_data['MaxBaleEjectionSpeed'].notnull().any():
        fig.add_trace(go.Scatter(
            x=filtered_data['BalesPerHectare'],
            y=filtered_data['MaxBaleEjectionSpeed'],
            mode='markers',
            name='Max Bale Ejection Speed',
            marker=dict(color='red'),
            hovertext=filtered_data['Date'].astype(str)
        ))

    # Plot Max Wrap Speed if available
    if 'MaxWrapSpeed' in filtered_data.columns and filtered_data['MaxWrapSpeed'].notnull().any():
        fig.add_trace(go.Scatter(
            x=filtered_data['BalesPerHectare'],
            y=filtered_data['MaxWrapSpeed'],
            mode='markers',
            name='Max Wrap Speed',
            marker=dict(color='green'),
            hovertext=filtered_data['Date'].astype(str)
        ))

    fig.update_layout(
        title='Speeds vs. Bales per Hectare',
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
    State('input-max-bale-ejection-speed', 'value'),
    State('input-max-wrap-speed', 'value'),
    State('input-date', 'value')
)
def add_data_to_csv(n_clicks, picker_speed, bales_per_hectare, max_bale_ejection_speed, max_wrap_speed, date):
    if n_clicks > 0:
        try:
            if picker_speed is None or bales_per_hectare is None or not date:
                return 'Please provide at least Picker Speed, Bales per Hectare, and Date.'

            # Convert the input date to datetime
            parsed_date = pd.to_datetime(date)

            # Create a new row of data
            new_data = pd.DataFrame({
                'Date': [parsed_date],
                'PickerSpeed': [picker_speed],
                'BalesPerHectare': [bales_per_hectare],
                'MaxBaleEjectionSpeed': [max_bale_ejection_speed],
                'MaxWrapSpeed': [max_wrap_speed]
            })

            # Append the new data to the CSV file
            new_data.to_csv('picker_data.csv', mode='a', header=False, index=False)

            return f"Data submitted successfully on {date}"
        except Exception as e:
            return f"Error: {str(e)}"
    return ''

# Callback to update histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    []
)
def update_histogram():
    data = pd.read_csv('picker_data.csv')
    fig = go.Figure()
    # Plot histograms for available speed variables
    if 'PickerSpeed' in data.columns and data['PickerSpeed'].notnull().any():
        fig.add_trace(go.Histogram(
            x=data['PickerSpeed'],
            nbinsx=20,
            name='Picker Speed',
            marker_color='blue',
            opacity=0.75
        ))
    if 'MaxBaleEjectionSpeed' in data.columns and data['MaxBaleEjectionSpeed'].notnull().any():
        fig.add_trace(go.Histogram(
            x=data['MaxBaleEjectionSpeed'],
            nbinsx=20,
            name='Max Bale Ejection Speed',
            marker_color='red',
            opacity=0.75
        ))
    if 'MaxWrapSpeed' in data.columns and data['MaxWrapSpeed'].notnull().any():
        fig.add_trace(go.Histogram(
            x=data['MaxWrapSpeed'],
            nbinsx=20,
            name='Max Wrap Speed',
            marker_color='green',
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
    []
)
def update_heatmap():
    data = pd.read_csv('picker_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    if data['Date'].notnull().any():
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
    else:
        return go.Figure()

# Callback for prediction output
@app.callback(
    Output('prediction-output', 'children'),
    [Input('input-speed', 'value')]
)
def predict_bale_yield(speed_value):
    if reg_model is not None:
        predicted_yield = reg_model.predict([[speed_value]])[0]
        return html.Div([html.H5(f"Predicted Bales per Hectare: {predicted_yield:.2f}")])
    else:
        return html.Div([html.H5("Prediction model not available.")])

if __name__ == '__main__':
    app.run_server(debug=True)
