import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
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
    return data

data = load_or_generate_data()
data['Date'] = pd.to_datetime(data['Date'])

# Machine learning model
# Load or train model
def load_or_train_model():
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
            html.Br(),
            html.H4("Upload Data"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload'),
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
    data = pd.concat([data, pd.DataFrame([new_data_point])], ignore_index=True)

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

# Callback to handle file uploads
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if set(['Date', 'PickerSpeed', 'BalesPerHectare']).issubset(df_uploaded.columns):
                df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
                global data
                data = pd.concat([data, df_uploaded], ignore_index=True)
                return html.Div(['Data uploaded successfully.'])
            else:
                return html.Div(['Invalid file format. Please ensure the CSV contains Date, PickerSpeed, and BalesPerHectare columns.'])
        except Exception as e:
            return html.Div([f'Error processing file: {str(e)}'])
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)

