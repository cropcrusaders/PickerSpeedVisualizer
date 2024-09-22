import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # Import PolynomialFeatures
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

# Precompute min and max values
picker_speed_min = round(data['PickerSpeed'].min(), 1)
picker_speed_max = round(data['PickerSpeed'].max(), 1)

bales_min = round(data['BalesPerHectare'].min(), 1)
bales_max = round(data['BalesPerHectare'].max(), 1)

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
                min=bales_min,
                max=bales_max,
                step=0.1,
                value=[bales_min, bales_max],
                marks={i: f'{i}' for i in np.arange(bales_min, bales_max + 1, 1)},
            ),
            html.Br(),
            html.Label('Picker Speed Range (km/h)'),
            dcc.RangeSlider(
                id='speed-range',
                min=picker_speed_min,
                max=picker_speed_max,
                step=0.1,
                value=[picker_speed_min, picker_speed_max],
                marks={round(i, 1): f'{round(i, 1)}' for i in np.arange(picker_speed_min, picker_speed_max + 0.5, 0.5)},
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

# Callback to update the scatter plot with polynomial trend lines
@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('speed-range', 'value'),
        Input('bales-range', 'value'),
        Input('data-submit-output', 'children')  # Trigger update when new data is submitted
    ]
)
def update_scatter(speed_range, bales_range, _):
    # Reload data to include any new submissions
    data = load_data()
    data = data.rename(columns={
        'Yield (Bales Per Hectare)': 'BalesPerHectare',
        'Max Wrap and Ejection Process (km/h)': 'MaxWrapEjectionSpeed',
        'Max for Row Units (km/h)': 'PickerSpeed'
    })
    data['BalesPerHectare'] = pd.to_numeric(data['BalesPerHectare'], errors='coerce')
    data['MaxWrapEjectionSpeed'] = pd.to_numeric(data['MaxWrapEjectionSpeed'], errors='coerce')
    data['PickerSpeed'] = pd.to_numeric(data['PickerSpeed'], errors='coerce')
    data = data.dropna(subset=['BalesPerHectare', 'MaxWrapEjectionSpeed', 'PickerSpeed'])

    filtered_data = data[
        (data['PickerSpeed'] >= speed_range[0]) &
        (data['PickerSpeed'] <= speed_range[1]) &
        (data['BalesPerHectare'] >= bales_range[0]) &
        (data['BalesPerHectare'] <= bales_range[1])
    ]

    # Create the figure
    fig = go.Figure()

    # Plot Picker Speed vs Yield
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['PickerSpeed'],
        mode='markers',
        name='Picker Speed',
        marker=dict(color='blue'),
    ))

    # Plot Max Wrap and Ejection Speed vs Yield
    fig.add_trace(go.Scatter(
        x=filtered_data['BalesPerHectare'],
        y=filtered_data['MaxWrapEjectionSpeed'],
        mode='markers',
        name='Max Wrap and Ejection Speed',
        marker=dict(color='red'),
    ))

    # Compute polynomial trend lines for each variable
    trend_variables = {
        'Picker Speed': ('PickerSpeed', 'blue'),
        'Max Wrap and Ejection Speed': ('MaxWrapEjectionSpeed', 'red'),
    }

    degree = 2  # Degree of the polynomial regression

    for name, (variable, color) in trend_variables.items():
        X = filtered_data[['BalesPerHectare']]
        y = filtered_data[variable]
        if len(X) > degree:  # Ensure enough data points
            # Fit polynomial regression
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)

            # Generate trend line
            trend_x_values = np.linspace(X['BalesPerHectare'].min(), X['BalesPerHectare'].max(), 100)
            trend_x = pd.DataFrame({'BalesPerHectare': trend_x_values})
            trend_x_poly = poly_features.transform(trend_x)
            trend_y = model.predict(trend_x_poly)

            fig.add_trace(go.Scatter(
                x=trend_x['BalesPerHectare'],
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
    State('input-bales-per-hectare', 'value'),
    State('input-picker-speed', 'value'),
    State('input-max-wrap-ejection-speed', 'value')
)
def add_data_to_csv(n_clicks, bales_per_hectare, picker_speed, max_wrap_ejection_speed):
    if n_clicks > 0:
        try:
            if picker_speed is None or bales_per_hectare is None or max_wrap_ejection_speed is None:
                return 'Please provide Yield, Picker Speed, and Max Wrap and Ejection Speed.'

            # Create a new row of data
            new_data = pd.DataFrame({
                'Yield (Bales Per Hectare)': [bales_per_hectare],
                'Max Wrap and Ejection Process (km/h)': [max_wrap_ejection_speed],
                'Max for Row Units (km/h)': [picker_speed],
            })

            # Append the new data to the CSV file
            new_data.to_csv('user_data.csv', mode='a', header=False, index=False)

            # Retrain the model with the new data
            global data, reg_model
            data = load_data()
            data = data.rename(columns={
                'Yield (Bales Per Hectare)': 'BalesPerHectare',
                'Max Wrap and Ejection Process (km/h)': 'MaxWrapEjectionSpeed',
                'Max for Row Units (km/h)': 'PickerSpeed'
            })
            data['BalesPerHectare'] = pd.to_numeric(data['BalesPerHectare'], errors='coerce')
            data['MaxWrapEjectionSpeed'] = pd.to_numeric(data['MaxWrapEjectionSpeed'], errors='coerce')
            data['PickerSpeed'] = pd.to_numeric(data['PickerSpeed'], errors='coerce')
            data = data.dropna(subset=['BalesPerHectare', 'MaxWrapEjectionSpeed', 'PickerSpeed'])

            reg_model = train_model()

            return "Data submitted successfully."
        except Exception as e:
            return f"Error: {str(e)}"
    return ''

# Callback to update histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('tabs', 'active_tab'), Input('data-submit-output', 'children')]
)
def update_histogram(active_tab, _):
    if active_tab != 'histogram-tab':
        raise dash.exceptions.PreventUpdate

    # Reload data to include any new submissions
    data = load_data()
    data = data.rename(columns={
        'Yield (Bales Per Hectare)': 'BalesPerHectare',
        'Max Wrap and Ejection Process (km/h)': 'MaxWrapEjectionSpeed',
        'Max for Row Units (km/h)': 'PickerSpeed'
    })
    data['BalesPerHectare'] = pd.to_numeric(data['BalesPerHectare'], errors='coerce')
    data['MaxWrapEjectionSpeed'] = pd.to_numeric(data['MaxWrapEjectionSpeed'], errors='coerce')
    data['PickerSpeed'] = pd.to_numeric(data['PickerSpeed'], errors='coerce')
    data = data.dropna(subset=['BalesPerHectare', 'MaxWrapEjectionSpeed', 'PickerSpeed'])

    fig = go.Figure()
    # Plot histograms for available variables
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
        Input('input-yield', 'value')
    ]
)
def predict_speeds(yield_value):
    # Reload the model in case it has been retrained
    reg_model = joblib.load('reg_model.pkl')
    input_df = pd.DataFrame({'BalesPerHectare': [yield_value]})
    predicted_speeds = reg_model.predict(input_df)[0]
    picker_speed = predicted_speeds[0]
    max_wrap_ejection_speed = predicted_speeds[1]
    return html.Div([
        html.H5(f"Predicted Picker Speed (km/h): {picker_speed:.2f}"),
        html.H5(f"Predicted Max Wrap and Ejection Speed (km/h): {max_wrap_ejection_speed:.2f}")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)

