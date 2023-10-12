import pandas as pd
from flask import Flask, render_template, request
from statsmodels.tsa.arima_model import ARIMA

# Load the CSV file
df = pd.read_csv('hydraulic_clamping_data1.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data=df)

@app.route('/forecast', methods=['POST'])
def forecast():
    start_date = pd.Timestamp(request.form['start_date'])
    end_date = pd.Timestamp(request.form['end_date'])
    column_to_forecast = request.form['column']  # 'Pressure_PSI', 'Temperature_C', or 'Clamping_Force_N'

    # Filter data for the selected time range
    subset = df[(df.index >= start_date) & (df.index <= end_date)]

    # Extract the time series data
    time_series = subset[column_to_forecast]

    # Fit ARIMA model and make forecasts
    model = ARIMA(time_series, order=(1, 1, 1))
    results = model.fit()
    forecast_steps = len(subset)
    forecast, _, _ = results.forecast(steps=forecast_steps)

    return render_template('forecast.html', data=subset, forecast=forecast)

if __name__ == '__main__':
    app.run(debug=True)