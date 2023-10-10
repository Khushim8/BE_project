import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Load the CSV file
df = pd.read_csv('hydraulic_clamping_data1.csv')

# Convert 'Timestamp' to a datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set 'Timestamp' as the index
df.set_index('Timestamp', inplace=True)

# Select the column to forecast
column_to_forecast = 'Pressure_PSI'

# Extract the time series data
time_series = df[column_to_forecast]

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(time_series)
plt.title(f'{column_to_forecast} Over Time')
plt.xlabel('Timestamp')
plt.ylabel(column_to_forecast)
plt.show()

# Fit ARIMA model and make forecasts
model = ARIMA(time_series, order=(1, 1, 1))
results = model.fit()
forecast_steps = 10  # Number of forecasted steps
forecast, stderr, conf_int = results.forecast(steps=forecast_steps)

print("Forecasted Pressure Values:")
print(forecast)

# Plot the actual data and forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Actual Data', linestyle='-')
plt.plot(pd.date_range(start=time_series.index[-1], periods=forecast_steps+1, closed='right'), [time_series.iloc[-1]] + list(forecast), label='Forecast', linestyle='--')
plt.title(f'{column_to_forecast} Forecast')
plt.xlabel('Timestamp')
plt.ylabel(column_to_forecast)
plt.legend()
plt.show()
