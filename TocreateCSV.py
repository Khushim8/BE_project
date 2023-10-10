import pandas as pd
import numpy as np
import random
import csv

# Define parameters for the hydraulic clamping machine
pressure_mean = 3000  # Mean hydraulic pressure in PSI
pressure_stddev = 500  # Standard deviation of hydraulic pressure
temperature_mean = 65  # Mean temperature in degrees Celsius
temperature_stddev = 5  # Standard deviation of temperature
force_mean = 7500  # Mean clamping force in Newtons
force_stddev = 1000  # Standard deviation of clamping force

# Define the number of data points you want to generate
num_data_points = 1000

# Create and open a CSV file for writing the dataset
with open('hydraulic_clamping_data1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Timestamp', 'Pressure_PSI', 'Temperature_C', 'Clamping_Force_N']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header row to the CSV file
    writer.writeheader()

    # Generate and write the data points to the CSV file
    for i in range(num_data_points):
        timestamp = pd.Timestamp('2023-01-01') + pd.DateOffset(days=i)  # Example: Create a timestamp for each day
        pressure = max(0, np.random.normal(pressure_mean, pressure_stddev))
        temperature = max(0, np.random.normal(temperature_mean, temperature_stddev))
        force = max(0, np.random.normal(force_mean, force_stddev))
        
        # Write the data point to the CSV file
        writer.writerow({'Timestamp': timestamp, 'Pressure_PSI': pressure, 'Temperature_C': temperature, 'Clamping_Force_N': force})

print(f"{num_data_points} data points generated and saved to 'hydraulic_clamping_data1.csv'.")
