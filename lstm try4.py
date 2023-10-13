import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('hydraulic_clamping_data1.csv')

# Extract the relevant columns
data = data[['Pressure', 'Temp', 'Force']]

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define the sequence length (number of time steps to consider for prediction)
sequence_length = 10

# Create sequences and corresponding labels
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(sequence_length, 3)))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(3))  # Output layer with 3 units (pressure, temp, force)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to the original scale
predictions = scaler.inverse_transform(predictions)

def generate_future_values(model, initial_sequence, future_steps):
    future_values = []

    current_sequence = initial_sequence
    for _ in range(future_steps):
        # Predict the next step and append it to the results
        next_step = model.predict(np.array([current_sequence]))[0]
        future_values.append(next_step)

        # Update the current sequence for the next prediction
        current_sequence = np.vstack((current_sequence[1:], next_step))

    return future_values


# Choose an initial sequence from the test set (or any other data)
initial_sequence = X_test[0]

# Define the number of future steps to predict
future_steps = 1000  # You can adjust this as needed

# Generate future values
future_predictions = generate_future_values(model, initial_sequence, future_steps)

# Inverse transform the predictions to the original scale
future_predictions = scaler.inverse_transform(future_predictions)

print("FUTURE PREDICTIONS: ", future_predictions)


