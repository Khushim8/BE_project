import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load the CSV file
df = pd.read_csv('hydraulic_clamping_data1.csv')

# Convert the 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set the 'Timestamp' column as the index
df.set_index('Timestamp', inplace=True)

# Now you can access the 'Pressure_PSI' and 'Temperature_C' columns as time series
pressure_series = df['Pressure_PSI']
temperature_series = df['Temperature_C']

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(pressure_series, label='Pressure (PSI)')
plt.plot(temperature_series, label='Temperature (C)')
plt.title('Hydraulic Pressure and Temperature Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (PSI) / Temperature (C)')
plt.legend()
plt.show()

# Function to create sequences of data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define the sequence length
seq_length = 10

# Create sequences for the 'Pressure_PSI' time series
pressure_data = pressure_series.values
pressure_sequences = create_sequences(pressure_data, seq_length)

# Split data into training and testing sets
train_size = int(0.8 * len(pressure_sequences))
train_sequences, test_sequences = pressure_sequences[:train_size], pressure_sequences[train_size:]

# Create DataLoader for training
train_loader = DataLoader(train_sequences, batch_size=32, shuffle=True)

# Define a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define input and output sizes
input_size = 1
hidden_size = 64

# Create the LSTM model
model = SimpleLSTM(input_size, hidden_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for seq, label in train_loader:
        optimizer.zero_grad()
        seq = seq.view(-1, seq_length, input_size).float()
        outputs = model(seq)
        label = label.view(-1, 1).float()
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prepare test data
test_data = np.array([item[0] for item in test_sequences])
test_labels = np.array([item[1] for item in test_sequences])

# Make predictions
model.eval()
with torch.no_grad():
    test_data = torch.Tensor(test_data).float().view(-1, seq_length, input_size)
    predictions = model(test_data).numpy()

# Calculate and print Mean Squared Error
mse = mean_squared_error(test_labels, predictions)
print(f'Mean Squared Error: {mse:.4f}')

print("Predicted Pressure Values:")
print(predictions)

# Ensure pressure_series.iloc[-1] is a scalar value
pressure_scalar = pressure_series.iloc[-1]

# Create a NumPy array for the x-axis
x = np.array(pd.date_range(start=pressure_series.index[-1], periods=len(predictions) + 1))

# Create a NumPy array for the y-axis
y = np.concatenate(([pressure_scalar], predictions.reshape(-1)))

# Plot the actual data and forecast
plt.figure(figsize=(12, 6))
plt.plot(pressure_series, label='Actual Data', linestyle='-')
plt.plot(x, y, label='Forecast', linestyle='--')
plt.title('Hydraulic Pressure Forecast')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (PSI)')
plt.legend()
plt.show()

