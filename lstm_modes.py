import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assuming your dataset is a Pandas DataFrame
# Example:
data = pd.read_csv('C:\\Users\\srushti\\Desktop\\lstm\\main_data.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define a function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Set sequence length
sequence_length = 2  # Adjust as needed

# Create sequences and targets
X, y = create_sequences(data_scaled, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with multiple layers
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(X_train.shape[2])
])

# Adjust optimizer, learning rate, and loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')  

# Train the model with more epochs and a larger batch size
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Inverse transform the predicted values
y_pred_inverse_transformed = scaler.inverse_transform(y_pred)

# Create a DataFrame with the original sensor values and the predicted values
predictions_df = pd.DataFrame(data=y_pred_inverse_transformed, columns=data.columns)

# Adjust threshold values for classification
pressure_threshold = 90
vibration_threshold = 0
temperature_threshold = 33
volume_flow_threshold = 2

# Define fault names and action descriptions
fault_definitions = {
    "Pressure Issue": {
        "action_description": "Check and adjust the pressure settings.",
    },
    "Vibration Issue": {
        "action_description": "Inspect and address any issues causing excessive vibration.",
    },
    "Temperature Issue": {
        "action_description": "Investigate and correct the temperature irregularities.",
    },
    "Volume Flow Issue": {
        "action_description": "Examine and optimize the volume flow system.",
    },
    "Optimal": {
        "action_description": "No specific action required. The system is operating optimally.",
    },
}

# Define a function to classify each row based on the specific thresholds
def classify_row(row):
    if row['Pressure'] < pressure_threshold:
        return "Pressure Issue"
    elif row['Vibration'] <= vibration_threshold:
        return "Vibration Issue"
    elif row['Temperature'] < temperature_threshold:
        return "Temperature Issue"
    elif row['Volume_Flow'] <= volume_flow_threshold:
        return "Volume Flow Issue"
    else:
        return "Optimal"

# Apply the classification function to each row
predictions_df['Fault_Type'] = predictions_df.apply(classify_row, axis=1)

# Add action descriptions to the DataFrame
predictions_df['Action_Description'] = predictions_df['Fault_Type'].map(lambda x: fault_definitions[x]['action_description'])

# Save the classified data to a new CSV file
predictions_df.to_csv('classified_predictions.csv', index=False)

# Mode Selection for Each Batch
# ...

# Mode Selection for Each Batch
batch_counter = 1
while batch_counter <= len(predictions_df) / 10:
    # Display 10 predicted values for the current batch only
    current_batch = predictions_df.iloc[(batch_counter - 1) * 10:batch_counter * 10]
    print(current_batch)

    # Ask for mode selection
    mode = input("\nEnter 'manual' for manual mode or 'auto' for automatic mode: ")

    # Check for faults based on the selected mode
    faults_detected = False
    if mode.lower() == 'manual':
        faults = current_batch[current_batch['Fault_Type'] != 'Optimal']

        if not faults.empty:
            faults_detected = True
            print("\nFault Detected in Manual Mode!")
            print(faults[['Fault_Type', 'Action_Description']])

            # Resolve all faults in the batch
            for _, fault_to_resolve in faults.iterrows():
                print(f"\nResolving Fault: {fault_to_resolve['Fault_Type']}")
                user_input = input("Enter action description: ")

                # Update action description for the resolved fault
                fault_index = fault_to_resolve.name
                predictions_df.at[fault_index, 'Action_Description'] = user_input

    elif mode.lower() == 'auto':
        faults_auto = current_batch[current_batch['Fault_Type'] != 'Optimal']

        if not faults_auto.empty:
            faults_detected = True
            print("\nFault Detected in Automatic Mode!")
            print(faults_auto[['Fault_Type', 'Action_Description']])

    else:
        print("Invalid mode entered. Please enter 'manual' or 'auto'.")

    if faults_detected:
        # Save the current batch to a separate CSV file after detecting and resolving the faults
        batch_filename = f'batch_{batch_counter}_resolved.csv'
        current_batch.to_csv(batch_filename, index=False)
        print(f"\nBatch {batch_counter} saved to CSV: {batch_filename}")

    batch_counter += 1

# Save the resolved faults to a single CSV file
resolved_faults_filename = 'resolved_faults.csv'
resolved_faults_df = predictions_df[predictions_df['Fault_Type'] != 'Optimal']
resolved_faults_df.to_csv(resolved_faults_filename, index=False)
print(f"\nAll resolved faults saved to CSV: {resolved_faults_filename}")

