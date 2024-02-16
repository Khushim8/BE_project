import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix

# Load data from CSV file
data = pd.read_csv("profile.csv")

# Preprocess data
X = data.drop(columns=['Stable']).values
y = data['Stable'].values

# Reshape X for LSTM input (assuming each row represents a time step)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (stable or not)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate LSTM model
evaluation_result = lstm_model.evaluate(X_test, y_test)
print(f"Test Loss: {evaluation_result[0]}")
print(f"Test Accuracy: {evaluation_result[1]}")

# Make predictions
y_pred = np.round(lstm_model.predict(X_test)).astype(int)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
