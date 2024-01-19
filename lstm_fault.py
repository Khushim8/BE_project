import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix

class HydraulicClampingSimulator:
    def __init__(self):
        self.machine_state = None
        self.operational_parameters = None
        self.lstm_model = None

    def set_idle_state(self):
        self.machine_state = "Idle"
        self.operational_parameters = {"vibration": 0, "oil_flow": 0, "oil_temperature": 30, "oil_pressure": 50}

    def set_standard_operation_state(self):
        self.machine_state = "Standard Operation"
        self.operational_parameters = {"vibration": 10, "oil_flow": 20, "oil_temperature": 40, "oil_pressure": 70}

    def set_high_operation_state(self):
        self.machine_state = "High Operation"
        self.operational_parameters = {"vibration": 20, "oil_flow": 30, "oil_temperature": 50, "oil_pressure": 90}

    def set_fault_state(self):
        self.machine_state = "Fault"
        self.operational_parameters = {"vibration": 30, "oil_flow": 0, "oil_temperature": 60, "oil_pressure": 120}

    def simulate_data(self, num_samples, include_faults=True):
        simulated_data = []
        for _ in range(num_samples):
            if include_faults and np.random.rand() < 0.1:  # Introduce faults in 10% of the cases
                self.set_fault_state()
            else:
                # Randomly choose other states for non-faulty data
                state_choice = np.random.choice(["Idle", "Standard Operation", "High Operation"])
                if state_choice == "Idle":
                    self.set_idle_state()
                elif state_choice == "Standard Operation":
                    self.set_standard_operation_state()
                else:
                    self.set_high_operation_state()

            noise = np.random.normal(0, 5, 4)
            parameters = {key: value + noise[i] for i, (key, value) in enumerate(self.operational_parameters.items())}

            simulated_data.append({"state": self.machine_state, "parameters": parameters})

        return simulated_data

    def save_to_csv(self, data, filename):
        keys = data[0].keys() if data else []
        with open(filename, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerows(data)

    def train_lstm(self, data):
        X = [sample["parameters"].values() for sample in data]
        y = [sample["state"] for sample in data]

        X = [list(x) for x in X]

        label_mapping = {"Idle": 0, "Standard Operation": 1, "High Operation": 2, "Fault": 3}
        y = [label_mapping[state] for state in y]

        X = np.array(X)
        y = np.array(y)

        X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        lstm_model.add(Dense(4, activation='softmax'))

        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

        evaluation_result = lstm_model.evaluate(X_test, y_test)
        print(f"Test Loss: {evaluation_result[0]}")
        print(f"Test Accuracy: {evaluation_result[1]}")

        y_pred = np.argmax(lstm_model.predict(X_test), axis=-1)

        state_mapping = {0: "Idle", 1: "Standard Operation", 2: "High Operation", 3: "Fault"}
        y_test_states = np.array([state_mapping[label] for label in y_test])
        y_pred_states = np.array([state_mapping[label] for label in y_pred])

        print("Classification Report:")
        print(classification_report(y_test_states, y_pred_states))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test_states, y_pred_states))

        self.lstm_model = lstm_model

    def predict_machine_state(self, operational_parameters):
        if self.lstm_model:
            X = np.array([list(operational_parameters.values())])
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            prediction = np.argmax(self.lstm_model.predict(X), axis=-1)[0]

            state_mapping = {0: "Idle", 1: "Standard Operation", 2: "High Operation", 3: "Fault"}
            return state_mapping[prediction]
        else:
            print("Error: LSTM model is not trained.")

# Example Usage:
simulator = HydraulicClampingSimulator()

simulator.set_idle_state()
idle_data = simulator.simulate_data(num_samples=100)

simulator.set_standard_operation_state()
standard_operation_data = simulator.simulate_data(num_samples=100)

simulator.set_high_operation_state()
high_operation_data = simulator.simulate_data(num_samples=100)

fault_data = simulator.simulate_data(num_samples=30, include_faults=True)

all_data = idle_data + standard_operation_data + high_operation_data + fault_data

simulator.save_to_csv(all_data, 'lstm_data_with_faults.csv')

simulator.train_lstm(all_data)

# Example prediction with operational parameters that indicate a fault
operational_parameters_faulty = {"vibration": 30, "oil_flow": 0, "oil_temperature": 60, "oil_pressure": 120}
predicted_state_faulty = simulator.predict_machine_state(operational_parameters_faulty)
print(f"Predicted Machine State (Faulty): {predicted_state_faulty}")

# Example prediction with operational parameters that indicate non-faulty conditions
operational_parameters_non_faulty = {"vibration": 15, "oil_flow": 25, "oil_temperature": 45, "oil_pressure": 80}
predicted_state_non_faulty = simulator.predict_machine_state(operational_parameters_non_faulty)
print(f"Predicted Machine State (Non-Faulty): {predicted_state_non_faulty}")
