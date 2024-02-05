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
        self.fault_description = None
        self.simulation_data = []

    def generate_random_operational_parameters(self):
        return {
            "vibration": np.random.uniform(0, 30),
            "oil_flow": np.random.uniform(0, 40),
            "oil_temperature": np.random.uniform(20, 70),
            "oil_pressure": np.random.uniform(40, 130)
        }

    def set_idle_state(self):
        self.machine_state = "Idle"
        self.operational_parameters = {"vibration": 0, "oil_flow": 0, "oil_temperature": 30, "oil_pressure": 50}

    def set_standard_operation_state(self):
        self.machine_state = "Standard Operation"
        self.operational_parameters = {"vibration": 10, "oil_flow": 20, "oil_temperature": 40, "oil_pressure": 70}

    def set_high_operation_state(self):
        self.machine_state = "High Operation"
        self.operational_parameters = {"vibration": 20, "oil_flow": 30, "oil_temperature": 50, "oil_pressure": 90}

    def set_fault_state(self, fault_description):
        self.machine_state = "Fault"
        self.operational_parameters = {"vibration": 30, "oil_flow": 0, "oil_temperature": 60, "oil_pressure": 120}
        self.fault_description = fault_description

    def generate_actions_for_fault(self, fault_description):
        actions_dict = {
            "Low Pressure": "Check for leaks, inspect pressure relief valve, and increase oil supply if necessary.",
            "High Pressure": "Inspect pressure relief valve, reduce load, and check for blockages in the hydraulic system.",
            "Low Temperature": "Check and repair the heating system, ensure proper insulation, and monitor ambient conditions.",
            "High Temperature": "Check and repair the cooling system, reduce load, and monitor ambient conditions.",
            "Low Oil Flow": "Inspect for blockages in pipelines, increase pump speed, and check for pump efficiency.",
            "No Oil Flow": "Inspect and repair pump, check for blockages in pipelines, and restart the pump."
        }
        return actions_dict.get(fault_description, "No specific actions defined for this fault.")

    def simulate_and_get_fault(self, operational_parameters):
        if np.random.rand() < 0.1:  # Introduce faults in 10% of the cases
            fault_description = np.random.choice(["Low Pressure", "High Pressure", "Low Temperature", "High Temperature", "Low Oil Flow", "No Oil Flow"])
            self.set_fault_state(fault_description)
            return fault_description
        else:
            return None

    def save_to_csv(self, filename):
        keys = self.simulation_data[0].keys() if self.simulation_data else []
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            for row in self.simulation_data:
                row["operational_parameters"] = str(row["operational_parameters"])  # Convert dict to string
                writer.writerow(row)

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

    def simulate_and_predict(self, num_samples, mode="manual"):
        for sample_num in range(1, num_samples + 1):
            operational_parameters = self.generate_random_operational_parameters()

            if mode == "manual":
                fault_description = self.simulate_and_get_fault(operational_parameters)
                print(f"\nSample {sample_num} - Operational Parameters: {operational_parameters}")
                print(f"Fault Description: {fault_description}")

                if fault_description:
                    action = input("Please enter the action you want to take: ")
                    print(f"Action recorded: {action}")
                else:
                    action = None

                self.simulation_data.append({
                    "sample_num": sample_num,
                    "operational_parameters": operational_parameters,
                    "fault_description": fault_description,
                    "action": action
                })

            elif mode == "auto":
                fault_description = self.simulate_and_get_fault(operational_parameters)
                action = self.generate_actions_for_fault(fault_description)
                print(f"\nSample {sample_num} - Fault: {fault_description}, Action: {action}")

                self.simulation_data.append({
                    "sample_num": sample_num,
                    "operational_parameters": operational_parameters,
                    "fault_description": fault_description,
                    "action": action
                })

        # Save simulation data to CSV
        self.save_to_csv("simulation_data.csv")


    def introduce_fault_based_on_thresholds(self):
        # Set thresholds for each sensor
        vibration_threshold = 25
        oil_flow_threshold = 10
        oil_temperature_threshold = 55
        oil_pressure_threshold = 100

        # Check if any parameter exceeds its threshold
        if (
            self.operational_parameters["vibration"] > vibration_threshold or
            self.operational_parameters["oil_flow"] < oil_flow_threshold or
            self.operational_parameters["oil_temperature"] > oil_temperature_threshold or
            self.operational_parameters["oil_pressure"] > oil_pressure_threshold
        ):
            return True  # Fault introduced
        else:
            return False  # No fault

    def simulate_and_get_fault_with_thresholds(self, operational_parameters):
        if self.introduce_fault_based_on_thresholds():
            return self.fault_description
        else:
            return None

    def simulate_and_predict_with_thresholds(self, num_samples, mode="manual"):
        for sample_num in range(1, num_samples + 1):
            operational_parameters = self.generate_random_operational_parameters()

            if mode == "manual":
                fault_description = self.simulate_and_get_fault_with_thresholds(operational_parameters)
                print(f"\nSample {sample_num} - Operational Parameters: {operational_parameters}")
                print(f"Fault Description: {fault_description}")

                if fault_description:
                    action = input("Please enter the action you want to take: ")
                    print(f"Action recorded: {action}")
                else:
                    action = None

                self.simulation_data.append({
                    "sample_num": sample_num,
                    "operational_parameters": operational_parameters,
                    "fault_description": fault_description,
                    "action": action
                })

            elif mode == "auto":
                fault_description = self.simulate_and_get_fault_with_thresholds(operational_parameters)
                action = self.generate_actions_for_fault(fault_description)
                print(f"\nSample {sample_num} - Fault: {fault_description}, Action: {action}")

                self.simulation_data.append({
                    "sample_num": sample_num,
                    "operational_parameters": operational_parameters,
                    "fault_description": fault_description,
                    "action": action
                })

        # Save simulation data to CSV
        self.save_to_csv("simulation_data_with_thresholds.csv")

    


# Example Usage:
simulator = HydraulicClampingSimulator()

while True:
    simulation_mode = input("Choose simulation mode (manual/auto), or enter 'exit' to exit: ").lower()

    if simulation_mode == 'exit':
        print("Exiting simulation.")
        break

    simulator.simulate_and_predict(num_samples=10, mode=simulation_mode)