import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class HydraulicClampingSimulator:
    def __init__(self):
        self.machine_state = None
        self.operational_parameters = None
        self.rf_model = None

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

    def train_random_forest(self, data):
        X = [sample["parameters"].values() for sample in data]
        y = [sample["state"] for sample in data]

        X = [list(x) for x in X]

        # Include "fault" as a feature
        label_mapping = {"Idle": 0, "Standard Operation": 1, "High Operation": 2, "Fault": 3}
        y = [label_mapping[state] for state in y]

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_model.fit(X_train, y_train)

        predictions = rf_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        target_names = ["Idle", "Standard Operation", "High Operation", "Fault"]
        print(classification_report(y_test, predictions, target_names=target_names))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        self.rf_model = rf_model

    def predict_machine_state(self, operational_parameters):
        if self.rf_model:
            X = np.array([list(operational_parameters.values())])

            prediction = self.rf_model.predict(X)[0]

            state_mapping = {0: "Idle", 1: "Standard Operation", 2: "High Operation", 3: "Fault"}
            return state_mapping[prediction]
        else:
            print("Error: Random Forest model is not trained.")

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

simulator.save_to_csv(all_data, 'random_forest_data_with_faults.csv')

simulator.train_random_forest(all_data)

# Example prediction with operational parameters that indicate a fault
operational_parameters_faulty = {"vibration": 30, "oil_flow": 0, "oil_temperature": 60, "oil_pressure": 120}
predicted_state_faulty = simulator.predict_machine_state(operational_parameters_faulty)
print(f"Predicted Machine State (Faulty): {predicted_state_faulty}")

# Example prediction with operational parameters that indicate non-faulty conditions
operational_parameters_non_faulty = {"vibration": 15, "oil_flow": 25, "oil_temperature": 45, "oil_pressure": 80}
predicted_state_non_faulty = simulator.predict_machine_state(operational_parameters_non_faulty)
print(f"Predicted Machine State (Non-Faulty): {predicted_state_non_faulty}")
