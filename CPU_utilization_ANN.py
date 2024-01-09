import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Specify the directory where your CSV files are stored
dataset_directory = 'actual_dataset'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(dataset_directory) if file.endswith('.csv')]

# Specify the future timestamp for prediction
future_timestamp = '2024-01-10 23:59:59'  # Replace with your desired future timestamp

# Initialize lists to store VMIDs and timestamps for each category
over_utilized_cpu_records = []
under_utilized_cpu_records = []
over_utilized_memory_records = []
under_utilized_memory_records = []

# Initialize dataframes to store features and labels
features = pd.DataFrame()
labels_cpu = pd.DataFrame()
labels_memory = pd.DataFrame()
#labels_under_cpu = pd.DataFrame()
#labels_under_memory = pd.DataFrame()

# Iterate through each CSV file
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(dataset_directory, csv_file)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract relevant columns for the given future timestamp
    data_at_timestamp = df[df['Timestamp [hr-m-s]'] <= future_timestamp][['CPU usage [%]', 'Memory Usage %', 'ID', 'Timestamp [hr-m-s]']]

    # Handle infinite or too large values
    data_at_timestamp.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_at_timestamp.dropna(inplace=True)

    # Thresholds
    cpu_over_threshold = 90.00
    cpu_under_threshold = 20.00
    memory_over_threshold = 90.00
    memory_under_threshold = 20.00

    # Categorize CPU and Memory utilization
    over_utilized_cpu_records.extend(data_at_timestamp[data_at_timestamp['CPU usage [%]'] > cpu_over_threshold][['ID', 'Timestamp [hr-m-s]']].values.tolist())
    under_utilized_cpu_records.extend(data_at_timestamp[data_at_timestamp['CPU usage [%]'] < cpu_under_threshold][['ID', 'Timestamp [hr-m-s]']].values.tolist())
    over_utilized_memory_records.extend(data_at_timestamp[data_at_timestamp['Memory Usage %'] > memory_over_threshold][['ID', 'Timestamp [hr-m-s]']].values.tolist())
    under_utilized_memory_records.extend(data_at_timestamp[data_at_timestamp['Memory Usage %'] < memory_under_threshold][['ID', 'Timestamp [hr-m-s]']].values.tolist())

    # Append data to features and labels
    features = pd.concat([features, data_at_timestamp[['CPU usage [%]', 'Memory Usage %']]], ignore_index=True)
    labels_cpu = pd.concat([labels_cpu, (data_at_timestamp['CPU usage [%]'] > cpu_over_threshold).astype(int)], ignore_index=True)
    labels_memory = pd.concat([labels_memory, (data_at_timestamp['Memory Usage %'] > memory_over_threshold).astype(int)], ignore_index=True)

# Split the data into training and testing sets for CPU utilization
X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(
    features, labels_cpu, test_size=0.2, random_state=42
)

# Split the data into training and testing sets for Memory utilization
X_train_memory, X_test_memory, y_train_memory, y_test_memory = train_test_split(
    features, labels_memory, test_size=0.2, random_state=42
)

# Handle infinite or too large values in features for CPU utilization
X_train_cpu.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_cpu.dropna(inplace=True)
X_test_cpu.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_cpu.dropna(inplace=True)

# Handle infinite or too large values in features for Memory utilization
X_train_memory.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_memory.dropna(inplace=True)
X_test_memory.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_memory.dropna(inplace=True)

# Standardize features using StandardScaler for CPU utilization
scaler_cpu = StandardScaler()

# Fit and transform the scaler on the training set for CPU utilization
X_train_scaled_cpu = scaler_cpu.fit_transform(X_train_cpu)

# Transform the testing set using the same scaler for CPU utilization
X_test_scaled_cpu = scaler_cpu.transform(X_test_cpu)

# Build and train the neural network for CPU utilization
model_cpu = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42, verbose=False)
model_cpu.fit(X_train_scaled_cpu, y_train_cpu)

# Standardize features using StandardScaler for Memory utilization
scaler_memory = StandardScaler()

# Fit and transform the scaler on the training set for Memory utilization
X_train_scaled_memory = scaler_memory.fit_transform(X_train_memory)

# Transform the testing set using the same scaler for Memory utilization
X_test_scaled_memory = scaler_memory.transform(X_test_memory)

# Build and train the neural network for Memory utilization
model_memory = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42, verbose=False)
model_memory.fit(X_train_scaled_memory, y_train_memory)

# Print training loss and accuracy for CPU utilization
print("\nTraining Loss and Accuracy for CPU Utilization:")
print(f"Iteration - loss: {model_cpu.loss_:.4f} - accuracy: {model_cpu.score(X_train_scaled_cpu, y_train_cpu):.4f}")

# Print training loss and accuracy for Memory utilization
print("\nTraining Loss and Accuracy for Memory Utilization:")
print(f"Iteration - loss: {model_memory.loss_:.4f} - accuracy: {model_memory.score(X_train_scaled_memory, y_train_memory):.4f}")


# Evaluate the models on the test set for CPU utilization
accuracy_cpu = model_cpu.score(X_test_scaled_cpu, y_test_cpu)
print(f'Test Accuracy (CPU): {accuracy_cpu:.4f}')

# Evaluate the models on the test set for Memory utilization
accuracy_memory = model_memory.score(X_test_scaled_memory, y_test_memory)
print(f'Test Accuracy (Memory): {accuracy_memory:.4f}')

# Print VMs for each category with corresponding timestamps in tabular format for CPU utilization
print("\nOver Utilized (CPU) VMIDs and Timestamps:")
print(pd.DataFrame(over_utilized_cpu_records, columns=['ID', 'Timestamp [hr-m-s]']))
print("\nUnder Utilized (CPU) VMIDs and Timestamps:")
print(pd.DataFrame(under_utilized_cpu_records, columns=['ID', 'Timestamp [hr-m-s]']))

# Print VMs for each category with corresponding timestamps in tabular format for Memory utilization
print("\nOver Utilized (Memory) VMIDs and Timestamps:")
print(pd.DataFrame(over_utilized_memory_records, columns=['ID', 'Timestamp [hr-m-s]']))
print("\nUnder Utilized (Memory) VMIDs and Timestamps:")
print(pd.DataFrame(under_utilized_memory_records, columns=['ID', 'Timestamp [hr-m-s]']))
