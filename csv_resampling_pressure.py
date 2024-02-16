import csv
import pandas as pd

def text_to_csv(input_text_path, output_csv_path):
    with open(input_text_path, 'r') as text_file:
        lines = text_file.readlines()

    # Process each line separately and split based on spaces
    data = [[word.strip()] for line in lines for word in line.split()]

    with open(output_csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def average_values(input_csv_path, output_csv_path, sampling_rate, num_attributes):
    # Load your dataset from CSV
    df = pd.read_csv(input_csv_path, header=None, names=['values'])

    # Calculate the number of periods (rows) in the resampled data
    num_periods = num_attributes / sampling_rate

    # Set the index to a datetime index
    df.index = pd.date_range(start='2024-02-15', periods=len(df), freq=f'{1/sampling_rate}S')

    # Resample the data to one-minute intervals and calculate the mean
    df_resampled = df.resample(f'{num_periods}S').mean()

    # Save the resampled data to a new CSV file
    df_resampled.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    input_text_filename = 'C://Users//khush_52iezzo//OneDrive//Documents//BE_project//condition+monitoring+of+hydraulic+systems//PS1.txt'  # Provide the path to your input text file
    output_csv_filename = 'PS1.csv'  # Provide the desired output CSV filename
    output_resampled_csv_filename = 'PS1_r.csv'  # Provide the desired resampled output CSV filename
    
    # Convert text to CSV
    text_to_csv(input_text_filename, output_csv_filename)
    
    # Process CSV file
    average_values(output_csv_filename, output_resampled_csv_filename, 100, 6000)  # Sampling rate is 100 Hz and 6000 attributes per sensor
