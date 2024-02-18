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

def average_values(input_csv_path, output_csv_path, sampling_rate):
    # Load your dataset from CSV
    df = pd.read_csv(input_csv_path, header=None, names=['values'])

    # Create a datetime index with appropriate frequency based on sampling rate
    df.index = pd.date_range(start='2024-02-15', periods=len(df), freq=f'{1/sampling_rate}S')

    # Resample the data to one-second intervals and calculate the mean
    df_resampled = df.resample('1S').mean()

    # Reset the index to remove date and time information
    df_resampled.reset_index(drop=True, inplace=True)

    # Save the resampled data to a new CSV file
    df_resampled.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    input_text_filename = 'C://Users//khush_52iezzo//OneDrive//Documents//BE_project//condition+monitoring+of+hydraulic+systems//VS1.txt'
    output_csv_filename = 'VS1.csv'
    output_resampled_csv_filename = 'VS1_r.csv'
    
    # Convert text to CSV
    text_to_csv(input_text_filename, output_csv_filename)
    
    # Process CSV file
    average_values(output_csv_filename, output_resampled_csv_filename, 1)


