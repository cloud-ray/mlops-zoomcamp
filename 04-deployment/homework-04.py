import argparse
import pickle
import pandas as pd
import numpy as np
import os

# Function to read and process the data
def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process year and month.")
    parser.add_argument('--year', type=int, required=True, help='Year for the data')
    parser.add_argument('--month', type=int, required=True, help='Month for the data')
    args = parser.parse_args()

    year = args.year
    month = args.month
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    # Load model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    # Read data
    df = read_data(filename)

    # Prepare data for prediction
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Calculate and print standard deviation and mean of predicted durations
    std_dev = np.std(y_pred)
    mean_pred_duration = np.mean(y_pred)
    print(f"Standard deviation of predicted duration: {std_dev:.2f}")
    print(f"Mean predicted duration: {mean_pred_duration:.2f}")

    # Create an artificial ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create a results dataframe with ride_id and predictions
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

    # Define the output file name
    output_file = f'results_{year}_{month:02d}.parquet'

    # Save the results to a parquet file
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    # Get the size of the output file
    file_size = os.path.getsize(output_file)
    results_file_size = file_size / (1024 * 1024)
    print(f"Output file size: {results_file_size:.2f} MB")
