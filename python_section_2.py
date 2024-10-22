import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix from the given dataset CSV file.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: A DataFrame representing distances between IDs.
    """

    df = pd.read_csv(file_path)

    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    n = len(ids)

    distance_matrix = pd.DataFrame(np.nan, index=ids, columns=ids)

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']  
        
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance  


    for k in ids:
        for i in ids:
            for j in ids:
                if pd.notna(distance_matrix.at[i, k]) and pd.notna(distance_matrix.at[k, j]):
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if pd.isna(distance_matrix.at[i, j]) or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance


    for id in ids:
        distance_matrix.at[id, id] = 0.0

    distance_matrix.fillna(np.inf, inplace=True)

    return distance_matrix





import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix DataFrame into a long format with id_start, id_end, and distance.

    Args:
        distance_matrix (pd.DataFrame): The distance matrix DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with three columns: id_start, id_end, and distance.
    """
    results = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end: 
                distance = distance_matrix.at[id_start, id_end]
       
                if pd.notna(distance) and distance != float('inf'):
                    results.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance
                    })

    unrolled_df = pd.DataFrame(results)

    return unrolled_df






import pandas as pd
import numpy as np

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): The DataFrame containing id_start and distances.
        reference_id (int): The ID to reference for calculating the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_threshold = reference_avg_distance * 0.90
    upper_threshold = reference_avg_distance * 1.10

    filtered_ids = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    result_ids = filtered_ids['id_start'].unique()
    
    return pd.DataFrame({'id_start': sorted(result_ids)})


df = pd.DataFrame({'id_start': [1, 2, 3, 1, 2], 'distance': [10, 15, 13, 11, 16]})
print(find_ids_within_ten_percentage_threshold(df, 1))



import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing vehicle distances.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for toll rates.
    """

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df[vehicle] * coefficient

    return df

data = {
    'id_start': [1001400]*10,
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420],
    'moto': [7.76, 23.92, 36.72, 54.08, 62.96, 75.44, 90.00, 100.56, 111.44, 121.76],
    'car': [11.64, 35.88, 55.08, 81.12, 94.44, 113.16, 135.00, 150.84, 167.16, 182.64],
    'rv': [14.55, 44.85, 68.85, 101.40, 118.05, 141.45, 168.75, 188.55, 208.95, 228.30],
    'bus': [21.34, 65.78, 100.98, 148.72, 173.14, 207.46, 247.50, 276.54, 306.46, 334.84],
    'truck': [34.92, 107.64, 165.24, 243.36, 283.32, 339.48, 405.00, 452.52, 501.48, 547.92]
}

df = pd.DataFrame(data)

result_df = calculate_toll_rate(df)

result_df



import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): The input DataFrame containing vehicle distances and toll rates.

    Returns:
        pandas.DataFrame: The updated DataFrame with time-based toll rates and time columns.
    """
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    time_intervals = {
        "Weekday": [
            (time(0, 0), time(10, 0), 0.8),
            (time(10, 0), time(18, 0), 1.2),
            (time(18, 0), time(23, 59), 0.8),
        ],
        "Weekend": [
            (time(0, 0), time(23, 59), 0.7)
        ]
    }

    result_rows = []

    
    for _, row in df.iterrows():
        
        for day in days_of_week[:5]:
            for start_time, end_time, discount in time_intervals["Weekday"]:
                result_rows.append([
                    row['id_start'],
                    row['id_end'],
                    row['moto'] * discount,
                    row['car'] * discount,
                    row['rv'] * discount,
                    row['bus'] * discount,
                    row['truck'] * discount,
                    day,
                    start_time,
                    day,
                    end_time
                ])
        
        
        for day in days_of_week[5:]:
            for start_time, end_time, discount in time_intervals["Weekend"]:
                result_rows.append([
                    row['id_start'],
                    row['id_end'],
                    row['moto'] * discount,
                    row['car'] * discount,
                    row['rv'] * discount,
                    row['bus'] * discount,
                    row['truck'] * discount,
                    day,
                    start_time,
                    day,
                    end_time
                ])

    
    result_df = pd.DataFrame(result_rows, columns=[
        'id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck', 
        'start_day', 'start_time', 'end_day', 'end_time'
    ])
    
    return result_df

