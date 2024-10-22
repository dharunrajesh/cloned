from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = []
        
        for j in range(n):
            if i + j < len(lst):
                group.append(lst[i + j])
        
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        result.extend(reversed_group)
    
    return result


print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))         
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4)) 


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}

    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    return dict(sorted(length_dict.items()))

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
print(group_by_length(["one", "two", "three", "four"])) 

    
from typing import Any, Dict
def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    flattened = {}  
    for key, value in nested_dict.items():
        
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    flattened.update(flatten_dict(item, list_key, sep))
                else:
                    flattened[list_key] = item
        
        else:
            flattened[new_key] = value

    return flattened


nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

result = flatten_dict(nested_dict)
print(result)

from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates).
    :return: List of unique permutations.
    """
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])  
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            path.append(nums[i])

            backtrack(path, used)
            path.pop()
            used[i] = False

    nums.sort()
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result

nums = [1, 1, 2]
result = unique_permutations(nums)
print(result)



import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r"\b(\d{2}-\d{2}-\d{4})\b|\b(\d{2}/\d{2}/\d{4})\b|\b(\d{4}\.\d{2}\.\d{2})\b"

    matches = re.findall(pattern, text)

    dates = [date for match in matches for date in match if date]

    return dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
result = find_all_dates(text)
print(result)


import pandas as pd
import polyline
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on Earth (in meters).
    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.
    Returns:
        Distance between the two points in meters.
    """
    R = 6371000

    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])

    distances = [0] 

    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ["latitude", "longitude"]]
        lat2, lon2 = df.loc[i, ["latitude", "longitude"]]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)

    df["distance"] = distances

    return df

polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = polyline_to_dataframe(polyline_str)
print(df)



from typing import List

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in its row and column (excluding itself).
    
    Args:
        matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
        List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)

    rotated_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i][k] for k in range(n) if k != j)
            
            col_sum = sum(rotated_matrix[k][j] for k in range(n) if k != i)
            
            transformed_matrix[i][j] = row_sum + col_sum

    return transformed_matrix

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
print(result)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verifies the completeness of the data by checking if the timestamps for each
    (id, id_2) pair cover a full 24-hour period for all 7 days of the week.

    Args:
        df (pd.DataFrame): Input DataFrame with columns (id, id_2, startDay, startTime, endDay, endTime).

    Returns:
        pd.Series: A boolean series with multi-index (id, id_2) indicating if timestamps are incomplete.
    """
    weekday_to_date = {
        "Monday": "2023-10-16",
        "Tuesday": "2023-10-17",
        "Wednesday": "2023-10-18",
        "Thursday": "2023-10-19",
        "Friday": "2023-10-20",
        "Saturday": "2023-10-21",
        "Sunday": "2023-10-22",
    }

    df['startDate'] = df['startDay'].map(weekday_to_date)
    df['endDate'] = df['endDay'].map(weekday_to_date)

    df['startTimestamp'] = pd.to_datetime(df['startDate'] + ' ' + df['startTime'])
    df['endTimestamp'] = pd.to_datetime(df['endDate'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'], group_keys=False)

    def is_incomplete(group):
        days_covered = group['startTimestamp'].dt.dayofweek.unique()
        all_days_covered = len(days_covered) == 7 

        full_day_coverage = True
        for day in range(7):
            day_data = group[group['startTimestamp'].dt.dayofweek == day]
            if not day_data.empty:
                
                hours = set()
                for _, row in day_data.iterrows():
                    start_hour = row['startTimestamp'].hour
                    end_hour = row['endTimestamp'].hour
                    hours.update(range(start_hour, end_hour + 1))

                if len(hours) < 24:
                    full_day_coverage = False
                    break

        return not (all_days_covered and full_day_coverage)

    result = grouped.apply(is_incomplete)
    result.index.names = ['id', 'id_2']
    return result

data = {
    'id': [1, 1, 1, 2, 2],
    'id_2': [1, 1, 1, 2, 2],
    'startDay': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'startTime': ['00:00:00', '12:00:00', '00:00:00', '00:00:00', '12:00:00'],
    'endDay': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'endTime': ['23:59:59', '23:59:59', '23:59:59', '23:59:59', '23:59:59'],
}

df = pd.DataFrame(data)
result = time_check(df)
print(result)
