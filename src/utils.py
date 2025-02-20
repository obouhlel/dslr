import csv
from typing import List, Dict

import pandas as pd

def is_nan(value):
    """
    Vérifie si une valeur est NaN (Not a Number)
    """
    return value != value


def ft_mean(series):
    """
    Calcule la moyenne d'une série pandas
    """
    total = 0
    count = 0
    for value in series:
        if not is_nan(value):
            total += value
            count += 1

    return total / count if count > 0 else float('nan')


def ft_std(series):
    """
    Calcule l'écart-type d'une série pandas
    """
    if len(series) <= 1:
        return float('nan')

    mean = ft_mean(series)

    sum_squared_diff = 0
    count = 0
    for value in series:
        if not is_nan(value):
            sum_squared_diff += (value - mean) ** 2
            count += 1

    variance = sum_squared_diff / count if count > 0 else float('nan')
    return variance ** 0.5


def ft_min(series):
    """
    Trouve la valeur minimale d'une série pandas
    """
    min_value = float('inf')
    found_valid = False

    for value in series:
        if not is_nan(value):
            min_value = value if min_value > value or not found_valid else min_value
            found_valid = True

    return min_value if found_valid else float('nan')


def ft_max(series):
    """
    Trouve la valeur maximale d'une série pandas
    """
    max_value = float('-inf')
    found_valid = False

    for value in series:
        if not is_nan(value):
            max_value = value if max_value < value or not found_valid else max_value
            found_valid = True

    return max_value if found_valid else float('nan')


def normalize_dataset(dataset, features):
    """
    Normalise les caractéristiques du dataset en utilisant le z-score
    """
    normalized = dataset.copy()

    for feature in features:
        feature_data = []
        for value in dataset[feature]:
            if not is_nan(value):
                feature_data.append(value)

        mean = ft_mean(feature_data)
        std = ft_std(feature_data)

        if std == 0:
            normalized[feature] = 0
        else:
            normalized_values = []
            for value in normalized[feature]:
                if not is_nan(value):
                    normalized_values.append((value - mean) / std)
                else:
                    normalized_values.append(float('nan'))

            normalized[feature] = normalized_values

    return normalized


# Read the dataset_train.csv and return a disctinary of datas
def load_students_from_csv(file_path: str) -> List[Dict[str, any]]:
    students = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                    row[key] = float(row[key]) if row[key] else None
            students.append(row)
    return students