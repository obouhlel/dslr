import csv
from typing import List, Dict, Any

import numpy as np
import pandas as pd

def is_nan(value: Any) -> bool:
    """
    Check if a value is NaN (Not a Number).
    """
    return value != value

def ft_mean(series: List[float]) -> float:
    """
    Calculate mean of a numeric series, ignoring NaN values.
    """
    total = 0
    count = 0
    for value in series:
        if not is_nan(value):
            total += value
            count += 1
    return total / count if count > 0 else float('nan')


def ft_std(series: List[float]) -> float:
    """
    Calculate standard deviation of a numeric series, ignoring NaN values.
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


def ft_min(series: List[float]) -> float:
    """
    Find minimum value in a numeric series, ignoring NaN values.
    """
    min_value = float('inf')
    found_valid = False

    for value in series:
        if not is_nan(value):
            min_value = min(min_value, value) if found_valid else value
            found_valid = True

    return min_value if found_valid else float('nan')


def ft_abs(n: float) -> float:
    """
    Calculate absolute value of a number.
    """
    return -n if n < 0 else n


def ft_max_dict(d: Dict[str, float]) -> str:
    """Retourne la clé avec la valeur maximale dans le dictionnaire."""
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key

def ft_max(series: List[float]) -> float:
    """
    Find maximum value in a numeric series, ignoring NaN values.
    """
    max_value = float('-inf')
    found_valid = False

    for value in series:
        if not is_nan(value):
            max_value = max(max_value, value) if found_valid else value
            found_valid = True

    return max_value if found_valid else float('nan')

def add_ones_column(X: np.ndarray) -> np.ndarray:
    """Add a column of ones to the feature matrix"""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def my_exp(x: float) -> float:
    """
    Calcule une approximation de e^x pour de petites valeurs de x
    """
    result = 1.0
    term = 1.0
    for i in range(1, 10):
        term *= x/i
        result += term
    return result

def ft_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Calcule la fonction sigmoïde pour des valeurs normalisées
    transforme notre prediction en probabilite entre 0 et 1
    """
    return 1.0 / (1.0 + my_exp(-z))

def ft_clip(value: float, min_val: float, max_val: float) -> float:
    """
    Restreint une valeur à l'intervalle [min_val, max_val]
    Si la valeur est plus petite que min_val, retourne min_val
    Si la valeur est plus grande que max_val, retourne max_val
    Sinon retourne la valeur comme elle est
    """
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def ft_log(x: float) -> float:
    """
    Calcule le logarithme naturel de x en utilisant une série
    Valable pour x proche de 1 (ce qui est notre cas avec les probabilités)
    """
    # Utilisation de la série de Taylor
    if x <= 0:
        return float('inf')
    y = x - 1
    return y - (y*y)/2 + (y*y*y)/3

def normalize_dataset(dataset: pd.DataFrame,
                      features: List[str]) -> pd.DataFrame:
    """
    Normalize dataset features using z-score standardization.
    Uses the z-score method: (x - mean) / std_deviation

    Example:
        Astronomy scores [100, 200, 300] become [-1.22, 0, 1.22]
        Defense scores [2, 4, 6] also become [-1.22, 0, 1.22]
    """
    normalized = dataset.copy()

    for feature in features:
        feature_data = [value for value in dataset[feature] if
                        not is_nan(value)]
        mean = ft_mean(feature_data)
        std = ft_std(feature_data)

        if std == 0:
            normalized[feature] = 0
        else:
            normalized[feature] = [
                (value - mean) / std if not is_nan(value) else float('nan')
                for value in normalized[feature]
            ]

    return normalized

def normalize_features(dataset: pd.DataFrame, features: List[str]) -> np.ndarray:
    """
    Normalise uniquement les features sélectionnées en utilisant le z-score.
    Remplace les NaN par la moyenne de la feature après normalisation.
    Retourne un numpy array des features normalisées.
    """
    normalized_features = []

    for feature in features:
        feature_data = dataset[feature].values

        mean = ft_mean(feature_data)
        std = ft_std(feature_data)

        if std == 0:
            normalized_feature = np.zeros_like(feature_data)
        else:
            normalized_feature = (feature_data - mean) / std

        normalized_feature[is_nan(normalized_feature)] = 0
        normalized_features.append(normalized_feature)

    return np.array(normalized_features).T

def covariance(x: List[float], y: List[float]) -> float:
    """
    Calculate covariance between two numeric series.
    Measures how two features change together.

    explanations:
    - Positive: When x goes up, y tends to go up too
    - Negative: When x goes up, y tends to go down
    - Zero: x and y don't seem to follow each other

    Example:
        - High covariance: Height vs Weight (tall → heavy)
        - Negative covariance: Study hours vs Gaming hours
        - Zero covariance: Shoe size vs Programming skill
    """
    x_len = len(x)
    y_len = len(y)
    if x_len < y_len:
        n = x_len
    else:
        n = y_len

    x_mean = ft_mean(x)
    y_mean = ft_mean(y)
    cov = 0.0

    for i in range(n):
        if not (is_nan(x[i]) or is_nan(y[i])):
            cov += (x[i] - x_mean) * (y[i] - y_mean)

    return cov / n


def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two numeric series.
    Like covariance, but scaled between -1 and 1 for easier interpretation.

    The result means:
    - 1: Perfect positive relationship (like y = x)
    - 0: No relationship (like random dots)
    - -1: Perfect negative relationship (like y = -x)
    """
    x_std = ft_std(x)
    y_std = ft_std(y)
    return covariance(x, y) / (x_std * y_std)


def load_students_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load student data from CSV file.
    """
    students = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in ['Index', 'Hogwarts House', 'First Name',
                               'Last Name', 'Birthday', 'Best Hand']:
                    row[key] = float(row[key]) if row[key] else None
            students.append(row)
    return students
