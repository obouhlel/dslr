import numpy as np
import pandas as pd
import os

from utils import normalize_features, ft_mean, ft_max_dict, add_ones_column, ft_sigmoid
from typing import List, Dict

def predict_house(X: np.ndarray, all_thetas: Dict[str, np.ndarray], houses: List[str]) -> List[str]:
    """
    Prédit la maison pour chaque étudiant.
    X: Données normalisées avec biais.
    thetas: Dictionnaire des paramètres theta pour chaque maison.
    houses: Liste des maisons.
    """
    predictions = []
    for i in range(X.shape[0]):
        student_features = X[i]
        probabilities = {}
        for house in houses:
            theta = all_thetas[house]
            z = np.dot(student_features, theta)
            prob = ft_sigmoid(z)
            probabilities[house] = prob
        predicted_house = ft_max_dict(probabilities)
        predictions.append(predicted_house)
    return predictions

def clean_nan_by_mean(dataset: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Remplace les NaN par la moyenne de la colonne pour les features données."""
    dataset_copy = dataset.copy()
    for feature in features:
        mean_value = ft_mean(dataset_copy[feature].dropna().values)
        dataset_copy[feature] = dataset_copy[feature].fillna(mean_value)
    return dataset_copy

def predict():
    dataset = pd.read_csv('../dataset/dataset_test.csv')
    features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                'Ancient Runes']
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

    X_norm = normalize_features(dataset, features)
    X_norm = add_ones_column(X_norm)

    all_thetas = {}
    for house in houses:
        params = f'../results/model_params_{house}.txt'
        if os.path.exists(params):
            all_thetas[house] = np.loadtxt(params)
        else:
            raise FileNotFoundError(f"Model parameters file for {house} not found at {params}")

    predictions = predict_house(X_norm, all_thetas, houses)
    
    results = pd.DataFrame({
        'Index': dataset['Index'],
        'Hogwarts House': predictions
    })
    results.to_csv('../results/houses.csv', index=False)
    print("Prédictions sauvegardées dans 'results/houses.csv'.")

if __name__ == "__main__":
    predict()