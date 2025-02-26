import numpy as np
import pandas as pd
import os
from logreg_train import ft_clip, my_exp, add_ones_column
from utils import normalize_features, ft_mean
from typing import List, Dict

# Fonctions nécessaires
def ft_sigmoid(z: float) -> float:
    """Fonction sigmoïde."""
    z = ft_clip(z, -500, 500)
    return 1.0 / (1.0 + my_exp(-z))

def ft_max_dict(d: Dict[str, float]) -> str:
    """Retourne la clé avec la valeur maximale dans le dictionnaire."""
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key

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
    for feature in features:
        mean_value = ft_mean(dataset[feature].dropna().values)
        dataset[feature].fillna(mean_value, inplace=True)
    return dataset

def predict():
    dataset = pd.read_csv('../dataset/dataset_test.csv')
    features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                'Ancient Runes']
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    clean_dataset = clean_nan_by_mean(dataset, features)
    X_norm = normalize_features(clean_dataset, features)
    X_norm = add_ones_column(X_norm)

    all_thetas = {}
    for house in houses:
        model_path = f'../results/model_params_{house}.txt'
        if os.path.exists(model_path):
            all_thetas[house] = np.loadtxt(model_path)
        else:
            raise FileNotFoundError(f"Model parameters file for {house} not found at {model_path}")

    predictions = predict_house(X_norm, all_thetas, houses)
    
    results = pd.DataFrame({
        'Index': dataset['Index'],
        'Hogwarts House': predictions
    })
    results.to_csv('../results/houses.csv', index=False)
    print("Prédictions sauvegardées dans 'houses.csv'.")

if __name__ == "__main__":
    predict()