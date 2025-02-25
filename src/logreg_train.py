import numpy as np
import pandas as pd

from src.utils import normalize_features


def validate_data(X, y):
    """Validate input data"""
    if X is None or y is None:
        return False
    if len(X) == 0 or len(y) == 0:
        print("Error: Empty dataset")
        return False
    if np.isnan(X).any() or np.isnan(y).any():
        print("Error: Dataset contains NaN values")
        return False
    return True

def add_ones_column(X):
    """Add a column of ones to the feature matrix"""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def my_exp(x):
    """
    Calcule une approximation de e^x pour de petites valeurs de x
    """
    result = 1.0
    term = 1.0
    for i in range(1, 10):
        term *= x/i
        result += term
    return result

def ft_sigmoid(z):
    """
    Calcule la fonction sigmoïde pour des valeurs normalisées
    transforme notre prediction en probabilite entre 0 et 1
    """
    return 1.0 / (1.0 + my_exp(-z))

def ft_clip(value, min_val, max_val):
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


def ft_log(x):
    """
    Calcule le logarithme naturel de x en utilisant une série
    Valable pour x proche de 1 (ce qui est notre cas avec les probabilités)
    """
    # Utilisation de la série de Taylor
    if x <= 0:
        return float('inf')
    y = x - 1
    return y - (y*y)/2 + (y*y*y)/3


def compute_logistic_cost(y_true, y_pred, m):
    """
    Calcule le coût (l'erreur) de nos prédictions en régression logistique.

    y_true: Les vraies valeurs (0 ou 1 pour chaque étudiant)
    y_pred: Nos prédictions (probabilités entre 0 et 1)
    m: Nombre total d'étudiants
    """
    epsilon = 1e-15
    total_cost = 0

    for i in range(len(y_true)):
        # Si la prédiction est trop proche de 0 ou 1, on l'ajuste légèrement
        pred = ft_clip(y_pred[i], epsilon, 1 - epsilon)

        if y_true[i] == 1:
            # Si c'est vraiment un Gryffondor, on veut une prédiction proche de 1
            cost_i = -ft_log(pred)
        else:
            # Si ce n'est pas un Gryffondor, on veut une prédiction proche de 0
            cost_i = -ft_log(1 - pred)

        total_cost += cost_i

    return total_cost / m  # On fait la moyenne


def gradient_descent(X, y, n_iteration, learning_rate=0.01):
    """
    Perform gradient descent with matrix operations
    """
    m = X.shape[0]  # nombre d'exemples
    n = X.shape[1]  # nombre de features

    # S'assurer que theta est un numpy array
    theta = np.zeros((n, 1), dtype=np.float64)
    cost_history = np.zeros(n_iteration)

    for i in range(n_iteration):
        linear_combination = np.dot(X, theta)  # Utiliser np.dot explicitement
        probabilities = ft_sigmoid(linear_combination)
        gradients = (1 / m) * np.dot(X.T, (probabilities - y))
        theta = theta - learning_rate * gradients

        current_cost = compute_logistic_cost(y, probabilities, m)
        cost_history[i] = current_cost

    return theta, cost_history

def save_parameters(filename, theta):
    """save parameters theta in filename"""
    try:
        with open(filename, 'w') as f:
            f.write(f"{theta}\n")
        print(f"Training completed. Parameters saved to {filename}")
    except IOError as e:
        print(f"Error saving to file: {e}")
        raise


def calculate_model_accuracy(X, y_true, thetas, houses):
    """
    Calcule la précision du modèle en utilisant tous les classifieurs

    Args:
        X: Features normalisées (avec colonne de biais)
        y_true: Vraies maisons des étudiants
        thetas: Dictionnaire des paramètres pour chaque maison
        houses: Liste des maisons
    """
    n_samples = X.shape[0]
    predictions = []

    # Pour chaque étudiant
    for i in range(n_samples):
        student_features = X[i]
        probabilities = {}

        # Calculer la probabilité pour chaque maison
        for house in houses:
            theta = thetas[house]
            z = np.dot(student_features, theta)
            prob = ft_sigmoid(z)
            probabilities[house] = prob

        # Choisir la maison avec la plus haute probabilité
        predicted_house = max(probabilities, key=probabilities.get)
        predictions.append(predicted_house)

    # Calculer la précision
    correct = sum(1 for pred, true in zip(predictions, y_true) if pred == true)
    accuracy = correct / n_samples
    print(f"Précision globale: {accuracy * 100:.2f}%")

    # Afficher la précision par maison
    for house in houses:
        house_mask = y_true == house
        house_correct = sum(1 for pred, true in zip(predictions, y_true)
                            if pred == true and true == house)
        house_total = sum(house_mask)
        house_accuracy = house_correct / house_total
        print(f"Précision pour {house}: {house_accuracy * 100:.2f}%")


def train_model():
    dataset = pd.read_csv('../dataset/dataset_train.csv')
    features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                'Ancient Runes']
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

    clean_dataset = dataset.dropna(subset=features + ['Hogwarts House'])

    all_thetas = {}

    X_norm = normalize_features(clean_dataset, features)
    X_norm = add_ones_column(X_norm)

    for house in houses:
        print(f"\nTraining model for {house}...")
        y = (clean_dataset['Hogwarts House'] == house).astype(
            int).to_numpy().reshape(-1, 1)

        theta, cost_history = gradient_descent(X_norm, y, 500)
        all_thetas[house] = theta

        save_parameters(f"model_params_{house}.txt", theta)

    # Évaluer le modèle
    print("\nÉvaluation du modèle:")
    calculate_model_accuracy(X_norm, clean_dataset['Hogwarts House'].values,
                             all_thetas, houses)

if __name__ == "__main__":
    train_model()
