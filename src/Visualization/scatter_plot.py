from src.utils import correlation, normalize_dataset, ft_abs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def get_correlation_matrix(dataset: pd.DataFrame,
                           course_col: List[str]) -> np.ndarray:
    """
    Calculate correlation matrix between all pairs of courses.
    Returns:
        A symmetric matrix containing correlation coefficients
    """
    normalized_dataset = normalize_dataset(dataset, course_col)
    n = len(course_col)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif j > i:
                corr_value = correlation(normalized_dataset[course_col[i]],
                                         normalized_dataset[course_col[j]])
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value

    return corr_matrix


def get_feature_min_max_corr(
        corr_matrix: np.ndarray, course_col: List[str]) -> Tuple[str, str, str, str]:
    """
    Find pairs of features with highest and lowest absolute correlation.
    """
    max_corr = -1
    min_corr = 1.0
    feature1 = feature2 = feature1_min = feature2_min = None
    n = len(course_col)

    for i in range(n):
        for j in range(i + 1, n):
            curr_corr = ft_abs(corr_matrix[i, j])
            if curr_corr > max_corr:
                max_corr = curr_corr
                feature1 = course_col[i]
                feature2 = course_col[j]
            if curr_corr < min_corr:
                min_corr = curr_corr
                feature1_min = course_col[i]
                feature2_min = course_col[j]

    return feature1, feature2, feature1_min, feature2_min


def plot_scatter(ax: plt.Axes, data: pd.DataFrame, feature_x: str,
                 feature_y: str, houses: List[str]) -> None:
    """
    Plot scatter points on given axis, colored by house.
    """
    colors = {'Gryffindor': 'red', 'Hufflepuff': 'gold',
              'Ravenclaw': 'blue', 'Slytherin': 'green'}

    for house in houses:
        house_data = data[data['Hogwarts House'] == house]
        ax.scatter(house_data[feature_x], house_data[feature_y],
                   alpha=0.5, s=5, color=colors[house], label=house)
    ax.tick_params(axis='both', labelsize=6)


def compare_scatter(dataset: pd.DataFrame, feature1: str, feature2: str,
                    feature1_min: str, feature2_min: str,
                    houses: List[str]) -> None:
    """
    Create comparative scatter plots of most and least correlated features.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    plot_scatter(ax1, dataset, feature1, feature2, houses)
    ax1.set_xlabel(feature1, fontsize=10)
    ax1.set_ylabel(feature2, fontsize=10)
    ax1.set_title(f'High correlation: {feature1} vs {feature2}', fontsize=12)

    plot_scatter(ax2, dataset, feature1_min, feature2_min, houses)
    ax2.set_xlabel(feature1_min, fontsize=10)
    ax2.set_ylabel(feature2_min, fontsize=10)
    ax2.set_title(f'Low correlation: {feature1_min} vs {feature2_min}',
                  fontsize=12)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function: load data, calculate correlations and display comparative plots.
    """
    try:
        dataset = pd.read_csv("../../dataset/dataset_train.csv")
    except BaseException:
        print("Error reading the file")
        return

    numeric_col = dataset.select_dtypes(include=['float64', 'int64']).columns
    course_col = [num for num in numeric_col if num != 'Index']

    corr_matrix = get_correlation_matrix(dataset, course_col)
    features = get_feature_min_max_corr(corr_matrix, course_col)
    houses = dataset["Hogwarts House"].unique()

    compare_scatter(dataset, *features, houses)


if __name__ == "__main__":
    main()
