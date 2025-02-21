import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from histogram import plot_histogram
from scatter_plot import plot_scatter


def create_pair_plot(data: pd.DataFrame, features: List[str]) -> None:
    """
    Creates a complete pair plot with histograms on diagonal and scatter plots

    The pair plot shows the relationships between all pairs of features,
    with histograms on the diagonal showing each feature's distribution.
    """
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    houses = data['Hogwarts House'].unique()

    for i in range(n):
        for j in range(n):
            if i == j:
                plot_histogram(axes[i, j], data, features[i], houses)
            else:
                plot_scatter(axes[i, j], data, features[j],
                             features[i], houses)

            if i == n - 1:
                axes[i, j].set_xlabel(features[j], fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(features[i], fontsize=8)

    plt.tight_layout()


def main() -> None:
    """
    Main function: loads data and creates a pair plot visualization.
    Limits the number of features to 4 for better visibility.
    """
    try:
        data = pd.read_csv("../dataset/dataset_train.csv")
    except BaseException:
        print("Error reading the file")
        return

    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [num for num in numeric_features if num != 'Index']

    if len(numeric_features) > 4:
        numeric_features = numeric_features[:4]

    create_pair_plot(data, numeric_features)
    plt.suptitle("Pair Plot - Feature Relationships Matrix", y=1.02)
    plt.show()


if __name__ == "__main__":
    main()
