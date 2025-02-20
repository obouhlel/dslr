from utils import correlation, normalize_dataset, ft_abs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot():
    dataset = pd.read_csv('dataset/dataset_train.csv')

    numeric_col = dataset.select_dtypes(include=['float64', 'int64']).columns
    course_col = [num for num in numeric_col if num != 'Index']

    normalized_dataset = normalize_dataset(dataset, course_col)

    n = len(course_col)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif j > i:
                corr_value = correlation(normalized_dataset[course_col[i]], normalized_dataset[course_col[j]])
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value

    max_corr = -1
    feature1 = None
    feature2 = None

    for i in range(n):
        for j in range(i + 1, n):
            if ft_abs(corr_matrix[i, j]) > max_corr:
                max_corr = abs(corr_matrix[i, j])
                feature1 = course_col[i]
                feature2 = course_col[j]

    plt.figure(figsize=(10, 6))
    plt.scatter(dataset[feature1], dataset[feature2], alpha=0.5)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Scatter plot entre {feature1} et {feature2}')
    plt.show()

if __name__ == "__main__":
    scatter_plot()
