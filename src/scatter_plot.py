from utils import correlation, normalize_dataset, ft_abs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_correlation(dataset, course_col):
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
    min_corr = 0.0
    feature1 = None
    feature2 = None
    feature1_min = None
    feature2_min = None

    for i in range(n):
        for j in range(i + 1, n):
            if ft_abs(corr_matrix[i, j]) > max_corr:
                max_corr = ft_abs(corr_matrix[i, j])
                feature1 = course_col[i]
                feature2 = course_col[j]
            if ft_abs(corr_matrix[i, j]) < min_corr:
                min_corr = ft_abs(corr_matrix[i, j])
                feature1_min = course_col[i]
                feature2_min = course_col[j]

    return (feature1, feature2, feature1_min, feature2_min)

def scatter_plot():
    dataset = pd.read_csv('dataset/dataset_train.csv')

    numeric_col = dataset.select_dtypes(include=['float64', 'int64']).columns
    course_col = [num for num in numeric_col if num != 'Index']

    print(course_col)

    (feature1, feature2, feature1_min, feature2_min) = get_correlation(dataset, course_col)

    houses = dataset["Hogwarts House"].unique()
    colors = ['red', 'blue', 'yellow', 'green']

    plt.figure(figsize=(10, 6))
    for k, house in enumerate(houses):
        house_data = dataset[dataset["Hogwarts House"] == house]
        x = house_data[feature1]
        y = house_data[feature2]
        plt.scatter(x, y, c=colors[k], alpha=0.5)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Scatter plot entre {feature1} et {feature2}')
    plt.show()

if __name__ == "__main__":
    scatter_plot()
