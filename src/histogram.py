from pprint import pprint

import matplotlib.pyplot as plt
import csv
import pandas as pd

from src.utils import ft_mean, ft_std, normalize_dataset, ft_max, ft_min


def plot_single_course_histogram(dataset, course_name):
    houses = dataset['Hogwarts House'].unique()
    colors = ['red', 'blue', 'yellow', 'green']
    plt.figure(figsize=(12, 8))

    for i, house in enumerate(houses):
        house_data = dataset[dataset['Hogwarts House'] == house][course_name].dropna()
        plt.hist(house_data, bins=20, alpha=0.5, label=house, color=colors[i])

    plt.title(f'Distribution des notes en {course_name} par maison', fontsize=16)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Nombre d\'élèves', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_comparative_histogram(original_dataset, normalized_dataset,
                               course_name):
    houses = original_dataset['Hogwarts House'].unique()
    colors = ['red', 'blue', 'yellow', 'green']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for i, house in enumerate(houses):
        house_data = \
        original_dataset[original_dataset['Hogwarts House'] == house][
            course_name].dropna()
        ax1.hist(house_data, bins=20, alpha=0.5, label=house,
                 color=colors[i])

    ax1.set_title(f'Distribution originale - {course_name}', fontsize=16)
    ax1.set_xlabel('Valeur originale', fontsize=14)
    ax1.set_ylabel('Nombre d\'élèves', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    for i, house in enumerate(houses):
        house_data = \
        normalized_dataset[normalized_dataset['Hogwarts House'] == house][
            course_name].dropna()
        ax2.hist(house_data, bins=20, alpha=0.5, label=house,
                 color=colors[i])

    ax2.set_title(f'Distribution normalisée - {course_name}', fontsize=16)
    ax2.set_xlabel('Valeur normalisée (z-score)', fontsize=14)
    ax2.set_ylabel('Nombre d\'élèves', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def calculate_homogeneity(dataset, course_col):
    homogeneity_results = {}
    houses = dataset['Hogwarts House'].unique()
    for course in course_col:
        stats = {}
        for house in houses:
            house_data = dataset[dataset['Hogwarts House'] == house][
                course].dropna()
            stats[house] = {
                'mean': ft_mean(house_data),
                'std': ft_std(house_data)
            }
        means = [stats[house]['mean'] for house in houses]
        mean_range = ft_max(means) - ft_min(means) # coder son propre maxx et
        # min
        homogeneity_results[course] = {
            'stats': stats,
            'mean_range': mean_range
        }
    sorted_courses = sorted(homogeneity_results.items(),
                            key=lambda x: x[1]['mean_range']) #verifier si
    # sorted est autoriser

    print("Cours classés du plus homogène au moins homogène:")
    for course, data in sorted_courses:
        print(f"{course}: écart entre moyennes = {data['mean_range']:.2f}")
        for house in houses:
            print(
                f"  {house}: moyenne = {data['stats'][house]['mean']:.2f}, écart-type = {data['stats'][house]['std']:.2f}")
        print()


def show_histogram():
    dataset = pd.read_csv('../dataset/dataset_train.csv')
    numeric_col = dataset.select_dtypes(include=['float64', 'int64']
                                           ).columns
    course_col = [num for num in numeric_col if num != 'Index']
    normalized_dataset = normalize_dataset(dataset, course_col)
    calculate_homogeneity(normalized_dataset, course_col)
    for course in course_col:
        plot_comparative_histogram(dataset, normalized_dataset, course)

if __name__ == "__main__":
    show_histogram()