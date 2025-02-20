from pprint import pprint

import matplotlib.pyplot as plt
import csv
import pandas as pd

from src.utils import ft_mean, ft_std, normalize_dataset, ft_max, ft_min

def plot_comparative_histogram(original_dataset,
                               course1, course2):
    houses = original_dataset['Hogwarts House'].unique()
    colors = ['red', 'blue', 'yellow', 'green']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for i, house in enumerate(houses):
        house_data = \
        original_dataset[original_dataset['Hogwarts House'] == house][
            course1].dropna()
        ax1.hist(house_data, bins=20, alpha=0.5, label=house,
                 color=colors[i])

    ax1.set_title(f'Distribution la plus homogène - {course1}',
                  fontsize=16)
    ax1.set_xlabel('Valeur', fontsize=14)
    ax1.set_ylabel('Nombre d\'élèves', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    for i, house in enumerate(houses):
        house_data = \
        original_dataset[original_dataset['Hogwarts House'] == house][
            course2].dropna()
        ax2.hist(house_data, bins=20, alpha=0.5, label=house,
                 color=colors[i])

    ax2.set_title(f'Distribution la moins homogène - {course2}',
                  fontsize=16)
    ax2.set_xlabel('Valeur', fontsize=14)
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
        mean_range = ft_max(means) - ft_min(means)
        homogeneity_results[course] = {
            'stats': stats,
            'mean_range': mean_range
        }
    sorted_courses = sorted(homogeneity_results.items(),
                            key=lambda x: x[1]['mean_range'])
    sorted_course_name = [course for course, data in sorted_courses]

    print("Cours classés du plus homogène au moins homogène:")
    for course, data in sorted_courses:
        print(f"{course}: écart entre moyennes = {data['mean_range']:.2f}")
        for house in houses:
            print(
                f"  {house}: moyenne = {data['stats'][house]['mean']:.2f}, "
                f"écart-type = {data['stats'][house]['std']:.2f}")
        print()

    return sorted_course_name


def show_histogram():
    dataset = pd.read_csv('../dataset/dataset_train.csv')
    numeric_col = dataset.select_dtypes(include=['float64', 'int64']
                                           ).columns
    course_col = [num for num in numeric_col if num != 'Index']
    normalized_dataset = normalize_dataset(dataset, course_col)
    sorted_courses = calculate_homogeneity(normalized_dataset, course_col)

    plot_comparative_histogram(dataset, sorted_courses[
        0], sorted_courses[-1])

if __name__ == "__main__":
    show_histogram()
