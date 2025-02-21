import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from src.utils import ft_mean, ft_std, normalize_dataset, ft_max, ft_min


def plot_histogram(ax: plt.Axes, dataset: pd.DataFrame, feature: str,
                   houses: List[str]) -> None:
    """
    Plot a histogram for each house on the given axis.
    """
    colors = {'Gryffindor': 'red', 'Hufflepuff': 'gold',
              'Ravenclaw': 'blue', 'Slytherin': 'green'}

    for house in houses:
        house_data = dataset[dataset['Hogwarts House'] == house][feature]
        ax.hist(house_data, bins=20, density=True, alpha=0.5,
                color=colors[house], label=house)

    ax.set_title(feature, fontsize=8)
    ax.tick_params(axis='both', labelsize=6)


def plot_comparative_histogram(dataset: pd.DataFrame, course1: str,
                               course2: str) -> None:
    """
    Compare distributions of two courses using histograms.
    """
    houses = dataset['Hogwarts House'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    features = [course1, course2]

    for ax, feature in zip(axes, features):
        plot_histogram(ax, dataset, feature, houses)
        ax.set_title(f'Distribution - {feature}', fontsize=16)
        ax.set_xlabel('Value', fontsize=14)
        ax.set_ylabel('Number of students', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_homogeneity(dataset: pd.DataFrame, course_col: List[str]) -> \
        List[str]:
    """
    Calculate and display distribution homogeneity for each course.
    """
    homogeneity_results: Dict[str, Dict[str, Any]] = {}
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
    sorted_course_names = [course for course, _ in sorted_courses]

    print("Courses ranked from most to least homogeneous:")
    for course, data in sorted_courses:
        print(f"{course}: mean range = {data['mean_range']:.2f}")
        for house in houses:
            print(f"  {house}: mean = {data['stats'][house]['mean']:.2f}, "
                  f"std = {data['stats'][house]['std']:.2f}")
        print()

    return sorted_course_names


def show_histogram() -> None:
    """
    Main function: load data and display histograms comparing
    most and least homogeneous course distributions.
    """
    dataset = pd.read_csv('../dataset/dataset_train.csv')
    numeric_col = dataset.select_dtypes(include=['float64', 'int64']).columns
    course_col = [num for num in numeric_col if num != 'Index']
    normalized_dataset = normalize_dataset(dataset, course_col)
    sorted_courses = calculate_homogeneity(normalized_dataset, course_col)
    plot_comparative_histogram(dataset, sorted_courses[0], sorted_courses[-1])


if __name__ == "__main__":
    show_histogram()
