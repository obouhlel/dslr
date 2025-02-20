import sys
import csv
from typing import List, Dict
from utils import load_students_from_csv, magical_courses

stats_keys = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

def calculate_statistics(students: List[Dict[str, any]], features: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical measures for given features in a list of student dictionaries.

    Args:
        students (List[Dict[str, any]]): A list of dictionaries where each dictionary represents a student and contains feature values.
        features (List[str]): A list of feature names to calculate statistics for.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a feature name and the value is another dictionary containing:
            - 'Count': The number of non-null values for the feature.
            - 'Mean': The mean (average) of the feature values.
            - 'Std': The standard deviation of the feature values.
            - 'Min': The minimum value of the feature.
            - '25%': The 25th percentile value of the feature.
            - '50%': The 50th percentile (median) value of the feature.
            - '75%': The 75th percentile value of the feature.
            - 'Max': The maximum value of the feature.
    """
    stats = {}

    for feature in features:
        values = [student[feature] for student in students if student[feature] is not None]
        if values:
            values.sort()

            count = len(values)
            mean = sum(values) / count

            variance = sum((x - mean) ** 2 for x in values) / count
            std = variance ** 0.5

            min_val = values[0]
            max_val = values[-1]

            stats[feature] = {
                'Count': count,
                'Mean': mean,
                'Std': std,
                'Min': min_val,
                '25%': percentile(25, count, values),
                '50%': percentile(50, count, values),
                '75%': percentile(75, count, values),
                'Max': max_val,
            }

    return stats

def percentile(p: float, count: float, values: List) -> float:
    index = (count - 1) * (p / 100)
    lower = int(index)
    upper = lower + 1
    if upper >= count:
        return values[-1]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)

def save_stats_to_csv(students_stats: Dict[str, Dict[str, float]], features: List[str], output_file: str):

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = [" "] + features
        writer.writerow(header)

        for key in stats_keys:
            row = [key] + [f"{students_stats[feature][key]:.2f}" for feature in features]
            writer.writerow(row)


def print_stats_table(students_stats: Dict[str, Dict[str, float]],
                      features: List[str]):
    col_width = 18
    first_col_width = 10

    print(" " * first_col_width, end="")
    for feature in features:
        print(f" | {feature[:col_width - 2]:^{col_width - 2}}", end="")
    print(" |")

    print("-" * first_col_width, end="")
    for _ in features:
        print("-+-" + "-" * (col_width - 2), end="")
    print("-|")

    stats_keys = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    for key in stats_keys:
        print(f"{key:<{first_col_width}}", end="")
        for feature in features:
            value = students_stats[feature].get(key, 0)
            if key == 'Count':
                formatted_value = f"{value:.0f}"
            else:
                formatted_value = f"{value:.6f}"
            print(f" | {formatted_value:>{col_width - 2}}", end="")
        print(" |")

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)
    file_name = sys.argv[1]
    students = load_students_from_csv(file_path=file_name)
    features = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration',
        'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
    ]
    students_stats = calculate_statistics(students, features)
    save_stats_to_csv(students_stats, features, output_file='students_stats.csv')
    print_stats_table(students_stats, features)


if __name__ == "__main__":
    describe()
