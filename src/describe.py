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

            def percentile(p: float) -> float:
                index = (count - 1) * (p / 100)
                lower = int(index)
                upper = lower + 1
                if upper >= count:
                    return values[-1]
                return values[lower] + (values[upper] - values[lower]) * (index - lower)

            stats[feature] = {
                'Count': count,
                'Mean': mean,
                'Std': std,
                'Min': min_val,
                '25%': percentile(25),
                '50%': percentile(50),
                '75%': percentile(75),
                'Max': max_val,
            }

    return stats

def save_stats_to_csv(students_stats: Dict[str, Dict[str, float]], features: List[str], output_file: str):
    col_width = max(len(feature) for feature in features) + 2

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = [" "] + features
        writer.writerow(header)

        for key in stats_keys:
            row = [key] + [f"{students_stats[feature][key]:.2f}" for feature in features]
            writer.writerow(row)

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)
    file_name = sys.argv[1]
    students = load_students_from_csv(file_path=file_name)
    students_stats = calculate_statistics(students=students, features=magical_courses)
    save_stats_to_csv(students_stats=students_stats, features=magical_courses, output_file='students_stats.csv')

if __name__ == "__main__":
    describe()
