import sys
import csv
from typing import List, Dict

def load_students_from_csv(file_path) -> List[Dict[str, any]]:
    students = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                    row[key] = float(row[key]) if row[key] else None
            students.append(row)
    return students

def calculate_statistics(students: List[Dict[str, any]]) -> Dict[str, Dict[str, float]]:
    fields = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration',
        'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
    ]

    stats = {}

    for field in fields:
        values = [student[field] for student in students if student[field] is not None]
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

            stats[field] = {
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

def save_statistics_to_csv(stats: Dict[str, Dict[str, float]], file_path: str):
    fields_top = list(stats.keys())
    fields_left = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    # Préparer les données pour l'écriture dans le CSV
    data_to_write = []
    for field in fields_top:
        row = [field] + [stats[field][stat] for stat in fields_left]
        data_to_write.append(row)

    # Écrire les données dans un fichier CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Écrire l'en-tête
        writer.writerow(['Field'] + fields_left)
        # Écrire les données
        writer.writerows(data_to_write)

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)
    file_name = sys.argv[1]
    students = load_students_from_csv(file_path=file_name)
    students_stats = calculate_statistics(students=students)
    save_statistics_to_csv(students_stats, "describe.csv")

if __name__ == "__main__":
    describe()
