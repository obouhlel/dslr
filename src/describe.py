import sys
import csv
import numpy as np
from typing import List
from object.stats import Stats
from object.student import Student

def load_students_from_csv(file_path) -> List[Student]:
    students = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                    row[key] = float(row[key]) if row[key] else None

            student = Student(
                index=row['Index'],
                house=row['Hogwarts House'],
                first_name=row['First Name'],
                last_name=row['Last Name'],
                birthday=row['Birthday'],
                best_hand=row['Best Hand'],
                arithmancy=row['Arithmancy'],
                astronomy=row['Astronomy'],
                herbology=row['Herbology'],
                defense=row['Defense Against the Dark Arts'],
                divination=row['Divination'],
                muggle_studies=row['Muggle Studies'],
                ancient_runes=row['Ancient Runes'],
                history_of_magic=row['History of Magic'],
                transfiguration=row['Transfiguration'],
                potions=row['Potions'],
                care_of_magical_creatures=row['Care of Magical Creatures'],
                charms=row['Charms'],
                flying=row['Flying']
            )
            students.append(student)
    return students

def calculate_statistics(students: List[Student]) -> dict:
    fields = [
        'arithmancy', 'astronomy', 'herbology', 'defense', 'divination',
        'muggle_studies', 'ancient_runes', 'history_of_magic', 'transfiguration',
        'potions', 'care_of_magical_creatures', 'charms', 'flying'
    ]

    stats = {}
    for field in fields:
        values = [getattr(student, field) for student in students if getattr(student, field) is not None]
        if values:
            stats[field] = {
                'Count': len(values),
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                '25%': np.percentile(values, 25),
                '50%': np.percentile(values, 50),
                '75%': np.percentile(values, 75),
                'Max': np.max(values),
            }
    return stats

def save_statistics_to_csv(stats: dict, file_path: str):
    fields_top = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration',
        'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
    ]
    fields_left = [
        'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'
    ]

    data_to_write = []
    for field in fields_top:
        field_key = field.lower().replace(' ', '_')
        if field_key in stats:
            row = [field] + [stats[field_key][stat] for stat in fields_left]
            data_to_write.append(row)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Field'] + fields_left)
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
