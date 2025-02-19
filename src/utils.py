import csv
from typing import List, Dict

magical_courses = [
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
    'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration',
    'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
]

hogwarts_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

# Read the dataset_train.csv and return a disctinary of datas
def load_students_from_csv(file_path: str) -> List[Dict[str, any]]:
    students = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                    row[key] = float(row[key]) if row[key] else 0.0
            students.append(row)
    return students