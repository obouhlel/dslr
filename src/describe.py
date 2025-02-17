import sys
import csv
from typing import List
from object.Student import Student

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

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)
    file_name = sys.argv[1]
    students = load_students_from_csv(file_path=file_name)

if __name__ == "__main__":
    describe()