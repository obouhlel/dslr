import matplotlib.pyplot as plt
import csv
import numpy as np
from describe import calculate_statistics, stats_keys
from utils import load_students_from_csv, hogwarts_houses, magical_courses

def histogram():
    students = load_students_from_csv('dataset/dataset_train.csv')

    # Get split houses
    ravenclaw_studs = [student for student in students if student['Hogwarts House'] == 'Ravenclaw']
    gryffindor_studs = [student for student in students if student['Hogwarts House'] == 'Gryffindor']
    hufflepuff_studs = [student for student in students if student['Hogwarts House'] == 'Hufflepuff']
    slytherin_studs = [student for student in students if student['Hogwarts House'] == 'Slytherin']

    # Calculate stats for each houses
    ravenclaw_stats = calculate_statistics(ravenclaw_studs, magical_courses)
    gryffindor_stats = calculate_statistics(gryffindor_studs, magical_courses)
    hufflepuff_stats = calculate_statistics(hufflepuff_studs, magical_courses)
    slytherin_stats = calculate_statistics(slytherin_studs, magical_courses)

    # Plot an histogram for each course with all means
    # for course in magical_courses:
    #     for key in stats_keys:
    course = magical_courses[1]
    key = stats_keys[-1]
    ravenclaw_value = ravenclaw_stats[course][key]
    gryffindor_value = gryffindor_stats[course][key]
    hufflepuff_value = hufflepuff_stats[course][key]
    slytherin_value = slytherin_stats[course][key]
    values = [ravenclaw_value, gryffindor_value, hufflepuff_value, slytherin_value]
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.bar(hogwarts_houses, values, color=['blue', 'red', 'orange', 'green'])
    plt.title(f'Histogram of {course} {key}')
    plt.xlabel('Hogwarts Houses')
    plt.ylabel('Mean Scores')
    plt.savefig(f'result/Histogram of {course} {key}')

if __name__ == '__main__':
    histogram()