#!/usr/bin/env python3
"""
    task 4
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
        plot a histogram of student scores for a project
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.hist(student_grades, np.arange(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10), np.arange(0, 101, 10))
    plt.ylim(0, 30)
    plt.show()
