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

    plt.hist(student_grades, bins=np.arange(40, 110, 10), edgecolor='black')

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.ylim(top=30)
    plt.xticks(np.arange(10, 101, 10))
    plt.show()
