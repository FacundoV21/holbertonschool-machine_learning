#!/usr/bin/env python3
"""
    task 6
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
        plot a stacked bar graph
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    x = np.arange(3)

    for i in range(len(fruit)):
        plt.bar(x, fruit[i], bottom=np.sum(fruit[:i], axis=0),
                 color=colors[i], label=fruits[i], width=0.5)

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, ['Farrah', 'Fred', 'Felicia'])
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    plt.show()
