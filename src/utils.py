import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_contour(f, x_range, y_range, paths=None, filename=None):
    """
    Plots the contour of the given function and optional algorithm paths.
    """
    # Create a grid of points
    filename = f.__name__ + "path"
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Compute function values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=170, cmap='viridis')

    # Define a list of line styles
    line_styles = ['-', '--', '-.', ':']

    # Plot algorithm paths if provided
    if paths is not None:
        for (path, name), line_style in zip(paths, line_styles):
            plt.plot(path[:, 0], path[:, 1], label=name, marker='.', linestyle=line_style)
        plt.legend()

    # Set the title
    plt.title("Contour plot of the function " + f.__name__)

    # Save the figure to a file if a filename is provided
    if filename is not None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join("plots", filename))
    else:
        plt.show()


def plot_function_values(f, function_values_dict, filename=None):
    """
    Plots function values for each method in function_values_dict.
    """
    filename = f.__name__
    plt.figure(figsize=(10, 8))

    # Define a list of line styles
    line_styles = ['-', '--', '-.', ':']

    for (method, function_values), line_style in zip(function_values_dict.items(), line_styles):
        plt.plot(function_values, label=method, marker='o', linestyle=line_style)
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.title(f.__name__ + ' values at each iteration for different methods')
    plt.legend()
    plt.grid(True)

    # Save the figure to a file if a filename is provided
    if filename is not None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join("plots", filename))
    else:
        plt.show()

