import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_contour(f, x_range, y_range, paths=None, filename=None):
    """
    Plots the contour of the given function and optional algorithm paths.
    """
    # Create a grid of points
    filename = f.__name__ + "path"
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)

    # Compute function values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j], 0]))[0]

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


def generate_grid():
    x = np.linspace(0, 2, 50)
    y = np.linspace(0, 2, 50)
    z = np.linspace(0, 2, 50)
    X, Y, Z = np.meshgrid(x, y, z)
    return X, Y, Z


def plot_path(ax, path, objectives):
    print("Length of path:", len(path))
    print("Length of objectives:", len(objectives))
    x_values = np.array(path)

    # Calculate the mean of each item in objectives
    objective_values = []
    for obj in objectives:
        try:
            objective_values.append(np.mean(np.array(obj)))
        except:
            objective_values.append(np.nan)

    ax.plot(x_values[:,0], x_values[:,1], x_values[:,2], objective_values, 'ro-', label='Path')
    ax.legend()



def plot_reg_path(func, x, path, objectives):
    X, Y, Z = generate_grid()

    # Evaluate constraints on grid points
    _, _, _, ineq_c, _, _ = func(np.vstack((X.flatten(), Y.flatten(), Z.flatten())))

    constraint_grid = np.array([c(np.vstack((X.flatten(), Y.flatten(), Z.flatten()))) for c in ineq_c])
    constraint_values_grid = np.all(constraint_grid >= 0, axis=0).reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[constraint_values_grid], Y[constraint_values_grid], Z[constraint_values_grid], c='b', alpha=0.3,label='Feasible Region')
    plot_path(ax, path, objectives)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Feasible Region and Path')
    ax.legend()

    plt.show()


def plot_obj_cons(x, path):
    final_iteration, final_x, final_fx, final_cons = path[-1]
    plt.scatter(final_iteration, final_fx, c='red', label='Final Objective Value')

    constraint_labels = ['Constraint 1', 'Constraint 2', 'Constraint 3', 'Constraint 4']
    for i, constraint_value in enumerate(final_cons):
        plt.scatter(final_iteration, constraint_value, label=constraint_labels[i])

    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Objective and Constraint Values')
    plt.legend()

    plt.show()


def plot_obj_k(x0, path, objectives):
    # Ensure that the path and objectives are numpy arrays for easier manipulation
    path = np.array(path)
    objectives = np.array(objectives)

    # Create an array for iteration count
    k_values = np.array(range(len(path)))

    # Assuming each element of 'objectives' has 6 elements, plot them individually
    for i in range(6):
        obj_i = objectives[:, i]
        plt.plot(k_values, obj_i, marker='o', linestyle='-', label=f'Objective {i+1}')

    plt.xlabel('Iterat  ion')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs. Iteration')
    plt.legend()

    plt.show()
