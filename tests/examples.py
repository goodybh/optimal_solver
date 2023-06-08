import numpy as np


def quadratic_example1(x, need_hessian = False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T @ Q @ x
    g = Q @ x
    if need_hessian:
        h = Q
    else:
        h = None
    return f, g, h


def quadratic_example2(x, need_hessian = False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T @ Q @ x
    g = Q @ x
    if need_hessian:
        h = Q
    else:
        h=None
    return f, g, h


def quadratic_example3(x, need_hessian = False):
    rotation_matrix = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q = rotation_matrix.T @ np.array([[100, 0], [0, 1]]) @ rotation_matrix
    f = x.T @ Q @ x
    g = Q @ x
    if need_hessian:
        h = Q
    else:
        h =None
    return f, g, h


def rosenbrock_function(x, need_hessian=False):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array([
        -400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2)
    ])

    if need_hessian:
        h = np.array([
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    else:
        h = None

    return f, g, h


def linear_function(x, need_hessian = False):
    a = np.array([2, 3])
    f = a.T @ x
    g = a
    if need_hessian:
        h = np.zeros((2, 2))
    else:
        h = None
    return f, g, h


def smooth_corner_triangles(x, need_hessian=False):
    a = x[0] + 3 * x[1] - 0.1
    b = x[0] - 3 * x[1] - 0.1
    c = -x[0] - 0.1

    f = np.sum([np.exp(a),np.exp(b), np.exp(c)])
    g = np.array([
        np.exp(a) + np.exp(b)  - np.exp(c),
        3*np.exp(a) - 3*np.exp(b)
    ])

    if need_hessian:
        h = np.array([
            [np.exp(a) + np.exp(b) + np.exp(c), 3*np.exp(a) -3 *np.exp(b)],
            [3*np.exp(a) -3 *np.exp(b), 9*np.exp(a) + 9 *np.exp(b)]
        ])
    else:
        h = None

    return f, g, h
