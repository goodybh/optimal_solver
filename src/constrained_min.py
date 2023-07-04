import autograd.numpy as np
from autograd import grad, hessian

def backtracking_line_search(f, g, x, dx, alpha=0.1, beta=0.95):
    """
    Backtracking line search to ensure the function decreases at each step.
    """
    while f(x + alpha*dx) > f(x) + alpha*beta*np.dot(g, dx):
        alpha *= beta
        if(alpha < 1e-10):
            break
    return alpha

def newton_step(H, g, eq_c_mat=None, eq_c_rhs=None, reg_term=1e-6):
    """
    Compute the Newton step direction.

    :param H: Hessian matrix
    :param g: Gradient vector
    :param eq_c_mat: Equality constraints matrix
    :param eq_c_rhs: Equality constraints right hand side
    :param reg_term: Regularization term for Hessian (default is 1e-6)
    :return: Newton step direction
    """
    n = len(g)

    # Make H positive definite if necessary
    eigvals = np.linalg.eigvals(H)
    if np.min(eigvals) <= 0:
        H += (np.abs(np.min(eigvals)) + reg_term) * np.eye(n)
    if eq_c_mat is not None and eq_c_rhs is not None:
        # Solve the KKT system if there are equality constraints
        m = eq_c_mat.shape[0]
        KKT_mat = np.block([
            [H, eq_c_mat.T],
            [eq_c_mat, np.zeros((m, m))]
        ])
        KKT_rhs = np.hstack([g, eq_c_rhs])
        sol = np.linalg.solve(KKT_mat, KKT_rhs)
        dx = sol[:n]
    else:
        # If there are no equality constraints, simply solve the Newton system
        dx = np.linalg.solve(H, -g)

    return dx

def interior_pt(f, x0, t=1, mu=10, tol_outer=1e-5, tol_inner=1e-5, max_iter=100):
    x = x0
    path = [x]
    objective_values = [f(x0)]  # Initialize the list of objective function values
    for _ in range(max_iter):
        _, _, _, ineq_c, eq_c_mat, eq_c_rhs = f(x, True)

        def f_barrier(x):
            f_barrier_value = 0
            for constraint in ineq_c:
                constraint_val = constraint(x)
                if -constraint_val <= 0:
                    print("ERROR negative log")
                    return np.inf
                else:
                    f_barrier_value -= np.log(-constraint_val)
            return f_barrier_value

        f_barrier_grad = grad(f_barrier)
        f_barrier_hessian = hessian(f_barrier)
        for _ in range(max_iter):
            g = f_barrier_grad(x)
            H = f_barrier_hessian(x)
            dx = newton_step(H, g, eq_c_mat, eq_c_rhs)
            if np.linalg.norm(dx) < tol_inner:
                break
            alpha = backtracking_line_search(f_barrier, g, x, dx)
            x = x + alpha * dx
        objective_values.append(f(x))  # Add the new function value to the list
        path.append(x)
        if t * np.linalg.norm(g) < tol_outer:
            break
        t *= mu
    return x, path, objective_values  # return the path and the list of objective function values
