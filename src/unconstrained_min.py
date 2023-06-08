import numpy as np

# The main line search function
def line_search_minimize(f, x0, method="gd", obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    """
    Perform line search optimization using one of four methods:
    Gradient descent ("gd"), Newton's method ("newton"), BFGS ("bfgs"), or SR1 ("sr1").
    """
    # Initial setup
    x = x0
    history = [x0]
    success = False

    methods = ["gd", "newton", "bfgs", "sr1"]
    if method not in methods:
        raise ValueError("Invalid method. Expected one of: %s" % methods)

    n = len(x0)
    B = np.eye(n)

    for i in range(max_iter):
        g = f(x)[1]

        if method == "gd":
            p = -g
        elif method == "newton":

            H = f(x,True)[2]
            try:
                p = -np.linalg.solve(H, g)
            except:
                p = -g
        elif method in ["bfgs", "sr1"]:
            p = -B.dot(g)

        step = line_search(f, x, p)
        if step is None:
            # Line search did not converge
            print("Line search did not converge at iteration ", i)
            break

        # Update and store history
        s = step * p
        x_next = x + s
        g_next = f(x_next)[1]
        y = g_next - g
        Bs = B.dot(s)

        if method == "bfgs" and s.dot(Bs)!=0:
            # BFGS update
            B += np.outer(y, y) / y.dot(s) - np.outer(Bs, Bs) / s.dot(Bs)
        elif method == "sr1":
            # SR1 update
            d = y - Bs
            if np.abs(s.dot(d)) > 0.01 * np.linalg.norm(s) * np.linalg.norm(d) :  # Threshold condition
                B += np.outer(d, d) / d.dot(s)

        x = x_next
        history.append(x)

        # Check convergence
        if(len(history) >= 2):
            if np.abs(f(history[-1])[0] - f(history[-2])[0]) < obj_tol:
                # Success: objective function change is below tolerance
                print("Objective function tolerance reached at iteration ", i)
                success = True
                break
            elif np.linalg.norm(history[-1] - history[-2]) < param_tol:
                # Success: point change is below tolerance
                print("Parameter tolerance reached at iteration ", i)
                success = True
                break

    # Return final point, final objective value, success flag, and history of points visited
    return history[-1], f(history[-1])[0], success, history



def line_search(f, x, p, alpha=1, c1=0.01, c2=0.5, max_iter=100):
    """
    Perform backtracking line search using the Wolfe conditions.
    """
    # Initialize
    alpha_prev = 0
    phi_0 = f(x)[0]
    phi_der_0 = f(x)[1].dot(p)
    alpha_i = alpha

    for i in range(max_iter):
        # Compute function and gradient at x + alpha_i * p
        phi_i = f(x + alpha_i * p)[0]
        phi_der_i = f(x + alpha_i * p)[1].dot(p)

        if phi_i > phi_0 + c1 * alpha_i * phi_der_0:
            return zoom(f, x, p, alpha_prev, alpha_i, phi_0, phi_der_0)

        # Wolfe condition
        if abs(phi_der_i) <= -c2 * phi_der_0:
            return alpha_i

        # Update alpha
        if phi_der_i >= 0:
            return zoom(f, x, p, alpha_i, alpha_prev, phi_0, phi_der_0)

        alpha_prev = alpha_i
        alpha_i = (alpha_i + alpha) / 2  # Increase alpha

    # If the loop completes, line search did not succeed
    print("Line search did not converge")
    return None


def zoom(f, x, p, alpha_low, alpha_high, phi_0, phi_der_0, c1=0.01, c2=0.9, max_iter=100):
    """
    Perform the zoom procedure during line search.
    """
    for i in range(max_iter):
        alpha = (alpha_low + alpha_high) / 2.0
        phi = f(x + alpha * p)[0]
        if phi > phi_0 + c1 * alpha * phi_der_0 or phi >= f(x + alpha_low * p)[0]:
            alpha_high = alpha
        else:
            phi_der =f(x + alpha * p)[1].dot(p)
            if abs(phi_der) <= -c2 * phi_der_0:
                return alpha
            if phi_der * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha
    print("Zoom did not converge")
    return None