import unittest
import numpy as np
from datetime import datetime
from src.constrained_min import interior_pt
from tests.examples import min_quadratic, max_sum
from src.utils import plot_contour, plot_function_values,plot_reg_path, plot_obj_k


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        x0 = np.array([0.1, 0.2, 0.7])
        expected_result = np.array([0, 0, -1])
        result, path, objectives = interior_pt(min_quadratic, x0)

        print("path", path)
        print("result", result)
        print("expected_result", expected_result)

        filename = min_quadratic.__name__ + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # plot using the refactored functions
        plot_reg_path(min_quadratic, x0, path,objectives)
        plot_obj_k(x0, path, objectives)

    def test_lp(self):
        # Define initial guess
        x0 =np.array([0.5, 0.75])

        # Define the expected result
        expected_result = np.array([2.0, 0.0])

        # Call the function
        result, path, _ = interior_pt(max_sum, x0)
        print("path", path)
        print("result", result)
        print("expected_result", expected_result)

        # Generate unique filename
        filename = max_sum.__name__ + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot the feasible region and the path taken by the algorithm
        plot_contour(max_sum, x_range=(-2, 2), y_range=(-2, 2),
                           paths=[(np.array(path), "Interior Point")], filename=filename)

        # Plot the graph of objective value vs. outer iteration number
        plot_function_values(max_sum, {"Interior Point": [-max_sum(x)[0] for x in path]}, filename=filename)



if __name__ == "__main__":
    unittest.main()
