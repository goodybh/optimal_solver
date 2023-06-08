import unittest
import numpy as np
from src.unconstrained_min import line_search_minimize
from src.utils import plot_contour, plot_function_values
from examples import quadratic_example1, quadratic_example2, quadratic_example3, rosenbrock_function, linear_function, smooth_corner_triangles
import sys
import contextlib

class TestUnconstrainedMin(unittest.TestCase):

    def setUp(self):
        self.methods = ["gd", "newton", "bfgs", "sr1"]
        self.x0 = np.array([1.0, 1.0])
        self.examples = [quadratic_example1, quadratic_example2, quadratic_example3, rosenbrock_function, linear_function, smooth_corner_triangles]

    def test_methods(self):
        # Open the text file
        with open('output.txt', 'w') as f, contextlib.redirect_stdout(f):
            for example in self.examples:
                dim=3
                paths = []
                function_values = {}
                for method in self.methods:
                    dim = 3
                    # Perform minimization
                    if example.__name__ == "rosenbrock_function":
                        dim = 8
                        if method == "gd":
                            max_iter = 10000
                        else:
                            max_iter = 100
                        x, fval, success, history = line_search_minimize(example, np.array([-1.0,2.0]), method, max_iter)
                    else:
                        x, fval, success, history = line_search_minimize(example, self.x0, method)

                    # Print final iteration report to the text file
                    print(f"\nExample: {example.__name__}")
                    print(f"Method: {method}")
                    print(f"Final point: {x}")
                    print(f"Final objective value: {fval}")
                    print(f"Success flag: {success}")

                    # Store history for plot
                    paths.append((np.array(history), method))

                    # Store function values for plot
                    function_values[method] = [example(h)[0] for h in history]

                # Create contour plot
                plot_contour(example, (-dim, dim), (-dim, dim), paths)

                # Create function values plot
                plot_function_values(example, function_values)


if __name__ == "__main__":
    unittest.main()
