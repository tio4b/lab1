import numpy as np
from scipy.optimize._linesearch import scalar_search_armijo, line_search_wolfe1
from typing import Callable, Tuple, List

def create_function_and_gradient_counters():
    function_eval_count = [0]
    gradient_eval_count = [0]

    def f(p: np.ndarray) -> float:
        function_eval_count[0] += 1
        x, y = p[0], p[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def grad_f(p: np.ndarray) -> np.ndarray:
        gradient_eval_count[0] += 1
        x, y = p[0], p[1]
        return np.array([10 * x + 8 * y - 34, 8 * x + 10 * y - 38])

    return f, grad_f, function_eval_count, gradient_eval_count

def gradient_descent(
    x0: List[float],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    method: str = "wolfe",
    tol: float = 1e-13,
    max_iter: int = 10001,
    c1: float = 0.001,
    c2: float = 0.9
) -> Tuple[np.ndarray, int]:
    xk = np.array(x0, dtype=float)
    fk = f(xk)
    gk = grad_f(xk)
    iters = 0

    while np.linalg.norm(gk) > tol and iters < max_iter:
        pk = -gk

        if method.lower() == "armijo":
            derphi0 = np.dot(gk, pk)
            phi = lambda alpha: f(xk + alpha * pk)
            alpha, _ = scalar_search_armijo(phi, fk, derphi0, c1)
            if alpha is None:
                alpha = 1.0

        elif method.lower() in ["wolfe", "armijo_wolfe"]:
            alpha, fc, gc, new_fval, old_fval, new_slope = line_search_wolfe1(
                f, grad_f, xk, pk, old_fval=fk, c1=c1, c2=c2
            )
            if alpha is None:
                alpha = 1.0

        else:
            raise ValueError("unsupported")

        xk = xk + alpha * pk
        fk = f(xk)
        gk = grad_f(xk)
        iters += 1

    return xk, iters

if __name__ == "__main__":
    f, grad_f, f_counter, g_counter = create_function_and_gradient_counters()
    x0 = [-4, -10]
    x_opt_armijo, it_armijo = gradient_descent(x0, f, grad_f, method="armijo")
    print("Armijo")
    print("Point: %.20f %.20f" % (x_opt_armijo[0], x_opt_armijo[1]))
    print("Iter: %d" % it_armijo)
    print("function eval: %d, grad eval: %d\n" % (f_counter[0], g_counter[0]))

    f, grad_f, f_counter, g_counter = create_function_and_gradient_counters()
    x_opt_wolfe, it_wolfe = gradient_descent(x0, f, grad_f, method="wolfe")
    print("Wolfe")
    print("Point: %.20f %.20f" % (x_opt_wolfe[0], x_opt_wolfe[1]))
    print("Iter: %d" % it_wolfe)
    print("function eval: %d, grad eval: %d" % (f_counter[0], g_counter[0]))
