from sympy import symbols, summation
from utils import without_piecewise, get_unique_var


def solve_rec_by_summing(rec_coeff, init_value, inhom_part):
    n = None
    for item in inhom_part.free_symbols:
        if str(item) == "n":
            n = item
    if n is None:
        n = symbols("n", integer=True)
    hom_solution = (rec_coeff**n) * init_value
    k = symbols(get_unique_var("k"), integer=True)
    summand = ((rec_coeff**k) * inhom_part.xreplace({n: n - k})).simplify()
    particular_solution = summation(summand, (k, 0, (n - 1)))
    particular_solution = without_piecewise(particular_solution)
    return (hom_solution + particular_solution).simplify()
