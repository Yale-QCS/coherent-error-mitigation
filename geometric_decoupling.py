
#documentation used: Mathematica, Numpy, SymPy, Matplotlib, SciPy
# https://reference.wolfram.com/language/#StringsAndText
# https://numpy.org/doc/stable/
# https://docs.sympy.org/latest/index.html
# https://matplotlib.org/stable/index.html
# https://docs.scipy.org/doc/scipy/index.html

# Was helpful for debugging:
# https://stackoverflow.com/questions/71470352/ufunctypeerror-cannot-cast-ufunc-det-input-from-dtypeo-to-dtypefloat64

from matplotlib.pyplot import plot
import numpy
import sympy
import matplotlib
import scipy
from scipy.optimize import fsolve
from numpy import polynomial
from matplotlib import pyplot
from sympy.simplify.simplify import nthroot
from sympy import chebyshevt
from sympy import plotting
from sympy.abc import x
from sympy.solvers import solve
from sympy.abc import w, x, y, z, a, b, c


# example error function, generally use d = 7
def deltax(x, d): 
    return x - (nthroot(x , d))**(d-2)

# way to evaluate example error function, generally use d = 7
def deltax_eval(n, d):
    return n - (nthroot(x,d).evalf(subs={x:n}))**(d-2)


#no echo - effective angle coefficient for rotation if we split up the rotation into x pieces without geometric decoupling
def no_echo(a, x, d, err_func):
    return a + err_func((a/(x)), d)*x


# modified version of chebyshev polynomial to be used in geometric decoupling constraints
def cheb_shifted_scaled_poly(m):
    polynomial = chebyshevt(m+1, 2*x-1)
    return polynomial + sympy.Poly((((-1)**(m+1))*(x - 1)) - x, x)

# returns the constraint function for the ith polynomial
def constraint_func(x, i, a_c):
    n = len(x)
    array = []
    for ii in range(i):
        array.append(0)
    array.append(1)

    if i == 0:
        constraint = -1*(a_c)
    else:
        constraint = 0
    for j in range(n):
        if i == 0:
            constraint += ((-1)**(j+1))*(x[j])
        else:
            curr_func = cheb_shifted_scaled_poly(i)
            constraint += ((-1)**(j+1)*curr_func(x[j]))
    return constraint


# sums all the constraints (except for alternating sum) to a single function to be minimized, prioritizing minimizing earlier terms in the sum
def nonlinear_constraint_func(n, a_c, weight):
    constraints = 0
    for i in range(len(n)):
        if i == 0:
            continue
        #constraints += abs(constraint_func(n,i, a_c))
        #changed to prioritize first terms in sum to best minimized error
        constraints += abs(constraint_func(n,i, a_c))*((weight)**(len(n) - i))
    return constraints



# constraint function saying that the alternating sum should equal a_c
def sum_func(x, a_c):
    total = -1*a_c
    for i in range(len(x)):
        total += ((-1)**(i+1))*(x[i])
    return abs(total)

# objective function using all equations including alternating sum
def nonlinear_constraint_func2(n, a_c, weight):
    constraints = 0
    for i in range(len(n)):
        constraints += abs(constraint_func(n,i, a_c))*((weight)**(len(n) - i))
    return constraints

# minimization constraint that alternating sum should equal a_c
def sum_constraint(a_c):
    return [{'type': 'eq', 'fun': sum_func, 'args': [a_c]}]


# solves the system of equations with the polynomials in the objective function, and the alternating sum as an additional constraint
def scipy_nonlinear_solver(a_c, n, weight):
    input_vals = [a_c * x/(2*n) for x in range(2*n)]
    constraints = sum_constraint(a_c)
    b = []
    for i in range(2*n):
        b.append([0,4])
    return scipy.optimize.minimize(nonlinear_constraint_func, input_vals, args=(a_c, weight), bounds=b, constraints=constraints)

# solves the system of equations with the polynomials in the objective function, and the alternating sum as an additional constraint (but it's an inequality constraint)
# does not have good performance
def scipy_nonlinear_solver_ineq(a_c, n, weight):
    input_vals = [a_c * x/(2*n) for x in range(2*n)]
    b = []
    for i in range(2*n):
        b.append([0,4])
    return scipy.optimize.minimize(nonlinear_constraint_func, input_vals, args=(a_c, weight), bounds=b, constraints=[{'type': 'ineq', 'fun': sum_func, 'args': [a_c]}])

# solves the system of equations with the polynomials and alternating sum in the objective function, and additional sum constraint
def scipy_nonlinear_solver_repeated_constraint(a_c, n, weight):
    input_vals = [a_c * x/(2*n) for x in range(2*n)]
    b = []
    for i in range(2*n):
        b.append([0,4])
    return scipy.optimize.minimize(nonlinear_constraint_func2, input_vals, args=(a_c, weight), bounds=b, constraints={'type': 'eq', 'fun': sum_func, 'args': [a_c]})

# applies geometric decoupling using the calculated turning points to get the effective angle coefficient of rotation
# for each of the three objective functions
def echoed_nonlinear(a, nn, err_func, weight):
    soln = scipy_nonlinear_solver(a, nn, weight).x.tolist()
    array = [((-1)**(n+1))*(soln[n] + err_func(soln[n])) for n in range(len(soln))]
    return sum(array)

def echoed_nonlinear_ineq(a, nn, err_func, weight):
    soln = scipy_nonlinear_solver_ineq(a, nn, weight).x.tolist()
    array = [((-1)**(n+1))*(soln[n] + err_func(soln[n])) for n in range(len(soln))]
    return sum(array)

def echoed_nonlinear_repeated_constraint(a, nn, err_func, weight):
    soln = scipy_nonlinear_solver_repeated_constraint(a, nn, weight).x.tolist()
    array = [((-1)**(n+1))*(soln[n] + err_func(soln[n])) for n in range(len(soln))]
    return sum(array)

# the analytic solution to the system of equations for a_c = 0.5
def cheb_polynomial_approx_solution(n, N):
    return 0.5 * (1 - numpy.cos((numpy.pi * n)/(2*N+1)))


# plots the errors for the first objective function
def plot_errors(angle, n, err_func, weight):
    x = []
    o = []
    m = []
    gd = []
    for i in range(n):
        if angle == 0.5:
            actual_soln = [cheb_polynomial_approx_solution(x+1, i+1) for x in range(2*(i+1))]
            array = [((-1)**(x+1))*(actual_soln[x] + err_func(actual_soln[x])) for x in range(len(actual_soln))]
            gd.append(abs(0.5 - sum(array)))
        o.append(abs(err_func(angle)))
        m.append(abs(angle - echoed_nonlinear(angle, i+1, err_func, weight)))
        x.append(i+1)
    pyplot.plot(x, o, color="red", label="with error")
    pyplot.scatter(x, m, label="numerically mitigated")
    if angle == 0.5:
        pyplot.scatter(x, gd, label="analytic solution")
    pyplot.legend()
    pyplot.xlabel("N (where 2N is the number of turning points)")
    pyplot.ylabel("Error in angle coefficient")
    pyplot.title("Rotational Noise Mitigation Using Geometric Decoupling")


# plots the errors for the second objective function
def plot_errors_ineq(angle, n, err_func, weight):
    x = []
    o = []
    m = []
    gd = []
    for i in range(n):
        if angle == 0.5:
            actual_soln = [cheb_polynomial_approx_solution(x+1, i+1) for x in range(2*(i+1))]
            array = [((-1)**(x+1))*(actual_soln[x] + err_func(actual_soln[x])) for x in range(len(actual_soln))]
            gd.append(abs(0.5 - sum(array)))
        o.append(abs(err_func(angle)))
        m.append(abs(angle - echoed_nonlinear_ineq(angle, i+1, err_func, weight)))
        x.append(i+1)
    pyplot.plot(x, o, color="red", label="with error")
    pyplot.scatter(x, m, label="numerically mitigated")
    if angle == 0.5:
        pyplot.scatter(x, gd, label="analytic solution")
    pyplot.legend()
    pyplot.xlabel("N (where 2N is the number of turning points)")
    pyplot.ylabel("Error in angle coefficient")
    pyplot.title("Rotational Noise Mitigation Using Geometric Decoupling")

# plots the errors for the third objective function
def plot_errors_repeated_constraint(angle, n, err_func, weight):
    x = []
    o = []
    m = []
    gd = []
    for i in range(n):
        if angle == 0.5:
            actual_soln = [cheb_polynomial_approx_solution(x+1, i+1) for x in range(2*(i+1))]
            array = [((-1)**(x+1))*(actual_soln[x] + err_func(actual_soln[x])) for x in range(len(actual_soln))]
            gd.append(abs(0.5 - sum(array)))
        o.append(abs(err_func(angle)))
        m.append(abs(angle - echoed_nonlinear_repeated_constraint(angle, i+1, err_func, weight)))
        x.append(i+1)
    pyplot.plot(x, o, color="red", label="with error")
    pyplot.scatter(x, m, label="numerically mitigated")
    if angle == 0.5:
        pyplot.scatter(x, gd, label="analytic solution")
    pyplot.legend()
    pyplot.xlabel("N (where 2N is the number of turning points)")
    pyplot.ylabel("Error in angle coefficient")
    pyplot.title("Rotational Noise Mitigation Using Geometric Decoupling")

