# referenced Numpy documentation: https://numpy.org/doc/stable/index.html
# referenced python cmath library documentation: https://docs.python.org/3.7/library/cmath.html#module-cmath
# referenced Sympy documentation: https://docs.sympy.org/latest/index.html
import numpy
import sympy
import cmath
from numpy import polynomial as pln
from sympy.abc import x
from sympy.simplify.simplify import nthroot
import scipy
from scipy import optimize
from matplotlib import pyplot

#implements the QSP polynomial construction described on page 8 of https://arxiv.org/pdf/1806.01838.pdf
def gen_polynomial_from_angles(angles):
    exp_p = cmath.exp(complex(0, angles[0]))
    P = pln.Polynomial([exp_p])

    Q = pln.Polynomial([0])
    for i in range(len(angles) - 1):
        exp_p = cmath.exp(complex(0, angles[i+1]))
        outside_term_p = pln.Polynomial([exp_p])
        inside_term_p1 = pln.polynomial.polymul(P, pln.Polynomial([0,1]))
        inside_term_p2 = pln.polynomial.polymul(Q, pln.Polynomial([-1, 0, 1]))
        inside_term_p = pln.polynomial.polyadd(inside_term_p1, inside_term_p2)

        exp_q = cmath.exp(complex(0, -1*angles[i+1]))
        outside_term_q = pln.polynomial.Polynomial([exp_q*complex(0,1)])
        inside_term_q1 = pln.polynomial.polymul(Q, pln.Polynomial([0,1]))
        inside_term_q = pln.polynomial.polyadd(inside_term_q1, P)


        P = pln.polynomial.polymul(outside_term_p, inside_term_p)
        Q = pln.polynomial.polymul(outside_term_q, inside_term_q)

    return P, Q

# definition of an error function that can be used
def deltax_eval(n):
    if n > 0:
        return n - (nthroot(x,7).evalf(subs={x:n}))**(7-2)
    else:
        return n - (nthroot(x,7).evalf(subs={x:abs(n)}))**(7-2)

# no echo determining the resulting error for splitting an angle a into x rotations
def no_echo(a, x, err_func=deltax_eval):
    return a + err_func((a/(x)))*x

# generating the resulting polynomial with noisy rotations
def gen_polynomial_with_noise_from_angles(angles, err_func=deltax_eval):
    # the list of angles is the list of phi values, which are each 1/2 of the angle rotation they represent
    # the angle coefficient is then 4/pi times these values -> pi/8 phi -> pi/4 rotation -> 0.5 coefficient
    new_angles = [(numpy.pi/4)*no_echo((angles[x]/(numpy.pi/4)), 1, err_func) for x in range(len(angles))]
    return gen_polynomial_from_angles(new_angles)

# finding the difference between the noisy and ideal polynomials
def error_polynom(angles, original_angles, err_func=deltax_eval):
    p1, q1 = gen_polynomial_with_noise_from_angles(angles, err_func)
    p2, q2 = gen_polynomial_from_angles(original_angles)
    return p1 - p2

# it is better to use the integral of the squared error to avoid cancellation, but this is provided as well
# calculates the magnitude of the integral of the error polynomial from -1 to 1
def integral_error(angles, original_angles, err_func=deltax_eval):
    print(angles)
    new_func = error_polynom(angles, original_angles, err_func)[0].integ()
    output = new_func(1) - new_func(-1)
    return abs(output)

# calculating the magnitude of the integral of the square of the error polynomial from -1 to 1
def integral_squared_error(angles, original_angles, err_func=deltax_eval):
    error = error_polynom(angles, original_angles, err_func)
    sq_error = pln.polynomial.polymul(error, error)
    new_func = sq_error[0].integ()
    output = new_func(1) - new_func(-1)
    return abs(output)


# function to apply the minimization given a list of ideal phi values and an error function
def get_new_phi_vals(initial_vals, err_func):
    output = optimize.minimize(integral_squared_error, x0=numpy.array(initial_vals), args=(initial_vals, err_func))
    return output.x.tolist()


# tests the error reduction for a sequence of angles where each angle is (pi/2)/length
def test_error_reduction_even_split(length, err_func=deltax_eval):
    initial_vals = []
    for i in range(length):
        initial_vals.append(0.25*numpy.pi*(i + 1)/length)
    results = get_new_phi_vals(initial_vals, err_func)
    errors = []
    initial_error = integral_squared_error(initial_vals, initial_vals, err_func)
    reduced_error = integral_squared_error(results, initial_vals, err_func)
    errors.append(reduced_error)
    if initial_error == 0:
        return initial_error, reduced_error, None
    return initial_error, reduced_error, reduced_error/initial_error

# tests the error reduction for an angle sequence of length "length" where each phi value is "angle"
def test_error_reduction_uniform(angle, length, err_func=deltax_eval):
    initial_vals = []
    for i in range(length):
        initial_vals.append(angle)
    results = get_new_phi_vals(initial_vals, err_func)
    errors = []
    initial_error = integral_squared_error(initial_vals, initial_vals, err_func)
    reduced_error = integral_squared_error(results, initial_vals, err_func)
    errors.append(reduced_error)
    if initial_error == 0:
        return initial_error, reduced_error, None
    return initial_error, reduced_error, reduced_error/initial_error

# calls the cooresponding error reduction function for integers n = 1 up to max_number
def get_even_split_errors(max_number, err_func = deltax_eval):
    x = []
    o = []
    r = []
    f = []
    for i in range(max_number):
        x.append(i+1)
        ie, re, fe = test_error_reduction_even_split(i+1, err_func)
        o.append(ie)
        r.append(re)
        f.append(fe)
    return x, o, r, f

# calls the cooresponding error reduction function for integers n = 1 up to max_number
def get_uniform_errors(angle, max_number, err_func = deltax_eval):
    x = []
    o = []
    r = []
    f = []
    for i in range(max_number):
        x.append(i+1)
        ie, re, fe = test_error_reduction_uniform(angle/2, i+1, err_func)
        o.append(ie)
        r.append(re)
        f.append(fe)
    return x, o, r, f

# helper function for plotting the results
def plot_results(x, o, r, f):
    pyplot.scatter(x, o, label="Without mitigation")
    pyplot.scatter(x, r, label="With mitigation")
    pyplot.legend()
    pyplot.xlabel("Number of angles in sequence")
    pyplot.ylabel("Error (distance between polynomials)")


