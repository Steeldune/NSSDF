import numpy as np
import matplotlib.pyplot as plt
from numpy import diff


def normal_function(x, mean, var):
    expo = -0.5 * np.power((x - mean) / var, 2)
    scalar = 1 / (var * np.sqrt(2 * np.pi))
    return scalar * np.exp(expo)


def normal_function_der(x, mean, var):
    scalar = -(x - mean) / (np.power(var, 2))
    return normal_function(x, mean, var) * scalar


def diffusion_function(z, h, K):
    z = z % (h / 2.0)
    return z * (h - 2.0 * z) * K


def diffusion_der_function(z, h, K):
    z = z % (h / 2.0)
    return (h - 4 * z) * K


def generic_derivative(x, dx):
    dy = diff(x) / dx
    return dy


def calcConcChange(x_axis, c, dx, dt):
    disp = diffusion_function(x_axis, h, K)

    term1 = -1 * generic_derivative(c[1:] * generic_derivative(disp, dx), dx)
    term2 = generic_derivative(generic_derivative(disp * c, dx), dx)
    dcdt = (term1 + term2) * dt
    out = c[2:] + dcdt
    return out


if __name__ == '__main__':
    normal_mean = 0.25
    normal_var = 0.07
    h = 1.0
    K = 1.0
    dt = 0.005
    iters = 4

    x_axis = np.linspace(0.01, 0.49, 300)
    x_der_axis = x_axis[1:]
    x_2der_axis = x_axis[1:-1]
    dx = x_axis[1] - x_axis[0]
    c1 = normal_function(x_axis, normal_mean, normal_var)
    cOri = c1.copy()
    # diffusion_array = diffusion_function(x_axis, h, K)
    #
    # term1 = normal_array * diffusion_der_function(x_axis, h, K)
    # dterm1 = -1 * generic_derivative(term1, dx)
    #
    # term2 = normal_array * diffusion_array
    # dterm2 = generic_derivative(term2, dx)
    # d2term2 = generic_derivative(dterm2, dx)
    #
    # concChange = (d2term2 + dterm1[1:]) * 0.01
    plt.figure()

    for i in range(iters):
        c1 = calcConcChange(x_axis[2*i:], c1, dx, dt)
    add_eles = 2*(iters)
    add_eles_array = np.zeros((add_eles), dtype=int)
    c1 = np.insert(c1, add_eles_array, 0)

    arrow_intervals = np.linspace(0, 299, 25, dtype=int)
    arrow_intervals = arrow_intervals[1:-1]

    for i in arrow_intervals:
        plt.arrow(x_axis[i], cOri[i], 0, (c1[i]-cOri[i]), width=0.005, length_includes_head=True, head_length=0.1)

    plt.plot(x_axis, cOri, label='Input Concentration')
    # plt.plot(x_axis, c1, label='One Calc Later')
    plt.grid()
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 6.0))
    plt.xlabel('position (z)')
    plt.ylabel('concentration density (%)')
    plt.title('Change of Input concentration over time')
    plt.legend()

    plt.show()
