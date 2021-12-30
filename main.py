# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.random import default_rng


class Wiener():
    def __init__(self, wiener_sample, scale=None, start=None,
                 end=None):  # Accepts an array with two rows, one with a Wiener process, the other the time samples.
        if scale is None:
            self.min_scale = wiener_sample[0, 1] - wiener_sample[0, 0]
        else:
            self.min_scale = scale
        if start is None:
            self.start = 0.0
        else:
            self.start = start
        if end is None:
            self.end = wiener_sample[0, -1]
        else:
            self.end = end
        self.max_samples = len(wiener_sample[0])
        self.sample = wiener_sample[1]
        self.axis_x = wiener_sample[0]
        self.solved = np.zeros(self.max_samples)

    @classmethod
    def generate_new(cls, scale, end, start=0.0):
        axis_x = np.arange(start, end, scale)
        random_table = rng.normal(size=len(axis_x))
        scale_variance = np.sqrt(scale)
        sample = np.cumsum(random_table * scale_variance + start)
        return cls(np.array([axis_x, sample]), scale=scale, start=start, end=end)

    @classmethod
    def generate_upscale(cls, wiener):
        upscale_wiener = np.zeros((2, len(wiener[0]) * 2 - 1))
        for i in range(len(wiener[0]) - 1):
            upscale_wiener[0, 2 * i] = wiener[0, i]
            upscale_wiener[1, 2 * i] = wiener[1, i]
            upscale_wiener[0, 2 * i + 1] = (wiener[0, i + 1] + wiener[0, i]) / 2
            upscale_wiener[1, 2 * i + 1] = middle_upscale(wiener[1, i], wiener[1, i + 1], wiener[0, i],
                                                          wiener[0, i + 1])
        upscale_wiener[0, -1] = wiener[0, -1]
        upscale_wiener[1, -1] = wiener[1, -1]
        return cls(upscale_wiener)

    @classmethod
    def generate_downscale(cls, wiener):
        downscale_wiener = np.zeros((2, len(wiener[0]) // 2))
        for i in range(len(downscale_wiener[0])):
            downscale_wiener[0, i] = wiener[0, i * 2]
            downscale_wiener[1, i] = wiener[1, i * 2]
        return cls(downscale_wiener)

    def apply_euler(self, f_func, g_func, ini_pos=0.0):
        self.solved[0] = ini_pos
        for i in range(self.max_samples - 1):
            x = self.solved[i]
            self.solved[i + 1] = x + f_func(x, self.axis_x[i]) * self.min_scale + g_func(x, self.axis_x[i]) * (
                    self.sample[i + 1] - self.sample[i])
        return self.solved.copy()

    def apply_milstein(self, f_func, g_func, g_func_der, ini_pos=0.0):
        self.solved[0] = ini_pos
        for i in range(self.max_samples - 1):
            x = self.solved[i]
            self.solved[i + 1] = x + f_func(x, self.axis_x[i]) * self.min_scale + g_func(x, self.axis_x[i]) * (
                    self.sample[i + 1] - self.sample[i]) + 0.5 * g_func(x, self.axis_x[i]) * g_func_der(x, self.axis_x[
                i]) * ((
                               self.sample[i + 1] - self.sample[i]) ** 2 - self.min_scale)

    def apply_function(self, function):
        self.solved = function(self.sample)
        return self.solved.copy()

    def plot_sample(self, axis, color=None, label=None):
        axis.plot(self.axis_x, self.sample, color=color, label=label)

    def plot_solved(self, axis, color=None, label=None, format_string=''):
        axis.plot(self.axis_x, self.solved, format_string, color=color, label=label)

    def export(self):
        return np.array([self.axis_x, self.sample])


def middle_upscale(a, b, t1, t2):
    mean = a + 0.5 * (b - a)
    variance = np.abs(0.5 * (t2 - t1))
    return rng.normal(mean, variance)


def func(x, b=0.3):
    return x


def f_func(x, t):
    return 0


def g_func(x, t, h=10.0, K=0.01):
    z = x % (h / 2.0)
    return np.sqrt(2 * (z * (h - 2 * z) * K))


def g_func_der(x, t, h=10.0, K=0.001):
    z = x % (h / 2.0)
    return (h - 4 * z) * K


if __name__ == '__main__':
    rng = default_rng()
    nr_samples = 50
    time_end = 20.0
    time_scale = 0.001
    nr_scales = 8
    nr_solvers = 8
    mfd = int(2 ** (nr_scales - 1))
    ini_point = 2.5

    scale_origin = 0

    wiener_list = [None for x in range(nr_scales)]
    wiener_list[scale_origin] = [Wiener.generate_new(time_scale, time_end) for i in range(nr_samples)]

    for i in range(scale_origin):
        wiener_list[scale_origin - i - 1] = [Wiener.generate_upscale(wiener.export()) for wiener in
                                             wiener_list[scale_origin - i]]

    for i in range(scale_origin + 1, nr_scales):
        wiener_list[i] = [Wiener.generate_downscale(wiener.export()) for wiener in wiener_list[i - 1]]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    solved_list = [np.zeros((nr_samples, wiener[0].max_samples)) for wiener in wiener_list]

    mean_error = np.zeros((nr_samples, nr_solvers))

    ax2 = fig.add_subplot(2, 1, 2)

    for j in tqdm(range(nr_samples)):
        # solved_list[0][j] = wiener_list[0][j].apply_function(func)
        # if j == 0:
        #     wiener_list[0][j].plot_solved(ax, label='Exact')
        error = np.zeros((nr_solvers, len(solved_list[0][0]) // mfd))

        for i in range(nr_scales - nr_solvers, nr_scales):
            solved_list[i][j] = wiener_list[i][j].apply_milstein(f_func, g_func, g_func_der, ini_pos=ini_point)
            if i == 0:
                wiener_list[i][j].plot_solved(ax, label='Solved {}'.format(i))
            if i > 0:
                for k in range(0, len(solved_list[0][0]) - mfd, mfd):
                    error[i - (nr_scales - nr_solvers), k // mfd] = np.abs(
                        solved_list[0][j, k] - solved_list[i][j, k // 2 ** i])

        mean_error[j] = [np.mean(error[s, 5:]) for s in range(nr_solvers)]
    delta_ts = [wiener_list[x][0].min_scale for x in range(nr_scales - nr_solvers, nr_scales)]

    # ax.legend()

    ax2.loglog(delta_ts[1:], np.mean(mean_error, axis=0)[1:], label='Sample Error')

    x_test = np.linspace(0.01, 1, 50)
    y_test = np.sqrt(x_test)
    ax2.loglog(x_test, y_test, label='Error fit O(dt^1/2)')

    ax2.grid()
    ax2.legend()

    plt.figure()
    end_histo = [solved_list[0][i][-1] for i in range(len(solved_list[0]))]
    plt.hist(end_histo, 40, range=(0.0, 10.0), density=True)
    plt.xlim(0.0, 10.0)

    plt.show()
