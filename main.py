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
    return np.exp(b * x)


def f_func(x, t):
    return 0.3 ** 2 / 2.0 * x


def g_func(x, t):
    return 0.3 * x


if __name__ == '__main__':
    rng = default_rng()
    nr_samples = 1000
    time_end = 10.0
    time_scale = 0.01
    nr_scales = 8
    nr_solvers = 6
    mfd = int(2**(nr_scales-1))

    scale_origin = 2

    wiener_list = [None for x in range(nr_scales)]
    wiener_list[scale_origin] = [Wiener.generate_new(time_scale, time_end) for i in range(nr_samples)]

    for i in range(scale_origin):
        wiener_list[scale_origin - i - 1] = [Wiener.generate_upscale(wiener.export()) for wiener in
                                             wiener_list[scale_origin - i]]

    for i in range(scale_origin + 1, nr_scales):
        wiener_list[i] = [Wiener.generate_downscale(wiener.export()) for wiener in wiener_list[i - 1]]

    # wiener_mids = [Wiener.generate_new(time_scale, time_end) for i in range(nr_samples)]
    # wiener_ups1 = [Wiener.generate_upscale(wiener_mid.export()) for wiener_mid in wiener_mids]
    # wiener_ups2 = [Wiener.generate_upscale(wiener_up.export()) for wiener_up in wiener_ups1]
    # wiener_downs1 = [Wiener.generate_downscale(wiener_mid.export()) for wiener_mid in wiener_mids]
    # wiener_downs2 = [Wiener.generate_downscale(wiener_down.export()) for wiener_down in wiener_downs1]
    # wiener_downs3 = [Wiener.generate_downscale(wiener_down.export()) for wiener_down in wiener_downs2]
    # wiener_downs4 = [Wiener.generate_downscale(wiener_down.export()) for wiener_down in wiener_downs3]
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    solved_list = [np.zeros((nr_samples, wiener[0].max_samples)) for wiener in wiener_list]

    # solved_exact = np.zeros((nr_samples, wiener_ups2[0].max_samples))
    # solved_mid = np.zeros((nr_samples, wiener_mids[0].max_samples))
    # solved_down1 = np.zeros((nr_samples, wiener_downs1[0].max_samples))
    # solved_down2 = np.zeros((nr_samples, wiener_downs2[0].max_samples))
    # solved_down3 = np.zeros((nr_samples, wiener_downs3[0].max_samples))
    # solved_down4 = np.zeros((nr_samples, wiener_downs4[0].max_samples))

    mean_error = np.zeros((nr_samples, nr_solvers))

    ax2 = fig.add_subplot(2, 1, 2)

    for j in range(nr_samples):
        solved_list[0][j] = wiener_list[0][j].apply_function(func)
        if j == 0:
            wiener_list[0][j].plot_solved(ax, label='Exact')
        ini_point = solved_list[0][j, 0]
        error = np.zeros((nr_solvers, len(solved_list[0][0]) // mfd))

        for i in range(nr_scales - nr_solvers, nr_scales):
            solved_list[i][j] = wiener_list[i][j].apply_euler(f_func, g_func, ini_pos=ini_point)
            if j == 0:
                wiener_list[i][j].plot_solved(ax, label='Solved {}'.format(i))

        # solved_exact[j] = wiener_list[0][j].apply_function(func)
        # if j == 0:
        #     wiener_list[0][j].plot_solved(ax, label='Exact')
        # ini_point = solved_exact[j, 0]
        #
        # # for i in range(scale_origin, nr_scales):
        #
        #
        # solved_mid[j] = wiener_mids[j].apply_euler(f_func, g_func, ini_pos=ini_point)
        # if j == 0:
        #     wiener_mids[j].plot_solved(ax, label='SDE_mid')
        #
        # solved_down1[j] = wiener_downs1[j].apply_euler(f_func, g_func, ini_pos=ini_point)
        # if j == 0:
        #     wiener_downs1[j].plot_solved(ax, label='SDE_down1')
        #
        # solved_down2[j] = wiener_downs2[j].apply_euler(f_func, g_func, ini_pos=ini_point)
        # if j == 0:
        #     wiener_downs2[j].plot_solved(ax, label='SDE_down2')
        #
        # solved_down3[j] = wiener_downs3[j].apply_euler(f_func, g_func, ini_pos=ini_point)
        # if j == 0:
        #     wiener_downs3[j].plot_solved(ax, label='SDE_down3')
        #
        # solved_down4[j] = wiener_downs4[j].apply_euler(f_func, g_func, ini_pos=ini_point)
        # if j == 0:
        #     wiener_downs4[j].plot_solved(ax, label='SDE_down4')

            for k in range(0, len(solved_list[0][0]) - mfd, mfd):
                error[i-(nr_scales-nr_solvers), k // mfd] = np.abs(solved_list[0][j, k] - solved_list[i][j, k // 2**i])

            # error[1, i // mfd] = np.abs(solved_exact[j, i] - solved_down1[j, k // (mfd // 8)])
            # error[2, i // mfd] = np.abs(solved_exact[j, i] - solved_down2[j, k // (mfd // 4)])
            # error[3, i // mfd] = np.abs(solved_exact[j, i] - solved_down3[j, k // (mfd // 2)])
            # error[4, i // mfd] = np.abs(solved_exact[j, i] - solved_down4[j, i // mfd])

        mean_error[j] = [np.mean(error[s, 5:]) for s in range(nr_solvers)]
    delta_ts = [wiener_list[x][0].min_scale for x in range(nr_scales-nr_solvers, nr_scales)]
    print(np.mean(mean_error, axis=0))
    ax.legend()

    ax2.loglog(delta_ts, np.mean(mean_error, axis=0))
    ax2.grid()

    plt.show()