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
                i]) * (np.power(self.sample[i + 1] - self.sample[i], 2.0) - self.min_scale)
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
    # return x
    return 0

def f_func(x, t, h=1.0, K=1.0):
    z = x % (h / 2.0)
    # return K * (h - 4 * z)
    return 0

def g_func(x, t, h=1.0, K=1.0):
    z = x % (h / 2.0)
    return np.sqrt(2.0 * (z * (h - 2.0 * z) * K))


def g_func_der(x, t, h=1.0, K=1.0):
    z = x % (h / 2.0)
    denom = np.sqrt(K * z * (h - 2.0 * z))
    return K * (h / np.sqrt(2.0) - 2.0 * np.sqrt(2.0) * z) / denom


def g_func_alt(x, t, h=1.0, k=2.0):
    z = x % (h / 2.0)
    return z * (h - 2 * z) * k


def g_func_der_alt(x, t, h=1.0, K=2.0):
    z = x % (h / 2.0)
    return K * (h - 4 * z)


if __name__ == '__main__':
    depth = 1.0
    rng = default_rng()
    nr_samples = 100
    time_end = 10
    time_scale = 0.001
    nr_scales = 7
    nr_solvers = 7
    mfd = int(2 ** (nr_scales - 1))
    ini_point = depth / 4
    nr_histo_frames = 5
    E_or_M = False

    scale_origin = 0

    wiener_list = [None for x in range(nr_scales)]
    wiener_list[scale_origin] = [Wiener.generate_new(time_scale, time_end) for i in range(nr_samples)]

    for i in range(scale_origin):
        wiener_list[scale_origin - i - 1] = [Wiener.generate_upscale(wiener.export()) for wiener in
                                             wiener_list[scale_origin - i]]
    for i in range(scale_origin + 1, nr_scales):
        wiener_list[i] = [Wiener.generate_downscale(wiener.export()) for wiener in wiener_list[i - 1]]

    fig, ax = plt.subplots()
    fig.set_size_inches(5.2, 5 / 4 * 3)
    fig.set_dpi(300)

    solved_list = [np.zeros((nr_samples, wiener[0].max_samples)) for wiener in wiener_list]

    mean_error = np.zeros((nr_samples, nr_solvers))

    for j in tqdm(range(nr_samples)):
        # solved_list[0][j] = wiener_list[0][j].apply_function(func)
        if j == 0:
            wiener_list[0][j].plot_solved(ax, label='Exact')
        error = np.zeros((nr_solvers, len(solved_list[0][0]) // mfd))

        for i in range(nr_scales - nr_solvers, nr_scales):
            if E_or_M:
                solved_list[i][j] = wiener_list[i][j].apply_euler(f_func, g_func, ini_pos=ini_point)
            else:
                solved_list[i][j] = wiener_list[i][j].apply_milstein(f_func, g_func_alt, g_func_der_alt, ini_pos=ini_point)

            if i == 0:
                wiener_list[i][j].plot_solved(ax, label='Solved {}'.format(i))
            if i > 0:
                for k in range(0, len(solved_list[0][0]) - mfd, mfd):
                    error[i - (nr_scales - nr_solvers), k // mfd] = np.abs(
                        solved_list[0][j, k] - solved_list[i][j, k // 2 ** i])

        mean_error[j] = [np.mean(error[s, 5:]) for s in range(nr_solvers)]
    delta_ts = [wiener_list[x][0].min_scale for x in range(nr_scales - nr_solvers, nr_scales)]

    ax.set_xlabel('time (t)')
    ax.set_ylabel('position (z)')
    ax.set_title('Position of Particles over time in {} scheme'.format(('Milstein', 'Euler')[E_or_M]))
    for i in np.arange(0.0, depth+0.1, depth*0.5):
        ax.axhline(i, linestyle='--', color='gray')

    # ax.legend()
    fig = plt.figure(figsize=(5.5, 5.3 / 4 * 3), dpi=300)
    ax3 = fig.add_subplot(2, 1, 1)
    for i, track in enumerate(solved_list):
        nr_datapoints = len(track[0])
        x_axis = wiener_list[i][0].axis_x
        plot_scale = wiener_list[i][0].min_scale
        # x_axis = np.linspace(0.0, time_end, nr_datapoints)
        ax3.plot(x_axis, track[0], label='dt = {}'.format(plot_scale))

    # ax3.legend(loc=3)
    ax3.set_xlabel('time (t)')
    ax3.set_ylabel('position (z)')
    ax3.set_title('A. One process with different sampling from the Wiener Process')
    for i in np.arange(0.0, depth+0.1, depth*0.5):
        ax3.axhline(i, linestyle='--', color='gray')

    ax2 = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('{} Strong Convergence'.format(('Milstein', 'Euler')[E_or_M]), fontsize=16)

    ax2.loglog(delta_ts[1:], np.mean(mean_error, axis=0)[1:], label='Sample Error')

    x_test = np.linspace(0.0001, 1, 50)
    y_test = np.sqrt(x_test)
    ax2.loglog(x_test, y_test, label='Error fit O(dt^1/2)')
    ax2.loglog(x_test, x_test, label='Error for O(dt^1)')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Error')
    ax2.set_title('B. Strong Convergence plot with example slopes (n={})'.format(nr_samples))

    ax2.grid()
    ax2.legend()
    disc_time_len = len(solved_list[0][0])

    time_frames = np.linspace(0, disc_time_len - 1, nr_histo_frames, dtype=int)

    for frame in time_frames:
        plt.figure()
        end_histo = [solved_list[0][i][frame] for i in range(nr_samples)]
        plt.hist(end_histo, 100, range=(0.0, depth), density=True)
        plt.title('Particle distribution at time t={:.2f}, n={}'.format(frame / disc_time_len * time_end, nr_samples))
        plt.xlabel('position (z)')
        plt.ylabel('particle frequency (%)')
        plt.xlim(0.0, depth)
        plt.ylim(0.0, 4.5)

    figWeak = plt.figure(figsize=(5.5, 5.3 / 4 * 3), dpi=300)
    ax5 = figWeak.add_subplot(1,1,1)
    figWeak.subplots_adjust(hspace=0.5)

    error_mean = np.zeros((nr_solvers, len(solved_list[0][0]) // mfd))
    particle_esc = np.zeros((nr_solvers))

    for scale in range(nr_scales):
        solved_scale = solved_list[scale].copy()
        scale_mean = np.mean(solved_scale, axis=0)
        x_axis = wiener_list[scale][0].axis_x
        scale_time = wiener_list[scale][0].min_scale
        particle_ends = np.array([solved_list[scale][i][-1] for i in range(nr_samples)])
        particle_ends_1 = 0.0 < particle_ends
        particle_ends_2 = particle_ends < depth*0.5
        particle_esc[scale] = nr_samples - np.sum(particle_ends_1 * particle_ends_2)
        if scale == 0:
            ori_mean = scale_mean.copy()
        if scale > 0:
            for k in range(0, len(solved_list[0][0]) - mfd, mfd):
                error_mean[scale - (nr_scales - nr_solvers), k // mfd] = np.abs(
                    ori_mean[k] - scale_mean[k // 2 ** scale])
    dt_variances = [error_mean[i][-1] for i in range(1, nr_scales)]
    particle_esc = particle_esc/nr_samples * 100
    ax5.loglog(delta_ts[1:], dt_variances, label='Sample Error')
    ax5.loglog(x_test, y_test, label='Error fit O(dt^1/2)')
    ax5.loglog(x_test, x_test, label='Error for O(dt^1)')
    ax5.grid()
    ax5.legend(loc=4)
    figWeak.suptitle('Weak Convergence {} scheme'.format(('Milstein', 'Euler')[E_or_M]), fontsize=15)

    ax5.set_title('Absolute error at t={} over dt'.format(time_end))
    ax5.set_xlabel('timestep (dt)')
    ax5.set_ylabel('Error'.format(time_end))

    figEscape = plt.figure(figsize=(5.5, 5.3 / 4 * 3), dpi=300)
    ax6 = figEscape.add_subplot(1,1,1)
    ax6.plot(delta_ts, particle_esc)
    ax6.grid()
    ax6.set_title('Escaped Particles in the {} scheme'.format(('Milstein', 'Euler')[E_or_M]), fontsize=15)
    ax6.set_xlabel('time-step (dt)')
    ax6.set_ylabel('frequency (%)')
    ax6.set_xscale('log')


    plt.show()
