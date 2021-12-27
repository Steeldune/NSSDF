import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2)
h = 10.0
K = 4.0

x_axis1 = np.linspace(0.0, h/2, 100)
x_axis2 = np.linspace(h/2, h, 100)
y_axis1 = (h*x_axis1 - 2*x_axis1**2)*K
y_axis2 = (h-x_axis2)*(2*x_axis2-h)*K
dy_axis1 = (h-4*x_axis1)*K
dy_axis2 = (3*h - 4*x_axis2)*K

axes[0].plot(x_axis1, y_axis1, color='black')
axes[0].plot(x_axis2, y_axis2, color='black')
axes[0].set_title('Diffusion function')
axes[0].set_xlim([0.0, h])
axes[0].set_ylim([0.0, h**2])
axes[0].grid()
axes[1].plot(x_axis1, dy_axis1, color='black')
axes[1].plot(x_axis2, dy_axis2, color='black')
axes[1].set_title('Derivative of Diffusion function')
axes[1].set_xlim([0.0, h])
axes[1].set_ylim([-h**2, h**2])
axes[1].grid()

fig.suptitle('Example of diffusion function (h={}, K={})'.format(h, K), fontsize=16)

plt.show()