import matplotlib.pyplot as plt
import numpy as np
import math

"""
This is a script to produce plots visualizing the difference between applying Gaussian noise
to the motion command before executing the motion (translational and rotational velocities)
or to the resulting robot pose after a noise free motion command was executed.
"""

# Fixing random state for reproducibility
np.random.seed(19680801)

N = 1000
vm = 1
wm = 1
alpha = math.pi/2
time = np.arange(0, 1, 0.01)


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def motion_model(vs, ws, dt):
    x = vs / ws * (np.sin(alpha + dt * ws) - math.sin(alpha))
    y = vs / ws * (-np.cos(alpha + dt * ws) + math.cos(alpha))
    return x, y


def set_axes(plt):
    axes = plt.gca()
    axes.set_xlim([-1, 0.2])
    axes.set_ylim([0, 1.2])


def noiseBeforeCommand():
    set_axes(plt)
    v = vm + np.random.normal(scale=.05, size=N)
    w = wm + np.random.normal(scale=0.6, size=N)
    u = np.vstack((v, w))

    x, y = motion_model(v, w, 1)
    plt.scatter(x, y, s=.2, c="black")
    plt.plot(motion_model(vm, wm, time)[0], motion_model(vm, wm, time)[1])
    plt.savefig('noise_before_command.png')
    plt.close()


def noiseAfterCommand():
    set_axes(plt)
    x, y = motion_model(vm, wm, 1)
    x = x + np.random.normal(scale=.075, size=N)
    y = y + np.random.normal(scale=.075, size=N)
    plt.scatter(x, y, s=.2, c="black")
    plt.plot(motion_model(vm, wm, time)[0], motion_model(vm, wm, time)[1])
    plt.savefig('noise_after_command.png')
    plt.close()


noiseBeforeCommand()
noiseAfterCommand()
