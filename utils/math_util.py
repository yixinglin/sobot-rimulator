
from math import pi


# map the given angle to the equivalent angle in [ -pi, pi ]
def normalize_angle(angle):
    return (angle + pi) % (2 * pi) - pi
