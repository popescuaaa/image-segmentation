import math


def pixel_distance_rgb(r: float, g: float, b: float):
    return math.sqrt(int(r) + int(g) + int(b))
