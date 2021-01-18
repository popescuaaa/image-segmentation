import math
from convertor import rgb_to_hsv, rgb_to_ycbcr

# Metrics modes
RGB = 'rgb'
HSV = 'hsv'
YCBCR = 'ycbcr'
ERROR = 'Incompatible mode'


def pixel_distance(r: float, g: float, b: float, mode: str):
    if mode == RGB:
        return math.sqrt(int(r) + int(g) + int(b))  # avoid maximum recursion depth for small noises
    elif mode == HSV:
        [_, _, v] = rgb_to_hsv(r, g, b)  # recommended values: ~ 30
        return v
    elif mode == YCBCR:
        [y, _, _] = rgb_to_ycbcr(r, g, b)  # recommended values: ~ 0.01
        return y
    else:
        return ERROR


def node_cluster_distance(c_mr: float, c_mg: float, c_mb: float, n_mr: float, n_mg: float, n_mb: float, mode: str):
    if mode == RGB:
        return math.sqrt((c_mr - n_mr) ** 2 + (c_mg - n_mg) ** 2 + (c_mb - n_mb) ** 2)
    elif mode == HSV:
        [_, _, cv] = rgb_to_hsv(c_mr, c_mg, c_mb)
        [_, _, nv] = rgb_to_hsv(n_mr, n_mg, n_mb)
        return abs(cv - nv)
    elif mode == YCBCR:
        [cy, _, _] = rgb_to_ycbcr(c_mr, c_mg, c_mb)
        [ny, _, _] = rgb_to_ycbcr(n_mr, n_mg, n_mb)
        return abs(cy-ny)
    else:
        return ERROR
