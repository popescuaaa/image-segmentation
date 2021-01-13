import math


def pixel_distance_rgb(r: float, g: float, b: float):
    return math.sqrt(int(r) + int(g) + int(b))


def node_cluster_distance(c_mr: float, c_mg: float, c_mb: float, n_mr: float, n_mg: float, n_mb: float):
    return math.sqrt((c_mr - n_mr) ** 2 + (c_mg - n_mg) ** 2 + (c_mb - n_mb) ** 2)
