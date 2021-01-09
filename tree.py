import numpy as np


class Node:
    def __init__(self, tolerance: float, image: np.ndarray):
        self.tolerance = tolerance
        self.image = image

        # every node has 4 descendents as splitting the image into 4 different regions
        # therefore we have top [ right, left ] and bottom [ right, left ]

        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
