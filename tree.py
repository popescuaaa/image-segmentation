import numpy as np
from dataclasses import dataclass
import matplotlib.image as img
import math
from copy import deepcopy

global_idx = 0
global_tree = {}
nodes = []


class Node:
    def __init__(self, idx, w, h, x, y, parent_idx):
        self.idx = idx
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.parent_idx = parent_idx


# criterion
def pixel_distance_rgb(r: float, g: float, b: float):
    return math.sqrt(int(r) + int(g) + int(b))


def homogenous(idx: int, tolerance: float, image: np.ndarray) -> bool:
    global nodes
    node = nodes[idx]

    w, h = node.w, node.h

    if w == 0 or h == 0:
        return True
    else:
        min_distance = 256
        max_distance = -1

        # Set the ranges for each traversal
        from_x = node.x
        from_y = node.y
        to_x = node.x + w
        to_y = node.y + h
        print('From {} to {}'.format(from_x, to_x))

        for r in range(from_x, to_x):
            for c in range(from_y, to_y):

                r, g, b = image[r][c]
                d = pixel_distance_rgb(r, g, b)

                if d < min_distance:
                    min_distance = d
                if d > max_distance:
                    max_distance = d

        return (max_distance - min_distance) < tolerance


def add_node(parent_idx: int, x: int, y: int, w: int, h: int, tolerance: float, image: np.ndarray) -> Node:
    # Create a new node as the successor of the current one
    global global_idx
    global global_tree
    global nodes

    new_node = Node(global_idx, w, h, x, y, parent_idx)

    nodes[new_node.idx] = new_node
    global_tree[new_node.idx] = []
    global_idx += 1

    if not homogenous(new_node.idx, tolerance, image):
        print('Entered here')
        split(new_node.idx, tolerance, image)

    print('Added')
    return new_node


def split(idx: int, tolerance: float, image: np.ndarray):
    global global_tree
    global nodes

    current_node = nodes[idx]
    w, h = current_node.w, current_node.h
    _w, _h = w // 2, h // 2  # new values
    _wo, _ho = w % 2, h % 2  # offsets

    original_x, original_y = current_node.x, current_node.y

    # Split the image to all successors
    print('Adding new nodes....')

    global_tree[idx].append(add_node(idx, original_x, original_y, _w, _h, tolerance, image))

    global_tree[idx].append(add_node(idx, original_x + _w, original_y, _w + _wo, _h, tolerance, image))

    global_tree[idx].append(add_node(idx, original_x, original_y + _h, _w, _h + _ho, tolerance, image))

    global_tree[idx].append(add_node(idx, original_x + _w, original_y + _h, _w + _wo, _h + _ho, tolerance, image))


def create_tree(tolerance: float, image: np.ndarray) -> None:
    global global_idx
    global global_tree
    global nodes

    w, h, c = image.shape
    nodes = [None] * (h * w // 2)  # worst case pixel by pixel
    global_tree = {}
    global_idx = 0

    root = Node(global_idx, w, h, 0, 0, 0)
    nodes[global_idx] = root
    global_idx += 1

    global_tree[root.idx] = []
    split(root.idx, tolerance, image)


def compute_mean(image: np.ndarray) -> np.ndarray:
    global nodes
    mr, mg, mb = 0, 0, 0
    new_image = deepcopy(image)

    for n in nodes:
        counter = 0
        for i in range(n.x, n.x + n.w):
            for j in range(n.y, n.y + n.h):
                r, g, b = image[i][j]
                mr += r
                mg += g
                mb += b
                counter += 1

        if counter == 0:
            r, g, b = image[n.x][n.y]
            print(r)
            print(g)
            print(b)
            mr = int(r)
            mg = int(g)
            mb = int(b)
        else:
            mr = mr / counter
            mg = mg / counter
            mb = mb / counter

        for i in range(n.x, n.x + n.w):
            for j in range(n.y, n.y + n.h):
                new_image[i][j] = (mr, mg, mb)

    return new_image


if __name__ == '__main__':
    # test
    t = 19
    i = img.imread('images/room.jpeg')
    create_tree(t, i)
    nodes = list(filter(lambda n: n is not None, nodes))
    new_image = compute_mean(i)
    img.imsave('images/room_mean.jpeg', new_image)
