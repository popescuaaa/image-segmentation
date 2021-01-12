import numpy as np
import matplotlib.image as img
from metrics import pixel_distance_rgb
from copy import deepcopy
import yaml

global_idx = 0
global_tree = {}
nodes = []
out_image = None


class Node:
    def __init__(self, idx, w, h, x, y, parent_idx):
        self.idx = idx
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.parent_idx = parent_idx

        # Mean values for RGB
        self.mr = 0.0
        self.mg = 0.0
        self.mb = 0.0


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

        mr = 0
        mg = 0
        mb = 0
        counter = 0

        for r in range(from_x, to_x):
            for c in range(from_y, to_y):

                r, g, b = image[r][c]
                d = pixel_distance_rgb(r, g, b)

                if d < min_distance:
                    min_distance = d
                if d > max_distance:
                    max_distance = d

                mr += r
                mg += g
                mb += b

                counter += 1

        nodes[idx].mr = mr // counter
        nodes[idx].mg = mg // counter
        nodes[idx].mb = mb // counter

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
        split(new_node.idx, tolerance, image)
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

    global_tree[idx].append(add_node(idx, original_x, original_y, _w, _h, tolerance, image).idx)

    global_tree[idx].append(add_node(idx, original_x + _w, original_y, _w + _wo, _h, tolerance, image).idx)

    global_tree[idx].append(add_node(idx, original_x, original_y + _h, _w, _h + _ho, tolerance, image).idx)

    global_tree[idx].append(add_node(idx, original_x + _w, original_y + _h, _w + _wo, _h + _ho, tolerance, image).idx)


def create_tree(tolerance: float, image: np.ndarray) -> None:
    global global_idx
    global global_tree
    global nodes

    w, h, c = image.shape
    nodes = [None] * (h * w)  # worst case pixel by pixel
    global_tree = {}
    global_idx = 0

    root = Node(global_idx, w, h, 0, 0, 0)
    nodes[global_idx] = root
    global_idx += 1

    global_tree[root.idx] = []
    split(root.idx, tolerance, image)


def apply_mean_rgb_node(idx: int) -> None:
    global global_tree
    global nodes
    global out_image

    if len(global_tree[idx]) == 0:
        # Color current node
        mr, mg, mb = nodes[idx].mr, nodes[idx].mg, nodes[idx].mb
        x, y = nodes[idx].x, nodes[idx].y
        w, h = nodes[idx].w, nodes[idx].h

        for i in range(x, x + w):
            for j in range(y, y + h):
                out_image[i][j] = (mr, mg, mb)

    else:
        # If the node has successors, we are sure that the branching factor is 4
        # and we can recursively apply mean value on them

        successors = global_tree[idx]

        top_left_idx = successors[0]
        top_right_idx = successors[1]
        bottom_left_idx = successors[2]
        bottom_right_idx = successors[3]

        apply_mean_rgb_node(top_left_idx)
        apply_mean_rgb_node(top_right_idx)
        apply_mean_rgb_node(bottom_left_idx)
        apply_mean_rgb_node(bottom_right_idx)


if __name__ == '__main__':

    # System configuration
    with open('config.yaml') as f:
        config = yaml.load(f)

    t = float(config['tolerance'])
    i = img.imread('images/{}'.format(config['target']))

    create_tree(t, i)

    out_image = deepcopy(i)

    for e in global_tree:
        apply_mean_rgb_node(e)

    img.imsave('images/{}'.format(config['result']), out_image)