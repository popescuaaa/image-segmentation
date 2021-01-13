import numpy as np
import matplotlib.image as img
import matplotlib.colors as colors
from metrics import pixel_distance_rgb, node_cluster_distance
from copy import deepcopy
import yaml

# Init environment

global_idx = 0
global_tree = {}
nodes = []

out_image = None

tree_list = []
clusters = []
cluster_idx = 0


# Node class

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

        self.cluster_idx = None


# Cluster class
# Clusters encapsulate a list of nodes or regions with similar light intensities

class Cluster:
    def __init__(self, idx: int):
        self.idx = idx

        # Mean value for RGB
        self.mr = 0.0
        self.mg = 0.0
        self.mb = 0.0

        # Cluster elements
        self.elements = []

    def add_node(self, idx: int):
        global nodes
        self.elements.append(idx)

        # Recompute mean values
        mr = 0.0
        mg = 0.0
        mb = 0.0
        for node_idx in self.elements:
            current_node = nodes[node_idx]
            mr += current_node.mr
            mg += current_node.mg
            mb += current_node.mb

        self.mr = mr / len(self.elements)
        self.mg = mg / len(self.elements)
        self.mb = mb / len(self.elements)

    def can_be_added(self, idx: int, tolerance: float):
        global nodes
        current_node = nodes[idx]
        d = node_cluster_distance(self.mr, self.mg, self.mb, current_node.mr, current_node.mg, current_node.mb)
        return d < tolerance


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


def apply_mean_rgb_node(idx: int, mr: float or None, mg: float or None, mb: float or None) -> None:
    global global_tree
    global nodes
    global out_image

    if len(global_tree[idx]) == 0:
        # Color current node
        if mr is None and mb is None and mg is None:
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


def apply_mean_cluster(idx: int) -> None:
    global clusters

    for e in clusters[idx].elements:
        apply_mean_rgb_node(e, clusters[idx].mr, clusters[idx].mg, clusters[idx].mb)


def tree_to_list(idx: int):
    global tree_list
    if not global_tree[idx]:
        tree_list.append(idx)
    else:
        tree_to_list(global_tree[idx][0])
        tree_to_list(global_tree[idx][1])
        tree_to_list(global_tree[idx][2])
        tree_to_list(global_tree[idx][3])


def create_clusters(tolerance: float):
    global clusters
    global cluster_idx
    global tree_list

    base_cluster = Cluster(cluster_idx)
    cluster_idx += 1

    base_cluster.add_node(tree_list[0])
    nodes[tree_list[0]].cluster_idx = base_cluster.idx

    clusters.append(base_cluster)

    for node_idx in tree_list[1:]:
        added = False

        if nodes[node_idx].cluster_idx is None:
            # It can be added to a cluster
            for cluster in clusters:
                if cluster.can_be_added(node_idx, tolerance):
                    cluster.add_node(node_idx)
                    nodes[node_idx].cluster_idx = cluster.idx
                    added = True

            if not added:
                new_cluster = Cluster(cluster_idx)
                cluster_idx += 1

                # Add the node to the new cluster
                new_cluster.add_node(node_idx)
                clusters.append(new_cluster)


if __name__ == '__main__':

    # System configuration
    with open('config.yaml') as f:
        config = yaml.load(f)

    t = float(config['tolerance'])
    i = img.imread('images/{}'.format(config['target']))

    mode = config['mode']

    # Split

    # Create a quadratic tree
    create_tree(t, i)

    print('Finished quadratic tree creation....')

    # Convert tree to list
    tree_to_list(0)

    out_image = deepcopy(i)

    for e in tree_list:
        apply_mean_rgb_node(e, None, None, None)

    img.imsave('images/{}'.format(config['split_result']), out_image)

    print('Saved split version of the image....')

    # Merge

    out_image = deepcopy(i)

    create_clusters(t)

    for c in clusters:
        apply_mean_cluster(c.idx)

    img.imsave('images/{}'.format(config['merge_result']), out_image)
