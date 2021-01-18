import numpy as np
import matplotlib.image as img
from metrics import pixel_distance_rgb, node_cluster_distance
from copy import deepcopy
import yaml
import time
from threading import Thread, Lock

# Init environment

global_idx = 0
global_tree = {}
nodes = {}

out_image = None

tree_list = []
clusters = {}
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

        # Cluster identification index
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

    def add_node(self, idx: int) -> None:
        global nodes
        if idx in self.elements:
            return
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

        nodes[idx].cluster_idx = self.idx

    def can_be_added(self, idx: int, tolerance: float):
        global nodes
        current_node = nodes[idx]
        d = node_cluster_distance(self.mr, self.mg, self.mb, current_node.mr, current_node.mg, current_node.mb)
        return d < tolerance


def touching(n1_idx: int, n2_idx: int) -> bool:
    global nodes
    n1 = nodes[n1_idx]
    n2 = nodes[n2_idx]

    if n1.x == n2.x or n1.x == n2.x + n2.w:
        return (n2.y <= n1.y <= n2.y + n2.h) or (n1.y <= n2.y <= n1.y + n1.h)
    if n1.y == n2.y or n1.y == n2.y + n2.h:
        return (n2.x <= n1.x <= n2.x + n2.w) or (n1.x <= n2.x <= n1.x + n1.w)

    return False


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
    nodes = {}
    global_tree = {}
    global_idx = 0

    root = Node(global_idx, w, h, 0, 0, -1)
    nodes[global_idx] = root
    global_idx += 1

    global_tree[root.idx] = []
    split(root.idx, tolerance, image)


def apply_mean_rgb_node(idx: int, mr: float or None = None, mg: float or None = None, mb: float or None = None) -> None:
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


def tree_to_list(idx: int) -> None:
    global tree_list
    if not global_tree[idx]:
        tree_list.append(idx)
    else:
        tree_to_list(global_tree[idx][0])
        tree_to_list(global_tree[idx][1])
        tree_to_list(global_tree[idx][2])
        tree_to_list(global_tree[idx][3])


class ClusterCreator(Thread):
    def __init__(self, idx: int, tolerance: float, start_index: int, end_index: int):
        super().__init__()
        self.idx = idx
        self.sync = Lock()
        self.start_index = start_index
        self.end_index = end_index
        self.tolerance = tolerance

    def merge(self, main_node: int, target: int):
        global nodes
        global clusters
        global cluster_idx

        main_cluster_idx = nodes[main_node].cluster_idx
        target_cluster_idx = nodes[target].cluster_idx

        if main_cluster_idx is None and target_cluster_idx is None:
            # Create a new cluster for both of them
            new_cluster = Cluster(cluster_idx)

            self.sync.acquire()
            cluster_idx += 1
            clusters[new_cluster.idx] = new_cluster
            self.sync.release()

            new_cluster.add_node(main_node)
            if new_cluster.can_be_added(target, self.tolerance):
                new_cluster.add_node(target)

        elif main_cluster_idx is None and target_cluster_idx is not None:
            if clusters[target_cluster_idx].can_be_added(main_node, self.tolerance):
                clusters[target_cluster_idx].add_node(main_node)

        elif main_cluster_idx is not None and target_cluster_idx is None:
            if clusters[main_cluster_idx].can_be_added(target, self.tolerance):
                clusters[main_cluster_idx].add_node(target)

        elif main_cluster_idx is not None and target_cluster_idx is not None:
            # Merge two clusters
            if main_cluster_idx != target_cluster_idx:
                main_cluster = clusters[main_cluster_idx]
                target_cluster = clusters[target_cluster_idx]

                added = []

                self.sync.acquire()

                for te in target_cluster.elements:
                    if main_cluster.can_be_added(te, self.tolerance):
                        main_cluster.add_node(te)
                        added.append(te)

                for tea in added:
                    target_cluster.elements.remove(tea)

                if len(target_cluster.elements) == 0:
                    print('Cluster: {} has been deleted by thread: {}'.format(target_cluster_idx, self.idx))
                    clusters.pop(target_cluster_idx)

                self.sync.release()

    def create_clusters(self):
        global clusters
        global cluster_idx
        global tree_list

        # Region growing
        nodes_stack = [tree_list[self.start_index]]  # nodes to cluster
        discovered = []
        while nodes_stack:
            print(len(nodes_stack))
            cni = nodes_stack.pop()  # This node has no cluster to be part of
            if cni not in discovered:
                discovered.append(cni)

            cnn = filter(lambda n: touching(n, cni) or touching(cni, n), tree_list[self.start_index:self.end_index])
            for n in cnn:
                self.merge(cni, n)
                if n not in discovered:
                    nodes_stack.append(n)
                    discovered.append(n)

    def run(self) -> None:
        self.create_clusters()


if __name__ == '__main__':

    # System configuration
    with open('config.yaml') as f:
        config = yaml.load(f)

    t = float(config['tolerance'])
    i = img.imread('images/{}'.format(config['target']))

    mode = config['mode']

    # Split
    # Create a quadratic tree

    start = time.process_time()
    create_tree(t, i)
    elapsed_time = time.process_time() - start
    print('Finished quadratic tree creation after {}....'.format(elapsed_time))

    # Convert tree to list
    tree_to_list(0)
    print('Number of nodes: {}'.format(len(tree_list)))
    out_image = deepcopy(i)
    for e in tree_list:
        apply_mean_rgb_node(e)  # with self mean values
    img.imsave('images/{}'.format(config['split_result']), out_image)
    print('Saved split version of the image....')

    # Merge
    print(len(tree_list))
    out_image = deepcopy(i)

    workers = []
    max_workers = 100
    step = len(tree_list) // 100

    start = time.process_time()

    for i in range(max_workers):
        if i == max_workers - 1:
            start_index = i * step
            end_index = len(tree_list) - start_index
            worker = ClusterCreator(i, t * 10, start_index, end_index)
            workers.append(worker)
            worker.start()
        else:
            start_index = i * step
            end_index = (i + 1) * step
            worker = ClusterCreator(i, t * 10, start_index, end_index)
            workers.append(worker)
            worker.start()

    for index, worker in enumerate(workers):
        worker.join()

    # create_clusters(t)
    elapsed_time = time.process_time() - start

    print('Cluster of nodes has been created in {}....'.format(elapsed_time))

    for c in clusters:
        print('Clusters[{}] = {}'.format(c, clusters[c].elements))
        apply_mean_cluster(clusters[c].idx)

    img.imsave('images/{}'.format(config['merge_result']), out_image)

    print('Saved merge version of the image....')
