#!/usr/bin/python3
import util

class HacNode:

    def __init__(self, points, children, height):
        """
        Constructor of HacNode, which represents a node in the HAC dendrogram
        """

        if len(points) > 0 and len(children) > 0:
            raise Exception("Errro: HacNode can only have children or points, "
                "but not both")
        
        if len(points) == 0 and len(children) == 0:
            raise Exception("Error: Either children or points must be non-empty")

        if len(children) != 0 and len(children) != 2:
            raise Exception("Error: Children must have length 0 or 2")

        if height < 0:
            raise Exception("Error: Height must be non-negative.")

        # Children are child HacNodes
        self.children = children
        # Height is the min distance between the two child
        self.height = height

        # The points of a node is the points in all it's children. 
        # It is more memory-intensive, but save time on tree traversal when 
        # calculating distances

        # Points are the indices of a point in the original dataset
        self.point_indices = []
        if len(self.children) > 0:            
            for child in self.children:
                self.point_indices += child.point_indices
        else:
            self.point_indices = points
        
    def single_linkage_distance(self, dist_matrix, another_hac_node):
        """
        Calculate the min distance between the current node and another_hac_node
        using the single linkage method. Namely, the min distances between the
        closest two point in the node.
        """
        min_dist = float("inf")
        for i in self.point_indices:
            for j in another_hac_node.point_indices:
                dist = dist_matrix[i][j]
                min_dist = min(min_dist, dist)
        return min_dist

    def get_clusters_under_height(self, h):
        """
        Get the clusters under the given height in index representation.
        """
        clusters = []
        if self.height > h:
            left_clusters = self.children[0].get_clusters_under_height(h)
            right_clusters = self.children[1].get_clusters_under_height(h)
            for cluster in left_clusters + right_clusters:
                clusters.append(cluster)
        else:
            clusters.append(self.point_indices)
        return clusters


def __construct_hac_dendrogram(data, dist_matrix, k):
    """
    Construct a HAC dendrogram using the given data and dist_matrics
    """
    nodes = []
    # Initialize each point in the dataset to be its own node
    # Using point index because it's more memory efficient
    for point_index in range(len(data)):
        nodes.append(HacNode([point_index], [], 0))

    while len(nodes) > k:
        size = len(nodes)
        cur_min_dist = float("inf")
        indices_to_merge = (-1, -1)

        # Find the 2 clusters that has the min distance to merge
        for i in range(size):
            for j in range(size):
                if i != j:
                    dist = nodes[i].single_linkage_distance(dist_matrix, nodes[j])
                    if dist < cur_min_dist:
                        cur_min_dist = dist
                        indices_to_merge = (i, j)

        # Merge the two nodes that has the min distance
        a = nodes[indices_to_merge[0]]
        b = nodes[indices_to_merge[1]]
        nodes.append(HacNode([], [a, b], cur_min_dist))
        # Delete the higher merged index first so it doesn't change impact the
        # lower index's position
        del nodes[max(indices_to_merge[0], indices_to_merge[1])]
        # Delete the lower merged index
        del nodes[min(indices_to_merge[0], indices_to_merge[1])]

    return nodes # root of dendrogram

def __get_distance_matrix(data):
    """
    Create a distance metrix to help accelerate the algorithm. A distance matrix
    contains the distance between any two points in the dataset
    """

    dist_matrix = []
    size = len(data)
    for i in range(size):
        row = []
        for j in range(size):
            dist = 0
            if i != j:
                dist = util.distance(data[i], data[j])
            row.append(dist)
        dist_matrix.append(row)
    return dist_matrix

def get_clusters(data, k):
    """
    Perform single-linkage Hierarchical Agglomerative Clustering
    """

    dist_matrix = __get_distance_matrix(data)
    dendrogram = __construct_hac_dendrogram(data, dist_matrix, k)
    clusters_by_index = []
    for node in dendrogram:
        clusters_by_index.append(node.point_indices)
    return clusters_by_index

# Tests
# if __name__ == '__main__':
#     a = [1, 3]
#     b = [1, 4]
#     c = [5, 2]
#     d = [5, 1]
#     e = [2, 2]
#     f = [7, 2]
#     data = [a, b, c, d, e, f]
#     clusters = get_clusters(data, 3)
#     print(clusters)