"""
Hierarchical Agglomerative Clustering for Polygon Unwrapping.

Bottom-up approach: starts with each face as its own cluster,
then iteratively merges clusters based on edge connectivity and distortion.

Good for: Finding natural cluster boundaries, controllable cluster count
Trade-off: O(n² log n) complexity
"""

import heapq
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import sys
sys.path.insert(0, '..')

from ..utils.mesh import Mesh
from ..utils.dual_graph import DualGraph


@dataclass
class Cluster:
    """A cluster of faces."""
    id: int
    faces: Set[int] = field(default_factory=set)
    adjacent_clusters: Set[int] = field(default_factory=set)
    distortion_estimate: float = 0.0

    def __hash__(self):
        return hash(self.id)


@dataclass
class ClusterResult:
    """Result of clustering algorithm."""
    clusters: List[Set[int]]
    interior_edges: int
    cut_edges: int
    merge_history: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def interior_edge_ratio(self) -> float:
        total = self.interior_edges + self.cut_edges
        return self.interior_edges / total if total > 0 else 0


class HierarchicalClusterer:
    """
    Hierarchical agglomerative clustering for mesh faces.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum distortion per cluster
        target_clusters: Target number of clusters (optional)
    """

    def __init__(self, mesh: Mesh,
                 max_distortion: float = 0.3,
                 target_clusters: Optional[int] = None):
        self.mesh = mesh
        self.max_distortion = max_distortion
        self.target_clusters = target_clusters

        self.dual_graph = DualGraph(mesh)
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
        self.merge_history = []

    def _initialize(self):
        """Create initial clusters (one per face)."""
        self.clusters = {}
        self.next_cluster_id = 0

        face_to_cluster = {}

        for face_id in self.mesh.faces:
            cluster = Cluster(
                id=self.next_cluster_id,
                faces={face_id},
                distortion_estimate=0.0
            )
            self.clusters[cluster.id] = cluster
            face_to_cluster[face_id] = cluster.id
            self.next_cluster_id += 1

        # Compute adjacencies
        for face_id in self.mesh.faces:
            my_cluster = face_to_cluster[face_id]
            for neighbor_id in self.dual_graph.get_neighbors(face_id):
                neighbor_cluster = face_to_cluster[neighbor_id]
                if my_cluster != neighbor_cluster:
                    self.clusters[my_cluster].adjacent_clusters.add(neighbor_cluster)

    def _count_shared_edges(self, c1: Cluster, c2: Cluster) -> int:
        """Count edges shared between two clusters."""
        count = 0
        for f1 in c1.faces:
            for neighbor in self.dual_graph.get_neighbors(f1):
                if neighbor in c2.faces:
                    count += 1
        return count // 2  # Each edge counted twice

    def _estimate_merged_distortion(self, c1: Cluster, c2: Cluster) -> float:
        """Estimate distortion of merged cluster."""
        merged_faces = c1.faces | c2.faces

        if len(merged_faces) <= 2:
            return 0.0

        # Use normal variance as distortion proxy
        normals = [self.mesh.get_face_normal(f) for f in merged_faces]
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-10:
            avg_normal = avg_normal / norm

        deviations = [1 - np.dot(n, avg_normal) for n in normals]
        return min(1.0, np.mean(deviations) * 2)

    def _compute_merge_score(self, c1_id: int, c2_id: int) -> Tuple[float, float]:
        """
        Compute score for merging two clusters.

        Returns (edge_gain, distortion).
        Higher edge_gain and lower distortion = better merge.
        """
        c1, c2 = self.clusters[c1_id], self.clusters[c2_id]

        edge_gain = self._count_shared_edges(c1, c2)
        distortion = self._estimate_merged_distortion(c1, c2)

        return edge_gain, distortion

    def _merge_clusters(self, c1_id: int, c2_id: int) -> int:
        """Merge two clusters and return new cluster ID."""
        c1, c2 = self.clusters[c1_id], self.clusters[c2_id]

        # Create merged cluster
        new_cluster = Cluster(
            id=self.next_cluster_id,
            faces=c1.faces | c2.faces,
            adjacent_clusters=(c1.adjacent_clusters | c2.adjacent_clusters) - {c1_id, c2_id},
            distortion_estimate=self._estimate_merged_distortion(c1, c2)
        )
        self.next_cluster_id += 1

        # Update adjacencies of neighboring clusters
        for adj_id in list(new_cluster.adjacent_clusters):
            if adj_id in self.clusters:
                adj = self.clusters[adj_id]
                adj.adjacent_clusters.discard(c1_id)
                adj.adjacent_clusters.discard(c2_id)
                adj.adjacent_clusters.add(new_cluster.id)

        # Record merge
        self.merge_history.append((c1_id, c2_id, new_cluster.id))

        # Remove old clusters, add new
        del self.clusters[c1_id]
        del self.clusters[c2_id]
        self.clusters[new_cluster.id] = new_cluster

        return new_cluster.id

    def cluster(self) -> ClusterResult:
        """
        Run hierarchical clustering.

        Returns ClusterResult with clusters and statistics.
        """
        self._initialize()
        self.merge_history = []

        # Priority queue: (-edge_gain, distortion, c1_id, c2_id)
        # Negative edge_gain for max-heap behavior
        merge_queue = []

        def add_merge_candidates(cluster_id: int):
            """Add potential merges for a cluster to the queue."""
            cluster = self.clusters.get(cluster_id)
            if not cluster:
                return

            for adj_id in cluster.adjacent_clusters:
                if adj_id in self.clusters:
                    edge_gain, distortion = self._compute_merge_score(cluster_id, adj_id)
                    if distortion <= self.max_distortion:
                        heapq.heappush(merge_queue, (-edge_gain, distortion, cluster_id, adj_id))

        # Initialize queue with all potential merges
        for cluster_id in list(self.clusters.keys()):
            add_merge_candidates(cluster_id)

        # Merge until stopping condition
        while merge_queue:
            # Check stopping conditions
            if self.target_clusters and len(self.clusters) <= self.target_clusters:
                break

            neg_gain, distortion, c1_id, c2_id = heapq.heappop(merge_queue)

            # Check if clusters still exist
            if c1_id not in self.clusters or c2_id not in self.clusters:
                continue

            # Verify distortion is still acceptable
            actual_gain, actual_distortion = self._compute_merge_score(c1_id, c2_id)
            if actual_distortion > self.max_distortion:
                continue

            # Perform merge
            new_id = self._merge_clusters(c1_id, c2_id)
            add_merge_candidates(new_id)

        # Compute edge statistics
        interior_edges = 0
        for c in self.clusters.values():
            faces_list = list(c.faces)
            for i, f1 in enumerate(faces_list):
                for f2 in faces_list[i+1:]:
                    if f2 in self.dual_graph.get_neighbors(f1):
                        interior_edges += 1

        total_edges = len(self.dual_graph.edges)
        cut_edges = total_edges - interior_edges

        return ClusterResult(
            clusters=[c.faces for c in self.clusters.values()],
            interior_edges=interior_edges,
            cut_edges=cut_edges,
            merge_history=self.merge_history
        )


def hierarchical_clustering(mesh: Mesh,
                            max_distortion: float = 0.3,
                            target_clusters: Optional[int] = None) -> ClusterResult:
    """
    Convenience function to run hierarchical clustering.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum distortion per cluster
        target_clusters: Stop when this many clusters reached

    Returns:
        ClusterResult with clusters and statistics
    """
    clusterer = HierarchicalClusterer(mesh, max_distortion, target_clusters)
    return clusterer.cluster()


class DendrogramNode:
    """Node in a dendrogram (for visualization)."""

    def __init__(self, id: int, faces: Set[int], height: float = 0):
        self.id = id
        self.faces = faces
        self.height = height
        self.left: Optional['DendrogramNode'] = None
        self.right: Optional['DendrogramNode'] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_dendrogram(merge_history: List[Tuple[int, int, int]],
                     initial_faces: Dict[int, Set[int]]) -> Dict[int, DendrogramNode]:
    """
    Build a dendrogram from merge history.

    Returns mapping from cluster ID to DendrogramNode.
    """
    nodes = {}

    # Create leaf nodes
    for cluster_id, faces in initial_faces.items():
        nodes[cluster_id] = DendrogramNode(cluster_id, faces, height=0)

    # Process merges
    for height, (c1_id, c2_id, new_id) in enumerate(merge_history, 1):
        left = nodes.get(c1_id)
        right = nodes.get(c2_id)

        if left and right:
            merged_faces = left.faces | right.faces
            new_node = DendrogramNode(new_id, merged_faces, height=height)
            new_node.left = left
            new_node.right = right
            nodes[new_id] = new_node

    return nodes


# Standalone script functionality
if __name__ == "__main__":
    from ..utils.mesh import create_cube, create_simple_dog

    print("=" * 60)
    print("Hierarchical Clustering Demo")
    print("=" * 60)

    # Test on cube
    print("\n--- Cube ---")
    cube = create_cube()
    print(f"Mesh: {cube}")

    result = hierarchical_clustering(cube, max_distortion=0.5, target_clusters=2)

    print(f"Clusters: {len(result.clusters)}")
    print(f"Interior edges: {result.interior_edges}")
    print(f"Cut edges: {result.cut_edges}")
    print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")
    print(f"Merge steps: {len(result.merge_history)}")

    for i, cluster in enumerate(result.clusters):
        print(f"  Cluster {i}: {cluster}")

    # Test on dog
    print("\n--- Simple Dog ---")
    dog = create_simple_dog()
    print(f"Mesh: {dog}")

    result = hierarchical_clustering(dog, max_distortion=0.3, target_clusters=5)

    print(f"Clusters: {len(result.clusters)}")
    print(f"Interior edges: {result.interior_edges}")
    print(f"Cut edges: {result.cut_edges}")
    print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")

    for i, cluster in enumerate(result.clusters):
        print(f"  Cluster {i}: {len(cluster)} faces")
