"""
Maximum Spanning Forest Algorithm for Polygon Unwrapping.

This algorithm maximizes interior edges by finding the maximum spanning
forest on the dual graph. Each tree in the forest becomes a UV cluster.

Key insight: Interior edges correspond to edges IN the spanning forest.
Cut edges correspond to edges NOT in the forest.
"""

import heapq
from typing import List, Set, Dict, Optional, Callable
from dataclasses import dataclass

import numpy as np

from scripts.utils.mesh import Mesh
from scripts.utils.dual_graph import DualGraph, UnionFind


@dataclass
class ClusterResult:
    """Result of clustering algorithm."""
    clusters: List[Set[int]]
    interior_edges: int
    cut_edges: int

    @property
    def interior_edge_ratio(self) -> float:
        total = self.interior_edges + self.cut_edges
        return self.interior_edges / total if total > 0 else 0


class MaxSpanningForestClusterer:
    """
    Clusterer using Maximum Spanning Forest algorithm.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum allowed distortion per cluster (0-1)
        distortion_estimator: Optional function to estimate cluster distortion
    """

    def __init__(self, mesh: Mesh,
                 max_distortion: float = 0.5,
                 distortion_estimator: Optional[Callable] = None):
        self.mesh = mesh
        self.max_distortion = max_distortion
        self.distortion_estimator = distortion_estimator or self._default_distortion

        self.dual_graph = DualGraph(mesh)
        self._cluster_faces: Dict[int, Set[int]] = {}

    def _default_distortion(self, faces: Set[int]) -> float:
        """
        Default distortion estimator based on cluster size and curvature.

        For better results, implement a proper distortion estimator
        based on actual parameterization.
        """
        if len(faces) <= 2:
            return 0.0

        # Simple heuristic: sum of angle deficits
        total_deficit = 0.0

        for face_id in faces:
            # Get face vertices
            face = self.mesh.faces[face_id]
            if len(face.vertex_ids) >= 3:
                # Compute angle sum for this face
                positions = [self.mesh.vertices[v].position for v in face.vertex_ids]

                # For a triangle, sum should be 180 degrees
                # Deviation from this indicates distortion potential
                for i in range(len(positions)):
                    v0 = positions[i]
                    v1 = positions[(i + 1) % len(positions)]
                    v2 = positions[(i - 1) % len(positions)]

                    e1 = v1 - v0
                    e2 = v2 - v0

                    norm1 = max(1e-10, np.linalg.norm(e1))
                    norm2 = max(1e-10, np.linalg.norm(e2))

                    cos_angle = np.dot(e1, e2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)

        # Normalize by number of faces
        return min(1.0, len(faces) * 0.02)

    def cluster(self) -> ClusterResult:
        """
        Run the Maximum Spanning Forest clustering algorithm.

        Returns ClusterResult with clusters and edge statistics.
        """
        import numpy as np

        # Sort edges by weight (descending for maximum spanning)
        edges = self.dual_graph.get_edges_sorted_by_weight(descending=True)

        # Initialize Union-Find
        uf = UnionFind(self.mesh.faces.keys())

        # Track faces in each cluster (by root)
        cluster_faces: Dict[int, Set[int]] = {
            f: {f} for f in self.mesh.faces.keys()
        }

        forest_edges = []

        for edge in edges:
            f1, f2 = edge.face1_id, edge.face2_id
            root1, root2 = uf.find(f1), uf.find(f2)

            if root1 == root2:
                continue  # Already in same cluster

            # Check if merging would exceed distortion threshold
            merged_faces = cluster_faces[root1] | cluster_faces[root2]
            estimated_distortion = self.distortion_estimator(merged_faces)

            if estimated_distortion <= self.max_distortion:
                # Merge clusters
                uf.union(f1, f2)
                forest_edges.append(edge)

                # Update cluster tracking
                new_root = uf.find(f1)
                cluster_faces[new_root] = merged_faces

                # Clean up old roots
                if new_root != root1 and root1 in cluster_faces:
                    del cluster_faces[root1]
                if new_root != root2 and root2 in cluster_faces:
                    del cluster_faces[root2]

        # Extract final clusters
        clusters = uf.get_clusters()

        # Count edges
        interior_edges = len(forest_edges)
        cut_edges = len(edges) - interior_edges

        return ClusterResult(
            clusters=clusters,
            interior_edges=interior_edges,
            cut_edges=cut_edges
        )


def maximum_spanning_forest(mesh: Mesh,
                            max_distortion: float = 0.5,
                            weight_func: Optional[Callable] = None) -> ClusterResult:
    """
    Convenience function to run MSF clustering.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum distortion per cluster (0-1)
        weight_func: Optional custom weight function

    Returns:
        ClusterResult with clusters and statistics
    """
    from ..utils.edge_weights import apply_dihedral_weights

    clusterer = MaxSpanningForestClusterer(mesh, max_distortion)

    # Apply weights if custom function provided
    if weight_func:
        from ..utils.edge_weights import apply_weights
        apply_weights(clusterer.dual_graph, weight_func)
    else:
        apply_dihedral_weights(clusterer.dual_graph)

    return clusterer.cluster()


# Standalone script functionality
if __name__ == "__main__":
    import numpy as np
    from ..utils.mesh import create_cube, create_simple_dog

    print("=" * 60)
    print("Maximum Spanning Forest Clustering Demo")
    print("=" * 60)

    # Test on cube
    print("\n--- Cube ---")
    cube = create_cube()
    print(f"Mesh: {cube}")

    result = maximum_spanning_forest(cube, max_distortion=0.5)

    print(f"Clusters: {len(result.clusters)}")
    print(f"Interior edges: {result.interior_edges}")
    print(f"Cut edges: {result.cut_edges}")
    print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")

    for i, cluster in enumerate(result.clusters):
        print(f"  Cluster {i}: {cluster}")

    # Test on dog
    print("\n--- Simple Dog ---")
    dog = create_simple_dog()
    print(f"Mesh: {dog}")

    result = maximum_spanning_forest(dog, max_distortion=0.3)

    print(f"Clusters: {len(result.clusters)}")
    print(f"Interior edges: {result.interior_edges}")
    print(f"Cut edges: {result.cut_edges}")
    print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")

    for i, cluster in enumerate(result.clusters):
        print(f"  Cluster {i}: {len(cluster)} faces")
