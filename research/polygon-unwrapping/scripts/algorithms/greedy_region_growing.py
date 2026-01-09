"""
Greedy Region Growing Algorithm for Polygon Unwrapping.

This algorithm grows clusters greedily from seed faces, always adding
the face that maximizes the number of shared edges with the current cluster.

Good for: Fast clustering, intuitive results
Trade-off: May not find globally optimal solution
"""

from typing import List, Set, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np

from scripts.utils.mesh import Mesh
from scripts.utils.dual_graph import DualGraph


@dataclass
class ClusterStats:
    """Statistics for a single cluster."""
    faces: Set[int] = field(default_factory=set)
    interior_edges: Set[Tuple[int, int]] = field(default_factory=set)
    boundary_edges: Set[Tuple[int, int]] = field(default_factory=set)
    distortion_estimate: float = 0.0


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


class GreedyRegionGrower:
    """
    Clusterer using greedy region growing.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum allowed distortion per cluster
        min_cluster_size: Minimum faces per cluster
    """

    def __init__(self, mesh: Mesh,
                 max_distortion: float = 0.3,
                 min_cluster_size: int = 3):
        self.mesh = mesh
        self.max_distortion = max_distortion
        self.min_cluster_size = min_cluster_size

        self.dual_graph = DualGraph(mesh)

    def _get_face_edges(self, face_id: int) -> Set[Tuple[int, int]]:
        """Get all edges of a face."""
        return set(self.mesh.faces[face_id].get_edges())

    def _count_shared_edges(self, face_id: int, cluster_edges: Set[Tuple[int, int]]) -> int:
        """Count edges shared between a face and a cluster."""
        face_edges = self._get_face_edges(face_id)
        return len(face_edges & cluster_edges)

    def _estimate_distortion(self, faces: Set[int]) -> float:
        """Estimate distortion for a cluster."""
        if len(faces) <= 2:
            return 0.0

        # Heuristic: based on total normal variation
        normals = [self.mesh.get_face_normal(f) for f in faces]
        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)

        # Variance from average
        deviations = [1 - np.dot(n, avg_normal) for n in normals]
        avg_deviation = np.mean(deviations)

        return min(1.0, avg_deviation * 2)

    def _grow_cluster(self, seed: int, available: Set[int]) -> ClusterStats:
        """Grow a cluster from a seed face."""
        cluster = ClusterStats()
        cluster.faces.add(seed)
        cluster.boundary_edges = self._get_face_edges(seed)
        available.remove(seed)

        while True:
            # Find all adjacent available faces
            candidates = []

            for face_id in cluster.faces:
                for neighbor_id in self.dual_graph.get_neighbors(face_id):
                    if neighbor_id in available:
                        shared = self._count_shared_edges(neighbor_id, cluster.boundary_edges)
                        candidates.append((neighbor_id, shared))

            if not candidates:
                break

            # Sort by shared edges (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Try to add best candidate
            added = False
            for candidate_id, shared_count in candidates:
                if candidate_id not in available:
                    continue

                # Check distortion
                new_faces = cluster.faces | {candidate_id}
                new_distortion = self._estimate_distortion(new_faces)

                if new_distortion <= self.max_distortion:
                    # Add face to cluster
                    cluster.faces.add(candidate_id)
                    available.remove(candidate_id)

                    # Update edges
                    candidate_edges = self._get_face_edges(candidate_id)
                    shared_edges = candidate_edges & cluster.boundary_edges

                    cluster.interior_edges |= shared_edges
                    cluster.boundary_edges = (
                        (cluster.boundary_edges | candidate_edges) -
                        cluster.interior_edges -
                        shared_edges
                    )
                    cluster.distortion_estimate = new_distortion

                    added = True
                    break

            if not added:
                break

        return cluster

    def _select_seed(self, available: Set[int]) -> int:
        """Select the best seed face for a new cluster."""
        # Choose face with most available neighbors (for better clustering)
        best_seed = None
        best_neighbor_count = -1

        for face_id in available:
            neighbor_count = sum(
                1 for n in self.dual_graph.get_neighbors(face_id)
                if n in available
            )
            if neighbor_count > best_neighbor_count:
                best_neighbor_count = neighbor_count
                best_seed = face_id

        return best_seed

    def cluster(self) -> ClusterResult:
        """
        Run the greedy region growing algorithm.

        Returns ClusterResult with clusters and statistics.
        """
        available = set(self.mesh.faces.keys())
        clusters = []
        total_interior = 0

        while available:
            seed = self._select_seed(available)
            cluster_stats = self._grow_cluster(seed, available)
            clusters.append(cluster_stats.faces)
            total_interior += len(cluster_stats.interior_edges)

        # Count cut edges
        total_edges = len(self.dual_graph.edges)
        cut_edges = total_edges - total_interior

        return ClusterResult(
            clusters=clusters,
            interior_edges=total_interior,
            cut_edges=cut_edges
        )


def greedy_region_growing(mesh: Mesh,
                          max_distortion: float = 0.3,
                          min_cluster_size: int = 3) -> ClusterResult:
    """
    Convenience function to run greedy region growing.

    Parameters:
        mesh: The mesh to cluster
        max_distortion: Maximum distortion per cluster
        min_cluster_size: Minimum faces per cluster

    Returns:
        ClusterResult with clusters and statistics
    """
    grower = GreedyRegionGrower(mesh, max_distortion, min_cluster_size)
    return grower.cluster()


class PriorityGreedyGrower(GreedyRegionGrower):
    """
    Enhanced greedy grower with priority-based candidate selection.

    Uses a priority queue for efficient candidate management.
    """

    def __init__(self, mesh: Mesh,
                 max_distortion: float = 0.3,
                 edge_weight_func: Optional[Callable] = None):
        super().__init__(mesh, max_distortion)
        self.edge_weight_func = edge_weight_func

    def _get_candidate_score(self, face_id: int, cluster: ClusterStats) -> float:
        """
        Score a candidate face for addition to cluster.

        Higher score = better candidate.
        """
        # Base score: number of shared edges
        shared = self._count_shared_edges(face_id, cluster.boundary_edges)

        # Bonus for edge weights if available
        if self.edge_weight_func:
            weight_sum = 0.0
            for cluster_face in cluster.faces:
                edge = self.dual_graph.get_edge(face_id, cluster_face)
                if edge:
                    weight_sum += edge.weight
            shared = shared * (1 + weight_sum)

        return shared

    def _grow_cluster(self, seed: int, available: Set[int]) -> ClusterStats:
        """Grow cluster using priority queue."""
        import heapq

        cluster = ClusterStats()
        cluster.faces.add(seed)
        cluster.boundary_edges = self._get_face_edges(seed)
        available.remove(seed)

        # Priority queue: (-score, face_id)
        pq = []
        added_to_pq = {seed}

        # Add initial neighbors
        for neighbor_id in self.dual_graph.get_neighbors(seed):
            if neighbor_id in available:
                score = self._get_candidate_score(neighbor_id, cluster)
                heapq.heappush(pq, (-score, neighbor_id))
                added_to_pq.add(neighbor_id)

        while pq:
            neg_score, candidate_id = heapq.heappop(pq)

            if candidate_id not in available:
                continue

            # Check distortion
            new_faces = cluster.faces | {candidate_id}
            new_distortion = self._estimate_distortion(new_faces)

            if new_distortion > self.max_distortion:
                continue

            # Add face to cluster
            cluster.faces.add(candidate_id)
            available.remove(candidate_id)

            # Update edges
            candidate_edges = self._get_face_edges(candidate_id)
            shared_edges = candidate_edges & cluster.boundary_edges
            cluster.interior_edges |= shared_edges
            cluster.boundary_edges = (
                (cluster.boundary_edges | candidate_edges) -
                cluster.interior_edges -
                shared_edges
            )
            cluster.distortion_estimate = new_distortion

            # Add new neighbors to queue
            for neighbor_id in self.dual_graph.get_neighbors(candidate_id):
                if neighbor_id in available and neighbor_id not in added_to_pq:
                    score = self._get_candidate_score(neighbor_id, cluster)
                    heapq.heappush(pq, (-score, neighbor_id))
                    added_to_pq.add(neighbor_id)

        return cluster


# Standalone script functionality
if __name__ == "__main__":
    from ..utils.mesh import create_cube, create_simple_dog

    print("=" * 60)
    print("Greedy Region Growing Clustering Demo")
    print("=" * 60)

    # Test on cube
    print("\n--- Cube ---")
    cube = create_cube()
    print(f"Mesh: {cube}")

    result = greedy_region_growing(cube, max_distortion=0.5)

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

    result = greedy_region_growing(dog, max_distortion=0.3)

    print(f"Clusters: {len(result.clusters)}")
    print(f"Interior edges: {result.interior_edges}")
    print(f"Cut edges: {result.cut_edges}")
    print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")

    for i, cluster in enumerate(result.clusters):
        print(f"  Cluster {i}: {len(cluster)} faces")
