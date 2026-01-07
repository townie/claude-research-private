"""
Optimization techniques for improving clustering results.

Includes:
- Boundary optimization (reassign boundary faces)
- Simulated annealing
- Small cluster merging
- High-distortion splitting
"""

import random
import math
from typing import List, Set, Dict, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np

from .mesh import Mesh
from .dual_graph import DualGraph
from .quality_metrics import compute_interior_edge_ratio, estimate_cluster_distortion


def get_boundary_faces(mesh: Mesh, cluster: Set[int], all_clusters: List[Set[int]]) -> Set[int]:
    """Get faces that are on the boundary of a cluster."""
    dual = DualGraph(mesh)
    cluster_idx = all_clusters.index(cluster)
    boundary = set()

    # Build face-to-cluster mapping
    face_to_cluster = {}
    for i, c in enumerate(all_clusters):
        for f in c:
            face_to_cluster[f] = i

    for face_id in cluster:
        for neighbor_id in dual.get_neighbors(face_id):
            if face_to_cluster.get(neighbor_id, -1) != cluster_idx:
                boundary.add(face_id)
                break

    return boundary


def optimize_boundaries(mesh: Mesh,
                        clusters: List[Set[int]],
                        max_iterations: int = 100,
                        max_distortion: float = 0.3) -> List[Set[int]]:
    """
    Iteratively reassign boundary faces to improve edge connectivity.

    Parameters:
        mesh: The mesh
        clusters: Initial clustering
        max_iterations: Maximum optimization iterations
        max_distortion: Maximum allowed distortion per cluster

    Returns:
        Optimized clusters
    """
    dual = DualGraph(mesh)
    clusters = [c.copy() for c in clusters]

    for iteration in range(max_iterations):
        improved = False

        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue

            boundary = get_boundary_faces(mesh, cluster, clusters)

            for face_id in list(boundary):
                # Find neighboring clusters
                neighbor_clusters = set()
                for neighbor_id in dual.get_neighbors(face_id):
                    for i, c in enumerate(clusters):
                        if neighbor_id in c and i != cluster_idx:
                            neighbor_clusters.add(i)

                for neighbor_idx in neighbor_clusters:
                    # Count edges in current vs potential cluster
                    current_edges = sum(
                        1 for n in dual.get_neighbors(face_id)
                        if n in clusters[cluster_idx]
                    )
                    new_edges = sum(
                        1 for n in dual.get_neighbors(face_id)
                        if n in clusters[neighbor_idx]
                    )

                    if new_edges > current_edges:
                        # Check distortion constraint
                        new_cluster = clusters[neighbor_idx] | {face_id}
                        new_dist = estimate_cluster_distortion(mesh, new_cluster)

                        old_cluster = clusters[cluster_idx] - {face_id}
                        if len(old_cluster) == 0:
                            continue

                        if new_dist <= max_distortion:
                            # Move face
                            clusters[cluster_idx].remove(face_id)
                            clusters[neighbor_idx].add(face_id)
                            improved = True
                            break

                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    # Remove empty clusters
    clusters = [c for c in clusters if len(c) > 0]
    return clusters


def merge_small_clusters(mesh: Mesh,
                         clusters: List[Set[int]],
                         min_size: int = 3,
                         max_distortion: float = 0.4) -> List[Set[int]]:
    """
    Merge clusters smaller than min_size with their best neighbor.

    Parameters:
        mesh: The mesh
        clusters: Initial clustering
        min_size: Minimum cluster size
        max_distortion: Maximum allowed distortion after merge

    Returns:
        Clusters with small ones merged
    """
    dual = DualGraph(mesh)
    clusters = [c.copy() for c in clusters]

    def are_adjacent(c1: Set[int], c2: Set[int]) -> bool:
        for f1 in c1:
            for n in dual.get_neighbors(f1):
                if n in c2:
                    return True
        return False

    def count_shared_edges(c1: Set[int], c2: Set[int]) -> int:
        count = 0
        for f1 in c1:
            for n in dual.get_neighbors(f1):
                if n in c2:
                    count += 1
        return count // 2

    merged = True
    while merged:
        merged = False

        for i, cluster in enumerate(clusters[:]):
            if len(cluster) >= min_size:
                continue

            # Find best neighbor
            best_neighbor_idx = None
            best_score = -1

            for j, other in enumerate(clusters):
                if i == j or not are_adjacent(cluster, other):
                    continue

                merged_cluster = cluster | other
                distortion = estimate_cluster_distortion(mesh, merged_cluster)

                if distortion <= max_distortion:
                    score = count_shared_edges(cluster, other)
                    if score > best_score:
                        best_score = score
                        best_neighbor_idx = j

            if best_neighbor_idx is not None:
                # Merge
                clusters[best_neighbor_idx].update(cluster)
                clusters.remove(cluster)
                merged = True
                break

    return clusters


def split_high_distortion_clusters(mesh: Mesh,
                                    clusters: List[Set[int]],
                                    max_distortion: float = 0.3) -> List[Set[int]]:
    """
    Split clusters that exceed distortion threshold.

    Parameters:
        mesh: The mesh
        clusters: Initial clustering
        max_distortion: Maximum allowed distortion

    Returns:
        Clusters with high-distortion ones split
    """
    dual = DualGraph(mesh)
    result = []

    for cluster in clusters:
        distortion = estimate_cluster_distortion(mesh, cluster)

        if distortion <= max_distortion or len(cluster) <= 3:
            result.append(cluster)
            continue

        # Need to split - find best cut edge
        sub_clusters = split_cluster(mesh, cluster, max_distortion)
        result.extend(sub_clusters)

    return result


def split_cluster(mesh: Mesh, cluster: Set[int], max_distortion: float) -> List[Set[int]]:
    """Split a single cluster into two or more pieces."""
    dual = DualGraph(mesh)

    if len(cluster) <= 3:
        return [cluster]

    # Find edge to cut (highest curvature change)
    best_edge = None
    best_curvature = -1

    faces_list = list(cluster)
    for f1 in faces_list:
        for f2 in dual.get_neighbors(f1):
            if f2 not in cluster:
                continue

            # Curvature = 1 - dot(normals)
            n1 = mesh.get_face_normal(f1)
            n2 = mesh.get_face_normal(f2)
            curvature = 1 - np.dot(n1, n2)

            if curvature > best_curvature:
                best_curvature = curvature
                best_edge = (f1, f2)

    if best_edge is None:
        return [cluster]

    # Split by BFS from each side of the cut edge
    f1, f2 = best_edge

    def bfs_cluster(start: int, exclude: int) -> Set[int]:
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in dual.get_neighbors(current):
                if neighbor in cluster and neighbor not in visited:
                    if not (current == start and neighbor == exclude):
                        visited.add(neighbor)
                        queue.append(neighbor)
        return visited

    cluster1 = bfs_cluster(f1, f2)
    cluster2 = cluster - cluster1

    if len(cluster2) == 0:
        return [cluster]

    # Recursively split if still too distorted
    results = []
    for sub in [cluster1, cluster2]:
        if estimate_cluster_distortion(mesh, sub) > max_distortion and len(sub) > 3:
            results.extend(split_cluster(mesh, sub, max_distortion))
        else:
            results.append(sub)

    return results


def simulated_annealing(mesh: Mesh,
                        clusters: List[Set[int]],
                        initial_temp: float = 1.0,
                        cooling_rate: float = 0.995,
                        min_temp: float = 0.01,
                        iterations_per_temp: int = 50,
                        max_distortion: float = 0.3) -> List[Set[int]]:
    """
    Use simulated annealing to optimize cluster assignments.

    Parameters:
        mesh: The mesh
        clusters: Initial clustering
        initial_temp: Starting temperature
        cooling_rate: Temperature decay rate per iteration
        min_temp: Stop when temperature drops below this
        iterations_per_temp: Moves to try at each temperature
        max_distortion: Maximum allowed distortion

    Returns:
        Optimized clusters
    """
    dual = DualGraph(mesh)

    def compute_score(clusters: List[Set[int]]) -> float:
        """Score to maximize."""
        ie_ratio = compute_interior_edge_ratio(mesh, clusters)

        # Penalty for high distortion
        distortions = [estimate_cluster_distortion(mesh, c) for c in clusters]
        max_dist = max(distortions) if distortions else 0
        dist_penalty = max(0, max_dist - max_distortion) * 2

        # Bonus for balanced sizes
        if clusters:
            sizes = [len(c) for c in clusters]
            balance = 1.0 - (np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0
        else:
            balance = 0

        return ie_ratio - dist_penalty + 0.1 * balance

    def generate_neighbor(clusters: List[Set[int]]) -> Optional[List[Set[int]]]:
        """Generate a neighbor solution by moving one boundary face."""
        new_clusters = [c.copy() for c in clusters]

        # Find all possible moves
        moves = []
        for i, cluster in enumerate(new_clusters):
            boundary = get_boundary_faces(mesh, cluster, new_clusters)
            for face_id in boundary:
                for j, other in enumerate(new_clusters):
                    if i == j:
                        continue
                    for n in dual.get_neighbors(face_id):
                        if n in other:
                            moves.append((face_id, i, j))
                            break

        if not moves:
            return None

        # Pick random move
        face_id, from_idx, to_idx = random.choice(moves)

        # Validate move
        if len(new_clusters[from_idx]) <= 1:
            return None

        new_target = new_clusters[to_idx] | {face_id}
        if estimate_cluster_distortion(mesh, new_target) > max_distortion * 1.5:
            return None

        # Apply move
        new_clusters[from_idx].remove(face_id)
        new_clusters[to_idx].add(face_id)

        # Remove empty clusters
        new_clusters = [c for c in new_clusters if len(c) > 0]

        return new_clusters

    # Initialize
    current = [c.copy() for c in clusters]
    current_score = compute_score(current)

    best = [c.copy() for c in current]
    best_score = current_score

    temp = initial_temp

    while temp > min_temp:
        for _ in range(iterations_per_temp):
            neighbor = generate_neighbor(current)

            if neighbor is None:
                continue

            neighbor_score = compute_score(neighbor)
            delta = neighbor_score - current_score

            # Accept or reject
            if delta > 0:
                current = neighbor
                current_score = neighbor_score

                if current_score > best_score:
                    best = [c.copy() for c in current]
                    best_score = current_score
            else:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current = neighbor
                    current_score = neighbor_score

        temp *= cooling_rate

    return best


def full_optimization_pipeline(mesh: Mesh,
                               clusters: List[Set[int]],
                               min_cluster_size: int = 3,
                               max_distortion: float = 0.3,
                               use_annealing: bool = True,
                               verbose: bool = False) -> List[Set[int]]:
    """
    Run full optimization pipeline.

    Phases:
    1. Merge small clusters
    2. Split high-distortion clusters
    3. Boundary optimization
    4. Simulated annealing (optional)
    5. Final cleanup
    """
    from .quality_metrics import compute_interior_edge_ratio

    if verbose:
        initial_ratio = compute_interior_edge_ratio(mesh, clusters)
        print(f"Initial: {len(clusters)} clusters, {initial_ratio:.1%} interior edges")

    # Phase 1: Merge small clusters
    clusters = merge_small_clusters(mesh, clusters, min_cluster_size, max_distortion)
    if verbose:
        ratio = compute_interior_edge_ratio(mesh, clusters)
        print(f"After merge: {len(clusters)} clusters, {ratio:.1%} interior edges")

    # Phase 2: Split high-distortion
    clusters = split_high_distortion_clusters(mesh, clusters, max_distortion)
    if verbose:
        ratio = compute_interior_edge_ratio(mesh, clusters)
        print(f"After split: {len(clusters)} clusters, {ratio:.1%} interior edges")

    # Phase 3: Boundary optimization
    clusters = optimize_boundaries(mesh, clusters, max_iterations=100, max_distortion=max_distortion)
    if verbose:
        ratio = compute_interior_edge_ratio(mesh, clusters)
        print(f"After boundary opt: {len(clusters)} clusters, {ratio:.1%} interior edges")

    # Phase 4: Simulated annealing
    if use_annealing:
        clusters = simulated_annealing(
            mesh, clusters,
            initial_temp=0.5,
            cooling_rate=0.99,
            iterations_per_temp=30,
            max_distortion=max_distortion
        )
        if verbose:
            ratio = compute_interior_edge_ratio(mesh, clusters)
            print(f"After annealing: {len(clusters)} clusters, {ratio:.1%} interior edges")

    # Phase 5: Final cleanup
    clusters = merge_small_clusters(mesh, clusters, min_size=2, max_distortion=max_distortion * 1.2)
    if verbose:
        ratio = compute_interior_edge_ratio(mesh, clusters)
        print(f"Final: {len(clusters)} clusters, {ratio:.1%} interior edges")

    return clusters
