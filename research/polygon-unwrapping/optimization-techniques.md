# Optimization Techniques for Polygon Unwrapping

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Overview

After initial clustering, several optimization techniques can improve edge connectivity, reduce distortion, and enhance UV quality. This document covers post-processing and refinement strategies.

---

## 1. Cluster Boundary Optimization

### 1.1 Boundary Face Reassignment

Move faces between clusters to improve edge connectivity.

```python
def optimize_boundary_faces(mesh, clusters, max_iterations=100):
    """
    Iteratively reassign boundary faces to improve edge count.
    """
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for cluster_idx, cluster in enumerate(clusters):
            boundary_faces = get_boundary_faces(mesh, cluster, clusters)

            for face_id in boundary_faces:
                # Find neighboring clusters
                neighbor_clusters = get_neighbor_cluster_ids(
                    mesh, face_id, cluster_idx, clusters
                )

                for neighbor_idx in neighbor_clusters:
                    # Calculate edge gain if we move this face
                    current_edges = count_interior_edges_for_face(
                        mesh, face_id, cluster_idx, clusters
                    )
                    new_edges = count_interior_edges_for_face(
                        mesh, face_id, neighbor_idx, clusters
                    )

                    # Check distortion constraint
                    can_move = check_distortion_after_move(
                        mesh, face_id, cluster_idx, neighbor_idx, clusters
                    )

                    if new_edges > current_edges and can_move:
                        # Move face to neighbor cluster
                        clusters[cluster_idx].remove(face_id)
                        clusters[neighbor_idx].add(face_id)
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    return clusters


def get_boundary_faces(mesh, cluster, all_clusters):
    """Get faces that are on the boundary of a cluster."""
    boundary = set()
    cluster_id = all_clusters.index(cluster)

    for face_id in cluster:
        for neighbor_id in mesh.faces[face_id].neighbors:
            neighbor_cluster = get_cluster_id(neighbor_id, all_clusters)
            if neighbor_cluster != cluster_id:
                boundary.add(face_id)
                break

    return boundary
```

### 1.2 Local Search Optimization

```python
def local_search_optimization(mesh, clusters, max_no_improve=50):
    """
    Local search: try random moves, accept improvements.
    """
    current_score = compute_total_interior_edges(mesh, clusters)
    no_improve_count = 0

    while no_improve_count < max_no_improve:
        # Pick random boundary face
        face_id, current_cluster = pick_random_boundary_face(clusters)

        if face_id is None:
            break

        # Try moving to each neighbor cluster
        neighbor_clusters = get_neighbor_cluster_ids(mesh, face_id, current_cluster, clusters)

        best_move = None
        best_gain = 0

        for neighbor_idx in neighbor_clusters:
            # Simulate move
            new_score = simulate_move_score(
                mesh, clusters, face_id, current_cluster, neighbor_idx
            )
            gain = new_score - current_score

            if gain > best_gain:
                # Check distortion
                if is_move_valid(mesh, clusters, face_id, current_cluster, neighbor_idx):
                    best_move = (face_id, current_cluster, neighbor_idx)
                    best_gain = gain

        if best_move:
            # Apply move
            face_id, from_cluster, to_cluster = best_move
            clusters[from_cluster].remove(face_id)
            clusters[to_cluster].add(face_id)
            current_score += best_gain
            no_improve_count = 0
        else:
            no_improve_count += 1

    return clusters
```

---

## 2. Cluster Merging & Splitting

### 2.1 Small Cluster Merging

Merge clusters that are too small to be useful.

```python
def merge_small_clusters(mesh, clusters, min_size=5):
    """
    Merge clusters smaller than min_size with their best neighbor.
    """
    merged = True

    while merged:
        merged = False

        for i, cluster in enumerate(clusters[:]):  # Copy for safe iteration
            if len(cluster) < min_size:
                # Find best neighbor to merge with
                neighbor_scores = []

                for j, other in enumerate(clusters):
                    if i == j:
                        continue

                    # Check if clusters are adjacent
                    if are_clusters_adjacent(mesh, cluster, other):
                        # Score = shared edges + inverse distortion impact
                        shared = count_shared_edges(mesh, cluster, other)
                        merged_distortion = estimate_merged_distortion(
                            mesh, cluster | other
                        )
                        score = shared / (1.0 + merged_distortion)
                        neighbor_scores.append((j, score))

                if neighbor_scores:
                    # Merge with highest scoring neighbor
                    best_neighbor, _ = max(neighbor_scores, key=lambda x: x[1])
                    clusters[best_neighbor].update(cluster)
                    clusters.remove(cluster)
                    merged = True
                    break

    return clusters
```

### 2.2 High-Distortion Splitting

Split clusters with too much distortion.

```python
def split_high_distortion_clusters(mesh, clusters, max_distortion=0.3):
    """
    Split clusters that exceed distortion threshold.
    """
    new_clusters = []

    for cluster in clusters:
        distortion = estimate_cluster_distortion(mesh, cluster)

        if distortion > max_distortion:
            # Need to split
            sub_clusters = split_cluster(mesh, cluster, max_distortion)
            new_clusters.extend(sub_clusters)
        else:
            new_clusters.append(cluster)

    return new_clusters


def split_cluster(mesh, cluster, max_distortion):
    """
    Split a single cluster into smaller pieces.
    """
    # Method 1: Cut at highest curvature edge
    faces = list(cluster)

    # Build mini dual graph for this cluster
    dual = build_cluster_dual_graph(mesh, cluster)

    # Find edge to cut (lowest weight = highest curvature)
    min_weight_edge = min(dual.edges, key=lambda e: e.weight)

    # Split into two components
    components = split_at_edge(dual, min_weight_edge)

    # Recursively split if still high distortion
    result = []
    for component in components:
        comp_distortion = estimate_cluster_distortion(mesh, component)
        if comp_distortion > max_distortion and len(component) > 3:
            result.extend(split_cluster(mesh, component, max_distortion))
        else:
            result.append(component)

    return result
```

---

## 3. Edge Weight Tuning

### 3.1 Adaptive Weight Adjustment

```python
class AdaptiveWeightOptimizer:
    """
    Automatically tune edge weights based on results.
    """

    def __init__(self, mesh, initial_weights):
        self.mesh = mesh
        self.weights = initial_weights.copy()
        self.history = []

    def optimize(self, iterations=20):
        """
        Iteratively adjust weights to improve results.
        """
        for i in range(iterations):
            # Run clustering with current weights
            clusters = self.cluster_with_weights()

            # Evaluate quality
            score = self.evaluate_clusters(clusters)
            self.history.append((self.weights.copy(), score))

            # Adjust weights based on analysis
            self.adjust_weights(clusters)

        # Return best weights
        best_idx = max(range(len(self.history)), key=lambda i: self.history[i][1])
        return self.history[best_idx][0]

    def adjust_weights(self, clusters):
        """
        Analyze clusters and adjust weights.
        """
        # If too many clusters, increase dihedral weight
        if len(clusters) > self.target_clusters * 1.5:
            self.weights['dihedral'] *= 1.1

        # If distortion too high, increase curvature weight
        max_dist = max(estimate_cluster_distortion(self.mesh, c) for c in clusters)
        if max_dist > self.max_distortion:
            self.weights['curvature'] *= 1.1

        # If seams too visible, increase visibility weight
        vis_score = compute_visibility_score(self.mesh, clusters)
        if vis_score > self.visibility_threshold:
            self.weights['visibility'] *= 1.1

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
```

### 3.2 Grid Search for Weights

```python
def grid_search_weights(mesh, weight_ranges, metric='interior_edges'):
    """
    Exhaustive search over weight combinations.
    """
    best_score = -float('inf')
    best_weights = None

    # Generate all combinations
    from itertools import product

    keys = list(weight_ranges.keys())
    value_lists = [weight_ranges[k] for k in keys]

    for values in product(*value_lists):
        weights = dict(zip(keys, values))

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Run clustering
        clusters = cluster_with_weights(mesh, weights)

        # Evaluate
        if metric == 'interior_edges':
            score = compute_interior_edge_ratio(mesh, clusters)
        elif metric == 'composite':
            score, _ = overall_quality_score(mesh, clusters)
        else:
            score = compute_metric(mesh, clusters, metric)

        if score > best_score:
            best_score = score
            best_weights = weights.copy()

    return best_weights, best_score


# Example usage:
weight_ranges = {
    'dihedral': [0.2, 0.3, 0.4, 0.5],
    'curvature': [0.1, 0.2, 0.3],
    'visibility': [0.1, 0.2, 0.3],
    'length': [0.05, 0.1, 0.15],
}
```

---

## 4. Simulated Annealing

### 4.1 Full Implementation

```python
import random
import math

def simulated_annealing_clustering(mesh, initial_clusters,
                                    initial_temp=1.0,
                                    cooling_rate=0.995,
                                    min_temp=0.01,
                                    iterations_per_temp=50):
    """
    Use simulated annealing to optimize cluster assignments.
    """
    current = [c.copy() for c in initial_clusters]
    current_score = compute_score(mesh, current)

    best = [c.copy() for c in current]
    best_score = current_score

    temp = initial_temp

    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Generate neighbor solution
            neighbor = generate_neighbor(mesh, current)

            if neighbor is None:
                continue

            neighbor_score = compute_score(mesh, neighbor)

            # Acceptance probability
            delta = neighbor_score - current_score

            if delta > 0:
                # Better solution - always accept
                current = neighbor
                current_score = neighbor_score

                if current_score > best_score:
                    best = [c.copy() for c in current]
                    best_score = current_score
            else:
                # Worse solution - accept with probability
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current = neighbor
                    current_score = neighbor_score

        # Cool down
        temp *= cooling_rate

    return best


def generate_neighbor(mesh, clusters):
    """
    Generate a neighbor solution by moving one boundary face.
    """
    # Collect all boundary faces
    boundary_moves = []

    for cluster_idx, cluster in enumerate(clusters):
        for face_id in get_boundary_faces(mesh, cluster, clusters):
            neighbors = get_neighbor_cluster_ids(mesh, face_id, cluster_idx, clusters)
            for neighbor_idx in neighbors:
                boundary_moves.append((face_id, cluster_idx, neighbor_idx))

    if not boundary_moves:
        return None

    # Pick random move
    face_id, from_idx, to_idx = random.choice(boundary_moves)

    # Check if move is valid (maintains connectivity, respects distortion)
    if not is_move_valid(mesh, clusters, face_id, from_idx, to_idx):
        return None

    # Apply move to copy
    new_clusters = [c.copy() for c in clusters]
    new_clusters[from_idx].remove(face_id)
    new_clusters[to_idx].add(face_id)

    # Remove empty clusters
    new_clusters = [c for c in new_clusters if len(c) > 0]

    return new_clusters


def compute_score(mesh, clusters):
    """
    Score function for optimization.
    Combines interior edges, distortion penalty, and cluster balance.
    """
    # Interior edge ratio (want to maximize)
    ie_ratio = compute_interior_edge_ratio(mesh, clusters)

    # Distortion penalty (want to minimize)
    max_distortion = max(
        estimate_cluster_distortion(mesh, c) for c in clusters
    )
    distortion_penalty = max(0, max_distortion - 0.3) * 2

    # Cluster balance (want balanced sizes)
    sizes = [len(c) for c in clusters]
    balance = 1.0 - (np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0

    # Combined score
    score = ie_ratio - distortion_penalty + 0.1 * balance

    return score
```

---

## 5. UV Island Optimization

### 5.1 UV Relaxation

After flattening, relax UVs to reduce distortion.

```python
def relax_uvs(mesh, uvs, cluster, iterations=100, step_size=0.1):
    """
    Laplacian smoothing of UVs to reduce distortion.
    """
    # Get boundary vertices (fixed during relaxation)
    boundary_verts = get_cluster_boundary_vertices(mesh, cluster)

    for _ in range(iterations):
        new_uvs = uvs.copy()

        for face_id in cluster:
            for vert_id in mesh.faces[face_id].vertices:
                if vert_id in boundary_verts:
                    continue  # Don't move boundary

                # Average of neighbors
                neighbors = mesh.get_vertex_neighbors(vert_id)
                neighbor_uvs = [uvs[v] for v in neighbors if v in uvs]

                if neighbor_uvs:
                    avg_uv = np.mean(neighbor_uvs, axis=0)
                    current_uv = uvs[vert_id]

                    # Move towards average
                    new_uvs[vert_id] = current_uv + step_size * (avg_uv - current_uv)

        uvs = new_uvs

    return uvs
```

### 5.2 Stretch Minimization

```python
def minimize_stretch(mesh, uvs, cluster, iterations=50):
    """
    Iteratively adjust UVs to minimize stretch distortion.
    """
    for _ in range(iterations):
        # Compute stretch gradient for each vertex
        gradients = {}

        for face_id in cluster:
            face_stretch = compute_face_stretch(mesh, uvs, face_id)

            if face_stretch > 1.1:  # Only optimize stretched faces
                grad = compute_stretch_gradient(mesh, uvs, face_id)
                for vert_id, g in grad.items():
                    if vert_id not in gradients:
                        gradients[vert_id] = np.zeros(2)
                    gradients[vert_id] += g

        # Apply gradients (gradient descent)
        step_size = 0.01
        for vert_id, grad in gradients.items():
            if not is_boundary_vertex(mesh, cluster, vert_id):
                uvs[vert_id] -= step_size * grad

    return uvs
```

---

## 6. Global Optimization Strategies

### 6.1 Multi-Resolution Optimization

```python
def multi_resolution_optimization(mesh, target_clusters):
    """
    Optimize at multiple resolution levels.
    """
    # Level 1: Coarse clustering
    simplified_mesh = simplify_mesh(mesh, target_faces=len(mesh.faces) // 4)
    coarse_clusters = cluster_mesh(simplified_mesh, target_clusters)

    # Map coarse clusters back to original mesh
    initial_clusters = map_clusters_to_original(
        coarse_clusters, simplified_mesh, mesh
    )

    # Level 2: Refine on original mesh
    refined_clusters = refine_clusters(mesh, initial_clusters)

    # Level 3: Local optimization
    optimized_clusters = local_search_optimization(mesh, refined_clusters)

    return optimized_clusters
```

### 6.2 Ensemble Method

```python
def ensemble_clustering(mesh, num_runs=5, target_clusters=5):
    """
    Run multiple clustering attempts with different strategies,
    then combine the best aspects.
    """
    results = []

    strategies = [
        ('msf_dihedral', {'weights': {'dihedral': 1.0}}),
        ('msf_curvature', {'weights': {'curvature': 0.5, 'dihedral': 0.5}}),
        ('greedy', {'method': 'greedy'}),
        ('hierarchical', {'method': 'hierarchical'}),
        ('spectral', {'method': 'spectral'}),
    ]

    for name, config in strategies:
        clusters = cluster_with_config(mesh, config)
        score, details = evaluate_clusters(mesh, clusters)
        results.append({
            'name': name,
            'clusters': clusters,
            'score': score,
            'details': details
        })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Return best
    return results[0]['clusters']


def hybrid_from_ensemble(results, mesh):
    """
    Create hybrid solution from multiple runs.
    Take best aspects of each.
    """
    # Find consensus on which faces should be in same cluster
    co_cluster_votes = defaultdict(int)

    for result in results:
        for cluster in result['clusters']:
            faces = list(cluster)
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    pair = tuple(sorted([faces[i], faces[j]]))
                    co_cluster_votes[pair] += 1

    # Build clusters from consensus
    threshold = len(results) // 2 + 1  # Majority vote

    # Union-Find to group faces
    parent = {f: f for f in mesh.faces}

    for (f1, f2), votes in co_cluster_votes.items():
        if votes >= threshold:
            union(parent, f1, f2)

    # Extract clusters
    clusters_dict = defaultdict(set)
    for f in mesh.faces:
        root = find(parent, f)
        clusters_dict[root].add(f)

    return list(clusters_dict.values())
```

---

## 7. Performance Optimization

### 7.1 Caching Edge Computations

```python
class CachedEdgeComputer:
    """
    Cache expensive edge computations.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._dihedral_cache = {}
        self._curvature_cache = {}
        self._length_cache = {}

    def get_dihedral(self, f1, f2):
        key = tuple(sorted([f1, f2]))
        if key not in self._dihedral_cache:
            self._dihedral_cache[key] = compute_dihedral_angle(
                self.mesh, f1, f2
            )
        return self._dihedral_cache[key]

    def get_edge_weight(self, f1, f2, edge):
        """
        Cached composite weight computation.
        """
        key = (f1, f2, edge)
        if key not in self._weight_cache:
            w_dihedral = self.get_dihedral(f1, f2)
            w_length = self.get_length(edge)
            self._weight_cache[key] = compute_composite(w_dihedral, w_length)
        return self._weight_cache[key]
```

### 7.2 Incremental Score Updates

```python
class IncrementalScorer:
    """
    Track score incrementally as faces move between clusters.
    """

    def __init__(self, mesh, clusters):
        self.mesh = mesh
        self.clusters = clusters
        self._interior_edges = self._compute_initial_interior()
        self._face_to_cluster = self._build_face_map()

    def _compute_initial_interior(self):
        """Compute initial interior edge count."""
        count = 0
        for edge in self.mesh.interior_edges():
            f1, f2 = self.mesh.get_edge_faces(edge)
            if self._same_cluster(f1, f2):
                count += 1
        return count

    def move_face(self, face_id, from_cluster, to_cluster):
        """
        Move face and update score incrementally.
        """
        # Edges that will change status
        for neighbor in self.mesh.faces[face_id].neighbors:
            neighbor_cluster = self._face_to_cluster[neighbor]

            # Was interior (same cluster), now boundary?
            if neighbor_cluster == from_cluster:
                self._interior_edges -= 1

            # Was boundary, now interior (same cluster)?
            if neighbor_cluster == to_cluster:
                self._interior_edges += 1

        # Update mappings
        self.clusters[from_cluster].remove(face_id)
        self.clusters[to_cluster].add(face_id)
        self._face_to_cluster[face_id] = to_cluster

    @property
    def interior_edge_count(self):
        return self._interior_edges
```

---

## 8. Optimization Pipeline Summary

```python
def full_optimization_pipeline(mesh, config):
    """
    Complete optimization pipeline.
    """
    # Phase 1: Initial clustering
    print("Phase 1: Initial clustering...")
    clusters = initial_clustering(mesh, config)

    # Phase 2: Merge small clusters
    print("Phase 2: Merging small clusters...")
    clusters = merge_small_clusters(mesh, clusters, min_size=config['min_size'])

    # Phase 3: Split high-distortion clusters
    print("Phase 3: Splitting high-distortion clusters...")
    clusters = split_high_distortion_clusters(
        mesh, clusters, max_distortion=config['max_distortion']
    )

    # Phase 4: Boundary optimization
    print("Phase 4: Boundary optimization...")
    clusters = optimize_boundary_faces(mesh, clusters, max_iterations=100)

    # Phase 5: Simulated annealing refinement
    if config.get('use_annealing', True):
        print("Phase 5: Simulated annealing...")
        clusters = simulated_annealing_clustering(
            mesh, clusters,
            initial_temp=0.5,
            cooling_rate=0.99,
            iterations_per_temp=30
        )

    # Phase 6: Final cleanup
    print("Phase 6: Final cleanup...")
    clusters = merge_small_clusters(mesh, clusters, min_size=3)

    # Report
    score, details = evaluate_clusters(mesh, clusters)
    print(f"\nFinal score: {score:.3f}")
    print(f"Interior edges: {details['interior_edges']:.1%}")
    print(f"Clusters: {len(clusters)}")

    return clusters
```
