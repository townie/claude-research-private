# Detailed Algorithms for Edge-Maximized Polygon Unwrapping

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Algorithm 1: Maximum Spanning Forest Clustering

This algorithm directly addresses maximizing interior edges by finding a maximum spanning forest on the dual graph.

### Theory

- Each tree in the spanning forest becomes one UV island
- Edges in the forest = interior edges (faces stay connected)
- Edges NOT in the forest = cut edges (seams)
- Maximum spanning forest = maximum interior edges

### Implementation

```python
import heapq
from collections import defaultdict
from typing import List, Set, Dict, Tuple

class Face:
    def __init__(self, id: int, vertices: List[int]):
        self.id = id
        self.vertices = vertices
        self.neighbors: List[int] = []  # Adjacent face IDs

class Mesh:
    def __init__(self):
        self.faces: Dict[int, Face] = {}
        self.edges: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def add_face(self, face: Face):
        self.faces[face.id] = face
        # Build edge-to-face mapping
        verts = face.vertices
        for i in range(len(verts)):
            edge = tuple(sorted([verts[i], verts[(i + 1) % len(verts)]]))
            self.edges[edge].append(face.id)

    def compute_adjacency(self):
        """Compute face adjacency from shared edges."""
        for edge, face_ids in self.edges.items():
            if len(face_ids) == 2:
                f1, f2 = face_ids
                self.faces[f1].neighbors.append(f2)
                self.faces[f2].neighbors.append(f1)


def compute_edge_weight(mesh: Mesh, face1_id: int, face2_id: int) -> float:
    """
    Compute weight for edge between two faces.
    Higher weight = prefer to keep this edge (not cut it).

    Factors:
    - Dihedral angle (flatter = higher weight)
    - Edge length (longer = higher weight, more visible if cut)
    - Curvature similarity
    """
    # Simplified: use dihedral angle
    # In practice, compute from actual geometry
    dihedral_angle = compute_dihedral_angle(mesh, face1_id, face2_id)

    # Prefer flat connections (dihedral near 180 degrees)
    flatness = abs(180 - dihedral_angle)
    weight = 1.0 / (1.0 + flatness / 180.0)

    return weight


def compute_dihedral_angle(mesh: Mesh, f1: int, f2: int) -> float:
    """Placeholder - compute actual dihedral angle from geometry."""
    # In real implementation: compute from face normals
    return 170.0  # Placeholder


def maximum_spanning_forest(mesh: Mesh, max_distortion: float = 0.3) -> List[Set[int]]:
    """
    Build maximum spanning forest on dual graph.

    Returns list of clusters (sets of face IDs).
    """
    # Build list of all dual edges with weights
    dual_edges = []
    seen_pairs = set()

    for face_id, face in mesh.faces.items():
        for neighbor_id in face.neighbors:
            pair = tuple(sorted([face_id, neighbor_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                weight = compute_edge_weight(mesh, face_id, neighbor_id)
                dual_edges.append((-weight, face_id, neighbor_id))  # Negative for max-heap behavior

    # Sort edges by weight (descending via negative weights)
    heapq.heapify(dual_edges)

    # Union-Find for forest construction
    parent = {f: f for f in mesh.faces}
    rank = {f: 0 for f in mesh.faces}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Track cluster distortion estimates
    cluster_faces: Dict[int, Set[int]] = {f: {f} for f in mesh.faces}

    # Kruskal's algorithm for maximum spanning forest
    forest_edges = []
    while dual_edges:
        neg_weight, f1, f2 = heapq.heappop(dual_edges)
        weight = -neg_weight

        root1, root2 = find(f1), find(f2)
        if root1 != root2:
            # Check if merging would exceed distortion threshold
            merged_faces = cluster_faces[root1] | cluster_faces[root2]
            estimated_distortion = estimate_cluster_distortion(mesh, merged_faces)

            if estimated_distortion <= max_distortion:
                union(f1, f2)
                forest_edges.append((f1, f2))

                # Update cluster tracking
                new_root = find(f1)
                cluster_faces[new_root] = merged_faces
                if new_root != root1:
                    del cluster_faces[root1]
                if new_root != root2:
                    del cluster_faces[root2]

    # Extract final clusters
    clusters_dict: Dict[int, Set[int]] = defaultdict(set)
    for face_id in mesh.faces:
        root = find(face_id)
        clusters_dict[root].add(face_id)

    return list(clusters_dict.values())


def estimate_cluster_distortion(mesh: Mesh, faces: Set[int]) -> float:
    """
    Estimate distortion if this set of faces were flattened together.

    Uses total angle deficit as proxy for distortion.
    """
    # Simplified estimation
    # Real implementation would do actual parameterization or use
    # analytical bounds based on Gaussian curvature
    if len(faces) <= 3:
        return 0.0

    # Placeholder: larger clusters have more potential distortion
    return len(faces) * 0.01


# Example usage
def example_usage():
    mesh = Mesh()

    # Add faces (example: simple cube-like structure)
    # In practice, load from file
    mesh.add_face(Face(0, [0, 1, 2, 3]))  # Front
    mesh.add_face(Face(1, [4, 5, 6, 7]))  # Back
    mesh.add_face(Face(2, [0, 1, 5, 4]))  # Bottom
    mesh.add_face(Face(3, [2, 3, 7, 6]))  # Top
    mesh.add_face(Face(4, [0, 3, 7, 4]))  # Left
    mesh.add_face(Face(5, [1, 2, 6, 5]))  # Right

    mesh.compute_adjacency()

    clusters = maximum_spanning_forest(mesh, max_distortion=0.5)

    print(f"Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: faces {cluster}")
```

---

## Algorithm 2: Greedy Region Growing with Edge Maximization

This approach grows clusters greedily, always adding the face that maximizes interior edges.

```python
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import math

@dataclass
class ClusterStats:
    faces: Set[int]
    boundary_edges: Set[Tuple[int, int]]
    interior_edges: Set[Tuple[int, int]]
    total_angle_deficit: float

def grow_cluster_maximize_edges(
    mesh: Mesh,
    seed: int,
    available: Set[int],
    max_distortion: float
) -> ClusterStats:
    """
    Grow a cluster from seed, prioritizing faces that add the most interior edges.
    """
    cluster = ClusterStats(
        faces={seed},
        boundary_edges=get_face_edges(mesh, seed),
        interior_edges=set(),
        total_angle_deficit=get_angle_deficit(mesh, seed)
    )
    available.remove(seed)

    while True:
        best_candidate = None
        best_edge_gain = -1
        best_new_distortion = float('inf')

        # Find all adjacent available faces
        adjacent = set()
        for face_id in cluster.faces:
            for neighbor in mesh.faces[face_id].neighbors:
                if neighbor in available:
                    adjacent.add(neighbor)

        for candidate in adjacent:
            # Count shared edges with cluster
            candidate_edges = get_face_edges(mesh, candidate)
            shared_edges = candidate_edges & cluster.boundary_edges
            edge_gain = len(shared_edges)

            # Estimate distortion with this face added
            new_deficit = cluster.total_angle_deficit + get_angle_deficit(mesh, candidate)
            estimated_distortion = estimate_distortion_from_deficit(
                new_deficit, len(cluster.faces) + 1
            )

            if estimated_distortion <= max_distortion:
                if edge_gain > best_edge_gain or \
                   (edge_gain == best_edge_gain and estimated_distortion < best_new_distortion):
                    best_candidate = candidate
                    best_edge_gain = edge_gain
                    best_new_distortion = estimated_distortion

        if best_candidate is None:
            break

        # Add best candidate to cluster
        candidate_edges = get_face_edges(mesh, best_candidate)
        shared = candidate_edges & cluster.boundary_edges

        cluster.faces.add(best_candidate)
        cluster.interior_edges |= shared
        cluster.boundary_edges = (cluster.boundary_edges | candidate_edges) - shared - cluster.interior_edges
        cluster.total_angle_deficit += get_angle_deficit(mesh, best_candidate)

        available.remove(best_candidate)

    return cluster


def get_face_edges(mesh: Mesh, face_id: int) -> Set[Tuple[int, int]]:
    """Get all edges of a face as sorted tuples."""
    face = mesh.faces[face_id]
    edges = set()
    verts = face.vertices
    for i in range(len(verts)):
        edge = tuple(sorted([verts[i], verts[(i + 1) % len(verts)]]))
        edges.add(edge)
    return edges


def get_angle_deficit(mesh: Mesh, face_id: int) -> float:
    """Get total angle deficit contribution from this face."""
    # Placeholder - compute from actual vertex angles
    return 0.0


def estimate_distortion_from_deficit(total_deficit: float, num_faces: int) -> float:
    """Estimate flattening distortion from angle deficit."""
    if num_faces == 0:
        return 0.0
    return abs(total_deficit) / (num_faces * math.pi)


def cluster_mesh_greedy(mesh: Mesh, max_distortion: float = 0.3) -> List[Set[int]]:
    """
    Cluster entire mesh using greedy region growing.
    """
    available = set(mesh.faces.keys())
    clusters = []

    while available:
        # Pick seed: face with most available neighbors (for better clustering)
        seed = max(
            available,
            key=lambda f: sum(1 for n in mesh.faces[f].neighbors if n in available)
        )

        cluster_stats = grow_cluster_maximize_edges(mesh, seed, available, max_distortion)
        clusters.append(cluster_stats.faces)

    return clusters
```

---

## Algorithm 3: Hierarchical Agglomerative Clustering

Bottom-up approach that merges clusters to maximize edge connectivity.

```python
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
import heapq

@dataclass
class Cluster:
    id: int
    faces: Set[int]
    adjacent_clusters: Set[int]
    distortion_estimate: float

class HierarchicalClusterer:
    def __init__(self, mesh: Mesh, max_distortion: float):
        self.mesh = mesh
        self.max_distortion = max_distortion
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0

    def initialize(self):
        """Create initial clusters (one per face)."""
        for face_id in self.mesh.faces:
            cluster = Cluster(
                id=self.next_cluster_id,
                faces={face_id},
                adjacent_clusters=set(),
                distortion_estimate=0.0
            )
            self.clusters[cluster.id] = cluster
            self.next_cluster_id += 1

        # Compute adjacencies
        face_to_cluster = {f: c.id for c in self.clusters.values() for f in c.faces}
        for face in self.mesh.faces.values():
            my_cluster = face_to_cluster[face.id]
            for neighbor in face.neighbors:
                neighbor_cluster = face_to_cluster[neighbor]
                if my_cluster != neighbor_cluster:
                    self.clusters[my_cluster].adjacent_clusters.add(neighbor_cluster)

    def compute_merge_score(self, c1_id: int, c2_id: int) -> Tuple[float, float]:
        """
        Compute score for merging two clusters.
        Returns (edge_gain, distortion_cost).
        """
        c1, c2 = self.clusters[c1_id], self.clusters[c2_id]

        # Count shared edges between clusters
        edge_gain = self.count_shared_edges(c1.faces, c2.faces)

        # Estimate merged distortion
        merged_distortion = self.estimate_merged_distortion(c1, c2)

        return (edge_gain, merged_distortion)

    def count_shared_edges(self, faces1: Set[int], faces2: Set[int]) -> int:
        """Count edges shared between two face sets."""
        count = 0
        for f1 in faces1:
            for neighbor in self.mesh.faces[f1].neighbors:
                if neighbor in faces2:
                    count += 1
        return count // 2  # Each edge counted twice

    def estimate_merged_distortion(self, c1: Cluster, c2: Cluster) -> float:
        """Estimate distortion of merged cluster."""
        # Simplified: use sum of distortions + penalty for size
        base = max(c1.distortion_estimate, c2.distortion_estimate)
        size_penalty = (len(c1.faces) + len(c2.faces)) * 0.01
        return base + size_penalty

    def merge_clusters(self, c1_id: int, c2_id: int) -> int:
        """Merge two clusters, return new cluster ID."""
        c1, c2 = self.clusters[c1_id], self.clusters[c2_id]

        new_cluster = Cluster(
            id=self.next_cluster_id,
            faces=c1.faces | c2.faces,
            adjacent_clusters=(c1.adjacent_clusters | c2.adjacent_clusters) - {c1_id, c2_id},
            distortion_estimate=self.estimate_merged_distortion(c1, c2)
        )
        self.next_cluster_id += 1

        # Update adjacencies of other clusters
        for adj_id in new_cluster.adjacent_clusters:
            adj = self.clusters[adj_id]
            adj.adjacent_clusters.discard(c1_id)
            adj.adjacent_clusters.discard(c2_id)
            adj.adjacent_clusters.add(new_cluster.id)

        # Remove old clusters, add new
        del self.clusters[c1_id]
        del self.clusters[c2_id]
        self.clusters[new_cluster.id] = new_cluster

        return new_cluster.id

    def run(self, target_clusters: Optional[int] = None) -> List[Set[int]]:
        """
        Run hierarchical clustering.

        Args:
            target_clusters: Stop when this many clusters reached.
                           If None, merge until distortion limit reached.
        """
        self.initialize()

        # Priority queue: (-edge_gain, distortion, c1_id, c2_id)
        # Negative edge_gain for max-heap behavior
        merge_queue = []

        def add_merge_candidates(cluster_id: int):
            cluster = self.clusters[cluster_id]
            for adj_id in cluster.adjacent_clusters:
                if adj_id in self.clusters:
                    edge_gain, distortion = self.compute_merge_score(cluster_id, adj_id)
                    if distortion <= self.max_distortion:
                        heapq.heappush(merge_queue, (-edge_gain, distortion, cluster_id, adj_id))

        # Initialize queue
        for cluster_id in self.clusters:
            add_merge_candidates(cluster_id)

        # Merge until stopping condition
        while merge_queue:
            if target_clusters and len(self.clusters) <= target_clusters:
                break

            neg_gain, distortion, c1_id, c2_id = heapq.heappop(merge_queue)

            # Check if clusters still exist (may have been merged already)
            if c1_id not in self.clusters or c2_id not in self.clusters:
                continue

            # Verify distortion is still acceptable
            actual_gain, actual_distortion = self.compute_merge_score(c1_id, c2_id)
            if actual_distortion > self.max_distortion:
                continue

            # Perform merge
            new_id = self.merge_clusters(c1_id, c2_id)
            add_merge_candidates(new_id)

        return [c.faces for c in self.clusters.values()]
```

---

## Algorithm 4: Cut Edge Selection via Minimum Cut

Frame the problem as finding minimum edge cut to separate mesh into flattenable regions.

```python
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def spectral_clustering_dual_graph(mesh: Mesh, num_clusters: int) -> List[Set[int]]:
    """
    Use spectral clustering on dual graph to partition faces.

    This naturally tends to cut along "narrow" connections,
    preserving large connected regions.
    """
    n = len(mesh.faces)
    face_ids = list(mesh.faces.keys())
    id_to_idx = {fid: i for i, fid in enumerate(face_ids)}

    # Build adjacency matrix for dual graph
    W = lil_matrix((n, n))
    for face in mesh.faces.values():
        i = id_to_idx[face.id]
        for neighbor in face.neighbors:
            j = id_to_idx[neighbor]
            # Weight by edge desirability (higher = prefer not to cut)
            weight = compute_edge_weight(mesh, face.id, neighbor)
            W[i, j] = weight
            W[j, i] = weight

    W = W.tocsr()

    # Build graph Laplacian
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt

    # Compute eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k=num_clusters, which='SM')

    # Use k-means on eigenvector embedding
    from sklearn.cluster import KMeans
    embedding = eigenvectors[:, 1:num_clusters]  # Skip first eigenvector
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    # Convert labels to clusters
    clusters = [set() for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].add(face_ids[idx])

    return clusters
```

---

## Performance Comparison

| Algorithm | Time Complexity | Edge Optimization | Distortion Control |
|-----------|-----------------|-------------------|-------------------|
| Maximum Spanning Forest | O(E log E) | Optimal | Approximate |
| Greedy Region Growing | O(F²) | Greedy (good) | Exact per step |
| Hierarchical Clustering | O(F² log F) | Good | Configurable |
| Spectral Clustering | O(F³) | Implicit | Post-hoc |

**Recommendations:**
- For **small meshes** (<1000 faces): Hierarchical or Spectral
- For **large meshes**: Maximum Spanning Forest or Greedy Growing
- For **strict distortion limits**: Greedy with per-step validation
- For **maximum edge preservation**: Maximum Spanning Forest

---

## Practical Tips

1. **Precompute adjacency**: Store face-to-face adjacency for O(1) neighbor lookup

2. **Use spatial indexing**: Octrees/KD-trees for large meshes

3. **Incremental distortion**: Track running distortion estimate to avoid recomputation

4. **Edge weight tuning**: Experiment with different weight functions:
   - Dihedral angle only
   - Edge length weighted
   - Curvature-aware
   - Feature-edge aware (preserve sharp edges)

5. **Post-processing**: After initial clustering:
   - Reassign boundary faces if it improves connectivity
   - Split high-distortion clusters
   - Merge tiny clusters with neighbors
