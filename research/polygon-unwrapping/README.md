# Polygon Unwrapping Research: Edge-Maximized Clustering

**Research Topic**: How to take a low polygon object and unwrap it into multiple smaller parts where the maximum number of edges touch in each cluster.

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Core Concepts](#core-concepts)
3. [Algorithms & Approaches](#algorithms--approaches)
4. [Edge-Maximized Clustering Strategies](#edge-maximized-clustering-strategies)
5. [Implementation Considerations](#implementation-considerations)
6. [References & Further Reading](#references--further-reading)

---

## Problem Definition

### The Challenge

Given a low-polygon 3D mesh, we want to:
1. **Segment** the mesh into multiple "charts" or "islands"
2. **Flatten** each chart into 2D (UV space)
3. **Maximize** the number of connected/touching edges within each cluster

### Why Maximize Touching Edges?

- **Reduced seams**: Fewer cut edges mean fewer visible seams in textures
- **Better texture continuity**: Connected faces share texture information smoothly
- **Efficient UV space**: Compact charts with high edge connectivity use texture space efficiently
- **Reduced distortion**: Larger connected regions can sometimes distribute distortion better

### Formal Problem Statement

```
Given: A mesh M = (V, E, F) where:
  - V = set of vertices
  - E = set of edges
  - F = set of faces (polygons)

Find: A partition P = {C₁, C₂, ..., Cₖ} of F such that:
  1. Each cluster Cᵢ is flattenable (can be mapped to 2D with bounded distortion)
  2. The sum of interior edges (edges connecting faces within same cluster) is maximized
  3. Equivalently: The number of cut edges (seams) is minimized
```

---

## Core Concepts

### 1. Mesh Topology Basics

```
Vertex (V): A point in 3D space
Edge (E): A connection between two vertices
Face (F): A polygon defined by a cycle of edges (typically triangles or quads)
Half-edge: Directed edge, useful for mesh traversal
```

### 2. Euler's Formula

For a closed manifold mesh:
```
V - E + F = 2(1 - g)
```
Where `g` is the genus (number of handles/holes).

### 3. Types of Edges

| Edge Type | Description |
|-----------|-------------|
| **Interior Edge** | Shared by exactly 2 faces |
| **Boundary Edge** | Belongs to only 1 face (mesh boundary) |
| **Cut Edge (Seam)** | Interior edge chosen to separate UV islands |

### 4. Gaussian Curvature and Developability

A surface region is **developable** (can flatten without distortion) if its Gaussian curvature is zero everywhere. For discrete meshes:

```
Angle Deficit at vertex v = 2π - Σ(angles around v)
```

High angle deficit = high curvature = harder to flatten without distortion.

---

## Algorithms & Approaches

### 1. Graph-Based Dual Mesh Approach

Convert the face adjacency into a graph problem:

```
Dual Graph G* = (F, E*)
- Nodes: Faces of the original mesh
- Edges: Connect faces that share an edge in the original mesh
```

**Maximizing touching edges** = **Finding large connected subgraphs** that remain flattenable.

### 2. Spanning Tree Method

A classic approach to unwrapping:

```python
# Pseudocode: Spanning Tree Unwrapping
def spanning_tree_unwrap(mesh):
    dual_graph = build_dual_graph(mesh)

    # Maximum spanning tree maximizes interior edges
    # Weight edges by desirability (curvature, angles, etc.)
    mst = maximum_spanning_tree(dual_graph, weight_function)

    # Cut edges NOT in the spanning tree
    cut_edges = all_edges - mst_edges

    return cut_edges
```

**Key insight**: A spanning tree of the dual graph corresponds to a single connected UV island. Edges NOT in the tree become cuts.

### 3. Region Growing / Flood Fill

```python
def region_growing_cluster(mesh, seed_face, max_distortion):
    cluster = {seed_face}
    frontier = get_adjacent_faces(seed_face)

    while frontier:
        best_face = select_best_candidate(frontier, cluster)

        if can_add_without_excess_distortion(cluster, best_face, max_distortion):
            cluster.add(best_face)
            frontier.update(get_adjacent_faces(best_face) - cluster)
            frontier.remove(best_face)
        else:
            frontier.remove(best_face)

    return cluster
```

### 4. Hierarchical Clustering

```
1. Start with each face as its own cluster
2. Iteratively merge adjacent clusters
3. Merge criterion: minimize distortion increase while maximizing edge connectivity
4. Stop when distortion threshold exceeded or desired cluster count reached
```

### 5. Spectral Clustering on Dual Graph

Use graph Laplacian eigenvalues to find natural cluster boundaries:

```python
def spectral_clustering(dual_graph, k_clusters):
    L = compute_laplacian(dual_graph)
    eigenvalues, eigenvectors = eig(L)

    # Use first k eigenvectors for embedding
    embedding = eigenvectors[:, :k]

    # Cluster in embedding space
    clusters = kmeans(embedding, k)
    return clusters
```

---

## Edge-Maximized Clustering Strategies

### Strategy 1: Minimum Cut Partitioning

**Goal**: Find the partition that minimizes the number of cut edges.

```python
def minimum_cut_partition(mesh, num_clusters):
    dual_graph = build_dual_graph(mesh)

    # Edge weights: higher = prefer NOT to cut
    for edge in dual_graph.edges:
        edge.weight = compute_edge_importance(edge)

    # Use normalized cuts or similar algorithm
    partition = normalized_cuts(dual_graph, num_clusters)

    return partition
```

**Edge Importance Factors**:
- Edge length (longer edges = more visible seams)
- Dihedral angle (flat connections preferred)
- Curvature continuity
- Texture feature alignment

### Strategy 2: Greedy Edge Preservation

```python
def greedy_edge_maximization(mesh, distortion_threshold):
    clusters = []
    unassigned = set(mesh.faces)

    while unassigned:
        # Start new cluster from face with most unassigned neighbors
        seed = max(unassigned, key=lambda f: count_unassigned_neighbors(f, unassigned))
        cluster = grow_cluster_max_edges(seed, unassigned, distortion_threshold)

        clusters.append(cluster)
        unassigned -= cluster

    return clusters

def grow_cluster_max_edges(seed, available, threshold):
    cluster = {seed}

    while True:
        # Find candidate that adds most edges to cluster
        candidates = get_adjacent_available(cluster, available)
        if not candidates:
            break

        # Sort by number of shared edges with current cluster
        candidates.sort(key=lambda f: count_shared_edges(f, cluster), reverse=True)

        for candidate in candidates:
            if would_not_exceed_distortion(cluster | {candidate}, threshold):
                cluster.add(candidate)
                break
        else:
            break  # No valid candidate found

    return cluster
```

### Strategy 3: Distortion-Aware Clustering

Balance edge connectivity with flattening distortion:

```python
def distortion_aware_clustering(mesh):
    # Compute per-face metrics
    for face in mesh.faces:
        face.gaussian_curvature = compute_gaussian_curvature(face)
        face.stretch_potential = estimate_stretch(face)

    # Group faces by similar distortion characteristics
    feature_vectors = [compute_features(f) for f in mesh.faces]
    initial_clusters = cluster_by_features(feature_vectors)

    # Refine clusters to maximize edge connectivity
    refined_clusters = refine_for_connectivity(initial_clusters, mesh)

    return refined_clusters
```

### Strategy 4: Priority-Based Merging

```python
def priority_merge_clustering(mesh, target_clusters):
    # Initialize: each face is its own cluster
    clusters = [{f} for f in mesh.faces]

    # Priority queue of potential merges
    # Priority = (edge_gain, -distortion_cost)
    merge_queue = PriorityQueue()

    for c1, c2 in adjacent_cluster_pairs(clusters):
        priority = compute_merge_priority(c1, c2)
        merge_queue.push((priority, c1, c2))

    while len(clusters) > target_clusters and merge_queue:
        priority, c1, c2 = merge_queue.pop()

        if c1 in clusters and c2 in clusters:  # Still valid
            merged = merge_clusters(c1, c2)
            clusters.remove(c1)
            clusters.remove(c2)
            clusters.append(merged)

            # Add new potential merges
            for neighbor in get_adjacent_clusters(merged, clusters):
                priority = compute_merge_priority(merged, neighbor)
                merge_queue.push((priority, merged, neighbor))

    return clusters
```

---

## Implementation Considerations

### Data Structures

#### Half-Edge Mesh Structure
```python
class HalfEdge:
    vertex: Vertex      # Target vertex
    face: Face          # Adjacent face
    twin: HalfEdge      # Opposite half-edge
    next: HalfEdge      # Next half-edge in face
    prev: HalfEdge      # Previous half-edge in face

class Face:
    half_edge: HalfEdge  # One half-edge of this face
    cluster_id: int      # Which cluster this face belongs to
```

#### Adjacency-Based Structure
```python
class Mesh:
    vertices: List[Vector3]
    faces: List[List[int]]  # Face as vertex indices

    # Precomputed adjacency
    face_adjacency: Dict[int, List[int]]  # face_id -> adjacent face_ids
    edge_faces: Dict[Tuple[int,int], List[int]]  # edge -> faces sharing it
```

### Distortion Metrics

#### 1. Stretch Metric (L² Stretch)
```python
def compute_stretch(face_3d, face_uv):
    # Compute singular values of the Jacobian
    J = compute_jacobian(face_3d, face_uv)
    s1, s2 = singular_values(J)

    L2_stretch = sqrt((s1² + s2²) / 2)
    return L2_stretch
```

#### 2. Angle Distortion (Conformal Error)
```python
def compute_angle_distortion(face_3d, face_uv):
    angles_3d = compute_angles(face_3d)
    angles_uv = compute_angles(face_uv)

    return sum(abs(a3d - auv) for a3d, auv in zip(angles_3d, angles_uv))
```

#### 3. Area Distortion
```python
def compute_area_distortion(face_3d, face_uv):
    area_3d = compute_area(face_3d)
    area_uv = compute_area(face_uv)

    return abs(log(area_uv / area_3d))
```

### Flattening Methods for Each Cluster

Once clusters are determined, each must be flattened:

| Method | Preserves | Complexity | Best For |
|--------|-----------|------------|----------|
| **LSCM** (Least Squares Conformal Maps) | Angles | O(n²) | General shapes |
| **ABF++** (Angle-Based Flattening) | Angles | O(n) | Large meshes |
| **Stretch Minimization** | Areas/Lengths | O(n²) | Low distortion |
| **Exponential Map** | Local distances | O(n) | Near-developable |

### Practical Algorithm: Complete Pipeline

```python
def unwrap_mesh_edge_maximized(mesh, max_distortion=0.1, min_cluster_size=4):
    """
    Main unwrapping pipeline that maximizes edge connectivity.
    """
    # Step 1: Analyze mesh
    compute_mesh_properties(mesh)  # curvature, angles, etc.

    # Step 2: Build dual graph with edge weights
    dual = build_weighted_dual_graph(mesh)

    # Step 3: Initial clustering via maximum spanning forest
    clusters = maximum_spanning_forest_clustering(dual, max_distortion)

    # Step 4: Refine clusters - merge small ones, split high-distortion ones
    clusters = refine_clusters(clusters, min_cluster_size, max_distortion)

    # Step 5: Flatten each cluster
    uv_maps = {}
    for cluster in clusters:
        uv_maps[cluster.id] = flatten_cluster(cluster, method='LSCM')

    # Step 6: Pack UV islands
    packed_uvs = pack_uv_islands(uv_maps)

    return packed_uvs, clusters
```

---

## References & Further Reading

### Academic Papers

1. **"Least Squares Conformal Maps"** - Lévy et al., 2002
   - Foundation for angle-preserving parameterization

2. **"ABF++: Fast and Robust Angle Based Flattening"** - Sheffer et al., 2005
   - Efficient angle-based method

3. **"Spectral Surface Quadrangulation"** - Dong et al., 2006
   - Spectral methods for mesh segmentation

4. **"D-Charts: Quasi-Developable Mesh Segmentation"** - Julius et al., 2005
   - Segmentation for near-developable regions

5. **"Variational Shape Approximation"** - Cohen-Steiner et al., 2004
   - Mesh clustering with geometric proxies

### Software Libraries

| Library | Language | Features |
|---------|----------|----------|
| **libigl** | C++/Python | General geometry processing |
| **OpenMesh** | C++ | Half-edge data structure |
| **CGAL** | C++ | Comprehensive geometry algorithms |
| **Blender (Python API)** | Python | UV unwrapping tools |
| **xatlas** | C++ | Automatic UV atlas generation |

### Key Concepts to Explore Further

- [ ] Graph partitioning algorithms (Metis, KaHIP)
- [ ] Mesh simplification and its effect on unwrapping
- [ ] Seamless texture synthesis across UV boundaries
- [ ] Hierarchical mesh segmentation
- [ ] Machine learning approaches to mesh segmentation

---

## Summary

To maximize touching edges in polygon unwrapping:

1. **Model as graph problem**: Use dual graph where faces are nodes
2. **Maximize interior edges**: Equivalent to finding minimum cut or maximum spanning forest
3. **Balance with distortion**: Large clusters = more edges but potentially more distortion
4. **Use appropriate algorithms**: Greedy region growing, hierarchical clustering, or spectral methods
5. **Refine iteratively**: Start with maximum connectivity, then split where distortion is unacceptable

The optimal solution depends on:
- Mesh topology and geometry
- Acceptable distortion levels
- Target number of UV islands
- Texture resolution and seam visibility requirements
