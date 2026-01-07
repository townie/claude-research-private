# Practical Examples: Polygon Unwrapping with Edge-Maximized Clustering

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Example 1: Unwrapping a Simple Cube

### The Problem

A cube has 6 faces. We want to unwrap it while maximizing connected edges.

```
    +-------+
   /|      /|
  / |     / |
 +-------+  |
 |  +....|..+
 | /     | /
 |/      |/
 +-------+

Faces: Front, Back, Top, Bottom, Left, Right
Edges: 12 total
```

### Face Adjacency (Dual Graph)

```
           Top
            |
Left --- Front --- Right --- Back
            |
         Bottom
```

### Optimal Unwrapping: Cross Pattern

Maximum interior edges = 5 (all edges except one cut)

```
         +---+
         | T |
     +---+---+---+---+
     | L | F | R | B |
     +---+---+---+---+
         | Bo|
         +---+

Interior edges: 5 (T-F, L-F, F-R, R-B, F-Bo)
Cut edges: 1 (to separate the net from forming a loop)
```

### Alternative: T-Shape

```
     +---+
     | T |
 +---+---+---+
 | L | F | R |
 +---+---+---+
     | Bo|
     +---+
     | B |
     +---+

Interior edges: 5
Cut edges: 1
```

### Code Example

```python
def unwrap_cube():
    # Define cube faces with adjacencies
    cube = {
        'front':  {'top', 'bottom', 'left', 'right'},
        'back':   {'top', 'bottom', 'left', 'right'},
        'top':    {'front', 'back', 'left', 'right'},
        'bottom': {'front', 'back', 'left', 'right'},
        'left':   {'front', 'back', 'top', 'bottom'},
        'right':  {'front', 'back', 'top', 'bottom'},
    }

    # Find maximum spanning tree (minimizes cuts)
    visited = {'front'}
    tree_edges = []
    frontier = [('front', n) for n in cube['front']]

    while frontier and len(visited) < 6:
        # Greedy: take first available edge
        for i, (src, dst) in enumerate(frontier):
            if dst not in visited:
                visited.add(dst)
                tree_edges.append((src, dst))
                frontier.extend((dst, n) for n in cube[dst] if n not in visited)
                break

    print(f"Interior edges (kept): {tree_edges}")
    print(f"Number of cuts: {12 - len(tree_edges)}")

    # Output:
    # Interior edges: [('front', 'top'), ('front', 'bottom'), ('front', 'left'),
    #                  ('front', 'right'), ('right', 'back')]
    # Number of cuts: 7  (but 5 unique interior edges in a tree of 6 nodes)
```

---

## Example 2: Unwrapping a Low-Poly Character

### Mesh Description

```
Simple humanoid: ~50 faces
- Head: 8 faces (roughly spherical)
- Torso: 12 faces (box-like)
- Arms: 8 faces each (cylinders)
- Legs: 10 faces each (cylinders)
```

### Natural Clusters (Body Parts)

Each body part forms a natural cluster:

```
Cluster 1: Head (8 faces)
  - All 8 faces connected internally
  - Interior edges: ~12
  - Seams: 2-3 (to open the "sphere")

Cluster 2: Torso (12 faces)
  - Interior edges: ~16
  - Seams: 4 (one at neck, two at shoulders, one at waist)

Cluster 3-4: Arms (8 faces each)
  - Interior edges: ~10 each
  - Seams: 1-2 each (along length)

Cluster 5-6: Legs (10 faces each)
  - Interior edges: ~14 each
  - Seams: 1-2 each
```

### Edge Maximization Strategy

1. **Identify natural boundaries**: Joints have high curvature changes
2. **Cluster by body part**: Maximizes interior connectivity
3. **Cut along long edges**: Less visible seams

```python
def cluster_character(mesh):
    # Weight edges by:
    # 1. Dihedral angle (flat connections = high weight)
    # 2. Not at joint boundaries (joint = low weight)

    edge_weights = {}
    for edge in mesh.edges:
        dihedral = mesh.dihedral_angle(edge)
        is_joint = mesh.is_joint_edge(edge)  # Custom detection

        weight = (180 - dihedral) / 180.0
        if is_joint:
            weight *= 0.1  # Heavily penalize joint edges

        edge_weights[edge] = weight

    # Maximum spanning forest with weights
    clusters = max_spanning_forest(mesh, edge_weights)

    return clusters
```

---

## Example 3: Cylindrical Object

### Problem

Unwrap a cylinder with end caps.

```
       ___
      /   \
     |     |  <- Top cap
     +-----+
     |     |
     |     |  <- Side (main body)
     |     |
     +-----+
     |     |  <- Bottom cap
      \___/
```

### Analysis

```
Top cap: ~8-12 triangular faces (pie slices)
Side: ~16-24 quad faces (grid wrapping around)
Bottom cap: ~8-12 triangular faces
```

### Unwrapping Options

**Option A: Three separate islands**
```
   [ Top Cap ]    [ Side (rectangle) ]    [ Bottom Cap ]

Cut edges: All edges connecting caps to side (~16-24)
Interior edges in each cluster: Maximized within region
```

**Option B: Connected T-shape**
```
   [ Top ]
   [  |  ]
   [ Side ]
   [  |  ]
   [Bottom]

Cut edges: ~8 (one seam along side, plus to open caps)
Interior edges: More, but higher distortion on caps
```

### Algorithm Selection

```python
def unwrap_cylinder(mesh, prioritize_edges=True):
    if prioritize_edges:
        # Option B: Maximize connectivity
        # Accept higher distortion for fewer seams
        return unwrap_connected(mesh, max_distortion=0.4)
    else:
        # Option A: Minimize distortion
        # Accept more seams for better texture quality
        return unwrap_separate_regions(mesh, max_distortion=0.15)
```

---

## Example 4: Complex Object - Low-Poly Tree

### Mesh Structure

```
Trunk: Cylinder (16 faces)
Branches: 3-4 smaller cylinders (8 faces each)
Foliage: Irregular blob (24 faces)

Total: ~64 faces
```

### Clustering Strategy

```python
def cluster_tree(mesh):
    # Step 1: Identify geometric features
    features = {
        'trunk': detect_cylindrical_region(mesh, axis='vertical'),
        'branches': detect_small_cylinders(mesh),
        'foliage': mesh.faces - trunk - branches
    }

    # Step 2: Create weighted dual graph
    dual = build_dual_graph(mesh)

    for edge in dual.edges:
        f1, f2 = edge.faces

        # Same feature type = high weight (keep connected)
        if get_feature(f1) == get_feature(f2):
            edge.weight = 1.0
        else:
            # Different features = low weight (prefer to cut)
            edge.weight = 0.2

    # Step 3: Cluster with feature-aware weights
    clusters = max_spanning_forest(dual)

    return clusters
```

### Expected Result

```
Cluster 1: Trunk (16 faces)
  - Unwraps to rectangle
  - Interior edges: 14
  - Cut: 2 (one seam)

Cluster 2-4: Branches (8 faces each)
  - Unwrap to small rectangles
  - Interior edges: 6 each
  - Cut: 2 each

Cluster 5: Foliage (24 faces)
  - Unwraps with some distortion
  - Interior edges: ~30
  - Cut: 3-4

Total interior edges: 14 + 18 + 30 = 62
Total cuts: 2 + 6 + 4 = 12
```

---

## Example 5: Step-by-Step Walkthrough

Let's trace through the algorithm on a simple 8-face mesh.

### Input Mesh: Octahedron (8 triangular faces)

```
Face adjacencies:
  F0: F1, F2, F3
  F1: F0, F2, F4
  F2: F0, F1, F5
  F3: F0, F4, F6
  F4: F1, F3, F7
  F5: F2, F6, F7
  F6: F3, F5, F7
  F7: F4, F5, F6
```

### Step 1: Build Dual Graph

```
Nodes: F0, F1, F2, F3, F4, F5, F6, F7
Edges (12 total):
  F0-F1, F0-F2, F0-F3
  F1-F2, F1-F4
  F2-F5
  F3-F4, F3-F6
  F4-F7
  F5-F6, F5-F7
  F6-F7
```

### Step 2: Compute Edge Weights

```python
# Simplified: all edges have weight 1.0 (equally preferred)
weights = {edge: 1.0 for edge in dual_edges}
```

### Step 3: Maximum Spanning Tree

Using Kruskal's algorithm:

```
Start: 8 components {F0}, {F1}, {F2}, {F3}, {F4}, {F5}, {F6}, {F7}

Add F0-F1: Merge {F0, F1}
Add F0-F2: Merge {F0, F1, F2}
Add F0-F3: Merge {F0, F1, F2, F3}
Add F1-F4: Merge {F0, F1, F2, F3, F4}
Add F2-F5: Merge {F0, F1, F2, F3, F4, F5}
Add F3-F6: Merge {F0, F1, F2, F3, F4, F5, F6}
Add F4-F7: Merge all into one component

Tree edges: F0-F1, F0-F2, F0-F3, F1-F4, F2-F5, F3-F6, F4-F7
Interior edges: 7
Cut edges: 12 - 7 = 5
```

### Step 4: Check Distortion

```python
estimated_distortion = estimate_distortion(all_8_faces)
if estimated_distortion > max_distortion:
    # Need to split into multiple clusters
    clusters = split_high_distortion_cluster(all_faces)
```

For an octahedron with moderate size, one cluster is likely fine.

### Step 5: Flatten

```
Result: Single UV island with 8 triangles
Cut edges form the boundary of the island
Interior edges are shared between adjacent triangles in UV space
```

### Visualization

```
3D Mesh              UV Unwrap

    /\                  /\
   /  \                /  \
  /----\       ->     /----\----\
  \    /              \    /    /
   \  /                \  /----/
    \/                  \/----/
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Over-connected Clusters

**Problem**: Algorithm creates one huge cluster with too much distortion.

**Solution**: Add distortion threshold check during merging.

```python
if estimated_distortion > threshold:
    reject_merge()  # Don't add this face/cluster
```

### Pitfall 2: Fragmented Clustering

**Problem**: Too many small clusters, lots of seams.

**Solution**: Post-process to merge tiny clusters.

```python
for cluster in clusters:
    if len(cluster) < min_size:
        merge_with_best_neighbor(cluster)
```

### Pitfall 3: Ignoring Geometric Features

**Problem**: Cuts go through important features (eyes, logos).

**Solution**: Add feature-aware edge weights.

```python
if edge_crosses_feature(edge):
    weight *= 0.01  # Strongly prefer to cut here instead
```

### Pitfall 4: Non-Manifold Geometry

**Problem**: Algorithm assumes clean manifold mesh.

**Solution**: Preprocess to split non-manifold edges.

```python
mesh = fix_non_manifold(mesh)  # Split edges shared by >2 faces
```

---

## Summary Checklist

For any polygon unwrapping task:

- [ ] Understand mesh topology (manifold, genus)
- [ ] Build dual graph from face adjacency
- [ ] Define edge weights based on:
  - [ ] Dihedral angles
  - [ ] Edge lengths
  - [ ] Feature boundaries
- [ ] Choose algorithm based on mesh size and requirements
- [ ] Set appropriate distortion threshold
- [ ] Run clustering algorithm
- [ ] Validate: check distortion, cluster sizes
- [ ] Post-process: merge small clusters, split large ones
- [ ] Flatten each cluster using appropriate method
- [ ] Pack UV islands efficiently
