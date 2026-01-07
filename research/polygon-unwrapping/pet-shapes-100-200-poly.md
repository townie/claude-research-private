# Unwrapping Pet Shapes: 100-200 Polygon Models

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Overview

This document focuses specifically on unwrapping **low-poly pet models** (dogs, cats, birds, etc.) with **100-200 polygons**. These models have unique characteristics that inform our clustering strategy.

---

## 1. Anatomy of a Low-Poly Pet

### Typical Polygon Distribution (150-poly dog example)

```
Body Part          | Polygons | % of Total
-------------------|----------|------------
Body/Torso         | 40-50    | 27-33%
Head               | 25-35    | 17-23%
Legs (x4)          | 40-60    | 27-40%
Tail               | 8-12     | 5-8%
Ears (x2)          | 10-15    | 7-10%
```

### Topology Diagram

```
         (ears)
           ▲▲
          /  \
    +----[HEAD]----+
    |      |       |
    |   [NECK]     |
    |      |       |
+--[BODY/TORSO]----+----[TAIL]
    |   |    |     |
   [L1][L2][L3][L4]
    |   |    |   |
   (paws)
```

---

## 2. Natural Cluster Boundaries for Pets

### Primary Seam Locations

For pets, natural seams occur at:

1. **Joints** - Where body parts connect (high curvature change)
2. **Underside** - Less visible in most viewing angles
3. **Inner legs** - Hidden from common views
4. **Behind ears** - Concealed area

### Recommended Cluster Strategy

```
Strategy A: Anatomical Clusters (5-7 islands)
├── Head + Ears (one island)
├── Body (one island, seam on belly)
├── Each leg (4 islands)
└── Tail (attached to body OR separate)

Strategy B: Minimal Clusters (2-3 islands)
├── Front half (head + front legs + front body)
├── Back half (back body + back legs + tail)
└── Optional: Separate head

Strategy C: Maximum Edge (1-2 islands)
├── Everything connected via strategic cuts
└── Single seam running along underside
```

---

## 3. Algorithm Adaptation for Pets

### 3.1 Joint Detection

Pets have distinct joints that make natural cut points.

```python
def detect_pet_joints(mesh):
    """
    Identify joint locations in pet mesh.
    Joints have:
    - High curvature change
    - Narrow cross-section
    - Connect larger body parts
    """
    joints = []

    for vertex in mesh.vertices:
        # Check 1: High curvature
        curvature = mesh.get_vertex_curvature(vertex)

        # Check 2: Narrow cross-section (low vertex valence in local region)
        local_area = mesh.get_local_area(vertex, radius=2)

        # Check 3: Connects different "mass" regions
        neighbors = mesh.get_vertex_neighbors(vertex, depth=3)
        mass_variance = compute_mass_distribution_variance(mesh, neighbors)

        joint_score = (
            0.4 * normalize(curvature) +
            0.3 * (1.0 / local_area) +
            0.3 * mass_variance
        )

        if joint_score > JOINT_THRESHOLD:
            joints.append(vertex)

    return joints


def compute_mass_distribution_variance(mesh, vertices):
    """
    High variance = connecting different sized regions = likely joint.
    """
    # Flood fill from each vertex, measure reachable area
    areas = []
    for v in vertices:
        area = flood_fill_area(mesh, v, max_depth=5)
        areas.append(area)

    return np.var(areas)
```

### 3.2 Pet-Specific Edge Weights

```python
def pet_edge_weight(mesh, face1_id, face2_id, edge):
    """
    Edge weight function optimized for pet meshes.
    """
    # Base: dihedral angle (standard)
    w_dihedral = dihedral_angle_weight(mesh, face1_id, face2_id)

    # Joint penalty: cut at joints
    joint_vertices = detect_pet_joints(mesh)  # cached
    edge_verts = mesh.get_edge_vertices(edge)
    is_at_joint = any(v in joint_vertices for v in edge_verts)
    w_joint = 0.2 if is_at_joint else 1.0

    # Visibility: prefer cuts on underside/inner surfaces
    edge_center = mesh.get_edge_center(edge)
    is_underside = edge_center[1] < mesh.center[1]  # Y-down
    is_inner_leg = is_inner_leg_region(mesh, edge)
    w_visibility = 0.7 if (is_underside or is_inner_leg) else 1.0

    # Combine (multiplicative - any low weight forces cut)
    return w_dihedral * w_joint * w_visibility
```

### 3.3 Body Part Clustering

```python
def cluster_pet_by_anatomy(mesh):
    """
    Cluster pet mesh by anatomical regions.
    """
    # Step 1: Detect body parts using heuristics
    body_parts = detect_body_parts(mesh)
    # Returns: {'head': [faces], 'body': [faces], 'legs': [[f], [f], [f], [f]], ...}

    # Step 2: For each body part, maximize internal edges
    clusters = []

    for part_name, faces in body_parts.items():
        if len(faces) < 3:
            # Too small, merge with nearest neighbor
            continue

        # Build sub-mesh for this part
        sub_mesh = extract_submesh(mesh, faces)

        # Cluster within part (usually 1 cluster per part)
        part_clusters = cluster_mesh(sub_mesh, max_distortion=0.3)
        clusters.extend(part_clusters)

    return clusters


def detect_body_parts(mesh):
    """
    Heuristic detection of pet body parts.
    """
    # Use geometric analysis
    center = mesh.bounding_box_center()
    bbox = mesh.bounding_box()

    body_parts = {
        'head': [],
        'body': [],
        'front_legs': [],
        'back_legs': [],
        'tail': [],
        'ears': []
    }

    for face in mesh.faces:
        centroid = mesh.get_face_centroid(face)

        # Simple heuristics based on position
        x, y, z = centroid - center
        length = bbox[1][0] - bbox[0][0]  # Assuming X is length axis

        # Head: front of mesh
        if x > length * 0.3:
            body_parts['head'].append(face.id)
        # Tail: back of mesh, narrow
        elif x < -length * 0.35 and is_narrow_region(mesh, face):
            body_parts['tail'].append(face.id)
        # Legs: below center Y, four quadrants
        elif y < center[1]:
            # Determine which leg
            leg_idx = classify_leg(centroid, center)
            body_parts['front_legs' if leg_idx < 2 else 'back_legs'].append(face.id)
        # Body: everything else
        else:
            body_parts['body'].append(face.id)

    return body_parts
```

---

## 4. Optimal Cluster Counts for Pets

### Analysis by Polygon Count

| Poly Count | Recommended Clusters | Interior Edge % |
|------------|---------------------|-----------------|
| 100-120    | 3-4                 | 85-90%          |
| 120-150    | 4-5                 | 87-92%          |
| 150-180    | 5-6                 | 88-93%          |
| 180-200    | 5-7                 | 89-94%          |

### Trade-off Analysis

```
Fewer Clusters (2-3):
  ✓ Maximum edge connectivity
  ✓ Fewer seams to hide
  ✗ Higher distortion
  ✗ Harder to texture efficiently

More Clusters (6-8):
  ✓ Lower distortion per island
  ✓ Easier texture painting
  ✗ More seams
  ✗ Less efficient UV space use
```

---

## 5. Example: 150-Poly Dog

### Mesh Breakdown

```
Face distribution:
- Head: 28 faces (including ears)
- Body: 44 faces
- Front left leg: 14 faces
- Front right leg: 14 faces
- Back left leg: 16 faces
- Back right leg: 16 faces
- Tail: 10 faces
- Neck (transition): 8 faces

Total edges: ~225
Interior edges goal: 200+ (89%+)
```

### Clustering Steps

```python
def unwrap_150_poly_dog(mesh):
    # Step 1: Identify anatomy
    parts = detect_body_parts(mesh)

    print(f"Detected parts: {list(parts.keys())}")

    # Step 2: Build weighted dual graph
    dual = build_dual_graph(mesh)

    for edge in dual.edges:
        edge.weight = pet_edge_weight(mesh, edge.f1, edge.f2, edge.id)

    # Step 3: Maximum spanning forest with anatomical hints
    # Force cuts at major joints
    joint_edges = get_joint_edges(mesh, parts)
    for edge in joint_edges:
        dual.edges[edge].weight *= 0.1  # Strong cut preference

    # Step 4: Cluster
    clusters = maximum_spanning_forest(dual, max_distortion=0.25)

    # Step 5: Post-process
    # Merge tiny clusters (< 5 faces) with neighbors
    clusters = merge_small_clusters(clusters, min_size=5)

    # Step 6: Report
    interior_edges = count_interior_edges(mesh, clusters)
    total_edges = len(mesh.interior_edges())

    print(f"Clusters: {len(clusters)}")
    print(f"Interior edges: {interior_edges}/{total_edges} ({100*interior_edges/total_edges:.1f}%)")

    return clusters
```

### Expected Output

```
Detected parts: ['head', 'body', 'front_legs', 'back_legs', 'tail']

Clusters: 5
  Cluster 0 (Head+Ears): 28 faces, 38 interior edges
  Cluster 1 (Body+Tail): 54 faces, 76 interior edges
  Cluster 2 (Front legs): 28 faces, 38 interior edges
  Cluster 3 (Back left leg): 16 faces, 20 interior edges
  Cluster 4 (Back right leg): 16 faces, 20 interior edges

Interior edges: 192/225 (85.3%)
Cut edges: 33 (at joints, underside)
```

---

## 6. UV Layout Patterns for Pets

### Pattern A: Spread Layout

```
+-------+   +-------+
| HEAD  |   | BODY  |
+-------+   +-------+

+--+ +--+   +--+ +--+
|FL| |FR|   |BL| |BR|
+--+ +--+   +--+ +--+

      +------+
      | TAIL |
      +------+
```

### Pattern B: Connected Layout (Maximum Edges)

```
    +-------+
    | HEAD  |
    +---+---+
        |
+--++---+---++--+
|FL||  BODY  ||FR|
+--++---+---++--+
        |
    +---+---+
    |BL| |BR|
    +--+ +--+
        |
    +------+
    | TAIL |
    +------+
```

### Pattern C: Symmetry-Optimized

```
+-------+-------+
| LEFT  | RIGHT |
| HALF  | HALF  |
+-------+-------+
   (mirrored UV mapping)
```

---

## 7. Distortion Budgets for Pet Parts

Different body parts can tolerate different distortion levels:

```python
DISTORTION_BUDGETS = {
    'head': 0.15,      # Low - faces are important
    'body': 0.25,      # Medium - large, visible area
    'legs': 0.35,      # Higher - cylindrical, less detail
    'tail': 0.40,      # Highest - small, cylindrical
    'ears': 0.20,      # Medium-low - distinctive feature
}

def cluster_with_budgets(mesh, parts, budgets):
    """
    Cluster each part with its own distortion budget.
    """
    clusters = []

    for part_name, faces in parts.items():
        budget = budgets.get(part_name, 0.25)
        sub_mesh = extract_submesh(mesh, faces)

        part_clusters = cluster_mesh(sub_mesh, max_distortion=budget)
        clusters.extend(part_clusters)

    return clusters
```

---

## 8. Common Pet-Specific Issues

### Issue 1: Leg Merging

**Problem**: Algorithm merges all 4 legs into one cluster.

**Solution**: Pre-segment legs before clustering.

```python
def prevent_leg_merging(mesh, dual_graph):
    """Add strong penalties for edges between different legs."""
    legs = detect_legs(mesh)

    for edge in dual_graph.edges:
        leg1 = get_leg_id(edge.f1, legs)
        leg2 = get_leg_id(edge.f2, legs)

        if leg1 is not None and leg2 is not None and leg1 != leg2:
            edge.weight *= 0.01  # Force cut between legs
```

### Issue 2: Head-Body Over-Connection

**Problem**: Head and body form one large cluster with high distortion.

**Solution**: Enforce neck cut.

```python
def enforce_neck_cut(mesh, dual_graph):
    """Identify and cut at neck region."""
    neck_edges = detect_neck_edges(mesh)

    for edge in neck_edges:
        dual_graph.edges[edge].weight *= 0.1
```

### Issue 3: Ear Fragmentation

**Problem**: Each ear face becomes its own cluster.

**Solution**: Group ears with head, or use minimum cluster size.

```python
def handle_ears(clusters, mesh, min_ear_size=3):
    """Merge tiny ear fragments with head cluster."""
    ear_faces = detect_ear_faces(mesh)
    head_cluster = find_head_cluster(clusters, mesh)

    for cluster in clusters[:]:  # Copy to allow modification
        if len(cluster) < min_ear_size:
            if any(f in ear_faces for f in cluster):
                # Merge with head
                head_cluster.update(cluster)
                clusters.remove(cluster)
```

### Issue 4: Asymmetric Cuts

**Problem**: Left and right sides have different seam patterns.

**Solution**: Mirror-aware clustering.

```python
def symmetric_clustering(mesh):
    """
    Ensure symmetric pets get symmetric UV seams.
    """
    # Detect symmetry plane
    symmetry_plane = detect_symmetry_plane(mesh)

    if symmetry_plane is None:
        return standard_clustering(mesh)

    # Cluster one half
    left_half = get_mesh_half(mesh, symmetry_plane, 'left')
    left_clusters = cluster_mesh(left_half)

    # Mirror to other half
    right_clusters = mirror_clusters(left_clusters, symmetry_plane)

    return left_clusters + right_clusters
```

---

## 9. Quick Reference: 100-200 Poly Pets

### Recommended Settings

```python
PET_UNWRAP_CONFIG = {
    # Clustering
    'max_distortion': 0.25,
    'min_cluster_size': 5,
    'target_clusters': 5,  # Head, body, 4 legs (tail attached to body)

    # Edge weights
    'dihedral_weight': 0.4,
    'joint_penalty': 0.8,
    'visibility_weight': 0.3,

    # Post-processing
    'merge_threshold': 5,  # Merge clusters smaller than this
    'split_threshold': 0.35,  # Split clusters with distortion above this
}
```

### Expected Results

| Pet Type | Poly Count | Clusters | Interior Edges |
|----------|------------|----------|----------------|
| Simple dog | 100 | 4 | ~85% |
| Cat | 120 | 4-5 | ~87% |
| Detailed dog | 150 | 5-6 | ~89% |
| Bird | 100 | 3-4 | ~90% |
| Rabbit | 130 | 4-5 | ~88% |
| Horse | 200 | 6-7 | ~91% |
