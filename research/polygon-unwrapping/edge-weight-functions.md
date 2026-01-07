# Edge Weight Functions: Deep Dive

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

The choice of edge weight function is **critical** for determining which edges stay connected (interior) vs. get cut (seams). This document breaks down various weighting strategies.

---

## 1. Geometric Weight Functions

### 1.1 Dihedral Angle Weight

The angle between two adjacent faces. Flat connections are preferred.

```python
import numpy as np

def dihedral_angle_weight(mesh, face1_id, face2_id):
    """
    Weight based on dihedral angle between faces.

    Flat (180°) -> weight = 1.0 (keep connected)
    Sharp (90°) -> weight = 0.5
    Very sharp (0°) -> weight = 0.0 (prefer to cut)
    """
    n1 = mesh.get_face_normal(face1_id)
    n2 = mesh.get_face_normal(face2_id)

    # Angle between normals
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    # Dihedral angle is supplement of angle between normals
    dihedral = 180.0 - angle_deg

    # Linear weight: 180° -> 1.0, 0° -> 0.0
    weight = dihedral / 180.0

    return weight
```

**Variations:**

```python
def dihedral_weight_exponential(dihedral_deg, sharpness=2.0):
    """Exponential falloff for sharper preference."""
    normalized = dihedral_deg / 180.0
    return normalized ** sharpness

def dihedral_weight_threshold(dihedral_deg, threshold=120.0):
    """Binary: above threshold = 1.0, below = 0.1"""
    return 1.0 if dihedral_deg >= threshold else 0.1

def dihedral_weight_sigmoid(dihedral_deg, midpoint=90.0, steepness=0.1):
    """Smooth sigmoid transition."""
    return 1.0 / (1.0 + np.exp(-steepness * (dihedral_deg - midpoint)))
```

### 1.2 Edge Length Weight

Longer edges create more visible seams when cut.

```python
def edge_length_weight(mesh, edge):
    """
    Prefer to keep longer edges (cutting them = more visible seam).
    """
    v1, v2 = mesh.get_edge_vertices(edge)
    length = np.linalg.norm(v2 - v1)

    # Normalize by average edge length
    avg_length = mesh.average_edge_length()
    normalized_length = length / avg_length

    # Longer = higher weight
    return min(normalized_length, 2.0)  # Cap at 2x

def edge_length_weight_inverse(mesh, edge):
    """
    Alternative: prefer to CUT longer edges (less distortion).
    Use when texture stretching is more important than seam visibility.
    """
    v1, v2 = mesh.get_edge_vertices(edge)
    length = np.linalg.norm(v2 - v1)
    avg_length = mesh.average_edge_length()

    return avg_length / (length + 0.001)  # Inverse relationship
```

### 1.3 Curvature Continuity Weight

Prefer to keep edges where curvature is similar on both sides.

```python
def curvature_continuity_weight(mesh, face1_id, face2_id):
    """
    High weight if curvature is similar across the edge.
    Low weight if there's a curvature discontinuity.
    """
    # Get mean curvature for each face (average of vertex curvatures)
    c1 = mesh.get_face_mean_curvature(face1_id)
    c2 = mesh.get_face_mean_curvature(face2_id)

    # Curvature difference
    diff = abs(c1 - c2)

    # Normalize by max curvature in mesh
    max_curv = mesh.max_curvature()
    normalized_diff = diff / (max_curv + 0.001)

    # Invert: small difference = high weight
    weight = 1.0 - min(normalized_diff, 1.0)

    return weight
```

### 1.4 Area Ratio Weight

Prefer edges between similarly-sized faces.

```python
def area_ratio_weight(mesh, face1_id, face2_id):
    """
    Prefer edges between faces of similar size.
    Cutting between very different sized faces can cause UV issues.
    """
    a1 = mesh.get_face_area(face1_id)
    a2 = mesh.get_face_area(face2_id)

    # Ratio closer to 1.0 = higher weight
    ratio = min(a1, a2) / max(a1, a2)

    return ratio
```

---

## 2. Semantic/Feature-Based Weights

### 2.1 Feature Edge Detection

Identify edges that should be cut (sharp features) vs. kept (smooth surfaces).

```python
def feature_edge_weight(mesh, edge, feature_angle_threshold=30.0):
    """
    Detect feature edges (sharp creases, corners).
    These should be CUT, not kept.
    """
    f1, f2 = mesh.get_edge_faces(edge)
    if f2 is None:  # Boundary edge
        return 0.0

    n1 = mesh.get_face_normal(f1)
    n2 = mesh.get_face_normal(f2)

    angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))

    if angle > feature_angle_threshold:
        # This is a feature edge - prefer to cut here
        return 0.1
    else:
        # Smooth surface - prefer to keep
        return 1.0
```

### 2.2 Material Boundary Weight

Cut at material boundaries for cleaner texture mapping.

```python
def material_boundary_weight(mesh, face1_id, face2_id):
    """
    If faces have different materials, prefer to cut between them.
    """
    mat1 = mesh.get_face_material(face1_id)
    mat2 = mesh.get_face_material(face2_id)

    if mat1 != mat2:
        return 0.05  # Strongly prefer to cut
    else:
        return 1.0   # Keep connected
```

### 2.3 UV Island Hint Weight

Use existing UV seams or artist hints.

```python
def uv_hint_weight(mesh, edge, existing_seams):
    """
    Respect existing UV seams from artist or previous unwrap.
    """
    if edge in existing_seams:
        return 0.0  # Must cut here
    else:
        return 1.0
```

---

## 3. Distortion-Predictive Weights

### 3.1 Stretch Potential Weight

Estimate how much cutting this edge would reduce stretch.

```python
def stretch_potential_weight(mesh, face1_id, face2_id):
    """
    Estimate stretch if these faces are in the same UV island.
    Higher potential stretch = lower weight (prefer to cut).
    """
    # Get Gaussian curvature for region around edge
    edge_vertices = mesh.get_shared_vertices(face1_id, face2_id)

    total_angle_deficit = 0.0
    for v in edge_vertices:
        angles = mesh.get_vertex_angles(v)
        deficit = 2 * np.pi - sum(angles)
        total_angle_deficit += abs(deficit)

    # High deficit = high stretch potential = low weight
    weight = 1.0 / (1.0 + total_angle_deficit)

    return weight
```

### 3.2 Developability Score

How close is this region to being developable (flattenable without distortion)?

```python
def developability_weight(mesh, face1_id, face2_id):
    """
    Score based on how developable the combined region would be.
    """
    # Get all vertices in both faces
    verts1 = set(mesh.faces[face1_id].vertices)
    verts2 = set(mesh.faces[face2_id].vertices)
    all_verts = verts1 | verts2

    # Sum of absolute angle deficits
    total_deficit = sum(
        abs(mesh.get_angle_deficit(v)) for v in all_verts
    )

    # Normalize by number of vertices
    avg_deficit = total_deficit / len(all_verts)

    # Lower deficit = more developable = higher weight
    weight = np.exp(-avg_deficit)

    return weight
```

---

## 4. Composite Weight Functions

### 4.1 Weighted Combination

```python
def composite_weight(mesh, face1_id, face2_id, edge):
    """
    Combine multiple weight functions with configurable importance.
    """
    weights = {
        'dihedral': 0.4,
        'edge_length': 0.2,
        'curvature': 0.2,
        'feature': 0.2,
    }

    w_dihedral = dihedral_angle_weight(mesh, face1_id, face2_id)
    w_length = edge_length_weight(mesh, edge)
    w_curvature = curvature_continuity_weight(mesh, face1_id, face2_id)
    w_feature = feature_edge_weight(mesh, edge)

    combined = (
        weights['dihedral'] * w_dihedral +
        weights['edge_length'] * w_length +
        weights['curvature'] * w_curvature +
        weights['feature'] * w_feature
    )

    return combined
```

### 4.2 Multiplicative Combination

```python
def multiplicative_weight(mesh, face1_id, face2_id, edge):
    """
    Multiply weights - any low weight vetoes the edge.
    """
    w1 = dihedral_angle_weight(mesh, face1_id, face2_id)
    w2 = feature_edge_weight(mesh, edge)
    w3 = material_boundary_weight(mesh, face1_id, face2_id)

    return w1 * w2 * w3
```

### 4.3 Hierarchical/Priority-Based

```python
def priority_weight(mesh, face1_id, face2_id, edge):
    """
    Check constraints in priority order.
    """
    # Priority 1: Material boundaries (must cut)
    if mesh.get_face_material(face1_id) != mesh.get_face_material(face2_id):
        return 0.0

    # Priority 2: Feature edges (should cut)
    if is_feature_edge(mesh, edge, threshold=45.0):
        return 0.1

    # Priority 3: Existing seams (respect artist intent)
    if edge in mesh.existing_seams:
        return 0.05

    # Default: use geometric weight
    return dihedral_angle_weight(mesh, face1_id, face2_id)
```

---

## 5. Adaptive Weight Functions

### 5.1 Mesh-Aware Normalization

```python
class AdaptiveWeightCalculator:
    def __init__(self, mesh):
        self.mesh = mesh
        self._precompute_statistics()

    def _precompute_statistics(self):
        """Compute mesh-wide statistics for normalization."""
        dihedrals = []
        lengths = []
        curvatures = []

        for edge in self.mesh.edges:
            f1, f2 = self.mesh.get_edge_faces(edge)
            if f2 is not None:
                dihedrals.append(self._raw_dihedral(f1, f2))
            lengths.append(self._raw_length(edge))

        for v in self.mesh.vertices:
            curvatures.append(self.mesh.get_vertex_curvature(v))

        self.dihedral_mean = np.mean(dihedrals)
        self.dihedral_std = np.std(dihedrals)
        self.length_mean = np.mean(lengths)
        self.length_std = np.std(lengths)
        self.curvature_range = max(curvatures) - min(curvatures)

    def normalized_dihedral_weight(self, face1_id, face2_id):
        """Z-score normalized dihedral weight."""
        raw = self._raw_dihedral(face1_id, face2_id)
        z_score = (raw - self.dihedral_mean) / (self.dihedral_std + 0.001)

        # Convert to 0-1 range using sigmoid
        return 1.0 / (1.0 + np.exp(-z_score))
```

### 5.2 Region-Aware Weights

```python
def region_aware_weight(mesh, face1_id, face2_id, current_cluster):
    """
    Weight depends on the cluster being built.
    """
    # If cluster is already large, be more strict about adding
    cluster_size = len(current_cluster)
    size_penalty = 1.0 / (1.0 + 0.01 * cluster_size)

    # If cluster has high distortion, be more strict
    cluster_distortion = estimate_cluster_distortion(mesh, current_cluster)
    distortion_penalty = 1.0 - cluster_distortion

    # Base geometric weight
    base_weight = dihedral_angle_weight(mesh, face1_id, face2_id)

    return base_weight * size_penalty * distortion_penalty
```

---

## 6. Weight Function Selection Guide

| Use Case | Recommended Weights | Rationale |
|----------|---------------------|-----------|
| **Game assets** | Dihedral + Feature + Length | Minimize visible seams |
| **CAD/Technical** | Feature + Material | Clean boundaries |
| **Organic models** | Dihedral + Curvature | Smooth surface flow |
| **Low-poly art** | Dihedral only | Simple, predictable |
| **Automatic UV** | Composite (all) | Balanced approach |
| **Artist-guided** | UV Hints + Dihedral | Respect artist intent |

---

## 7. Tuning Weight Parameters

### Parameter Sensitivity Analysis

```python
def analyze_weight_sensitivity(mesh, weight_func, param_name, param_range):
    """
    Analyze how changing a weight parameter affects results.
    """
    results = []

    for param_value in param_range:
        # Set parameter
        set_weight_param(weight_func, param_name, param_value)

        # Run clustering
        clusters = cluster_mesh(mesh, weight_func)

        # Measure quality
        total_interior = count_interior_edges(clusters)
        total_distortion = measure_total_distortion(clusters)
        num_clusters = len(clusters)

        results.append({
            'param': param_value,
            'interior_edges': total_interior,
            'distortion': total_distortion,
            'num_clusters': num_clusters,
        })

    return results
```

### Visualization

```python
def visualize_edge_weights(mesh, weight_func):
    """
    Color-code edges by their weight for debugging.
    """
    edge_colors = {}

    for edge in mesh.edges:
        f1, f2 = mesh.get_edge_faces(edge)
        if f2 is not None:
            weight = weight_func(mesh, f1, f2, edge)
            # Map weight to color: red (low) -> green (high)
            edge_colors[edge] = weight_to_color(weight)

    return edge_colors

def weight_to_color(weight):
    """Convert weight [0,1] to RGB color."""
    # Red (cut) -> Yellow -> Green (keep)
    r = 1.0 - weight
    g = weight
    b = 0.0
    return (r, g, b)
```
