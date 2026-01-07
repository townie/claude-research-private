# Quality Metrics for Polygon Unwrapping

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Overview

How do we measure if our unwrapping is "good"? This document breaks down the key quality metrics for evaluating edge-maximized polygon unwrapping.

---

## 1. Edge Connectivity Metrics

### 1.1 Interior Edge Ratio

The primary metric for edge-maximized unwrapping.

```python
def interior_edge_ratio(mesh, clusters):
    """
    Ratio of edges that remain connected (interior) vs. total edges.

    Higher is better. Maximum possible depends on topology.
    """
    total_interior_edges = 0
    total_edges = 0

    for edge in mesh.interior_edges():  # Exclude boundary edges
        total_edges += 1

        f1, f2 = mesh.get_edge_faces(edge)
        c1 = get_cluster_id(f1, clusters)
        c2 = get_cluster_id(f2, clusters)

        if c1 == c2:
            total_interior_edges += 1

    ratio = total_interior_edges / total_edges
    return ratio

# Interpretation:
# 0.95+ : Excellent (near-optimal)
# 0.90-0.95 : Very good
# 0.85-0.90 : Good
# 0.80-0.85 : Acceptable
# < 0.80 : Poor (too many seams)
```

### 1.2 Seam Length

Total length of all cut edges (seams).

```python
def total_seam_length(mesh, clusters):
    """
    Sum of lengths of all cut edges.

    Lower is better (shorter seams = less visible).
    """
    seam_length = 0.0

    for edge in mesh.interior_edges():
        f1, f2 = mesh.get_edge_faces(edge)
        c1 = get_cluster_id(f1, clusters)
        c2 = get_cluster_id(f2, clusters)

        if c1 != c2:  # This is a cut edge
            v1, v2 = mesh.get_edge_vertices(edge)
            length = np.linalg.norm(v2 - v1)
            seam_length += length

    return seam_length

def normalized_seam_length(mesh, clusters):
    """Normalize by total edge length for comparison across meshes."""
    seam = total_seam_length(mesh, clusters)
    total = sum(mesh.get_edge_length(e) for e in mesh.edges)
    return seam / total
```

### 1.3 Seam Visibility Score

Weight seams by their visibility.

```python
def visibility_weighted_seam_score(mesh, clusters, camera_positions=None):
    """
    Seams in visible areas are worse than hidden seams.
    """
    if camera_positions is None:
        # Default: assume view from all sides
        camera_positions = [
            (1, 0, 0), (-1, 0, 0),  # Left/Right
            (0, 1, 0), (0, -1, 0),  # Top/Bottom
            (0, 0, 1), (0, 0, -1),  # Front/Back
        ]

    visibility_score = 0.0

    for edge in mesh.cut_edges(clusters):
        edge_center = mesh.get_edge_center(edge)
        edge_normal = mesh.get_edge_normal(edge)  # Average of face normals

        # How visible is this edge from common views?
        max_visibility = 0.0
        for cam_pos in camera_positions:
            cam_dir = normalize(np.array(cam_pos) - edge_center)
            visibility = max(0, np.dot(edge_normal, cam_dir))
            max_visibility = max(max_visibility, visibility)

        edge_length = mesh.get_edge_length(edge)
        visibility_score += edge_length * max_visibility

    return visibility_score

# Lower is better (seams are hidden)
```

---

## 2. Distortion Metrics

### 2.1 Stretch (L² Metric)

Measures how much triangles are stretched in UV space.

```python
def compute_L2_stretch(mesh, uvs, cluster):
    """
    L² stretch metric per face.

    1.0 = no stretch (ideal)
    > 1.0 = stretched
    < 1.0 = compressed
    """
    stretches = []

    for face_id in cluster:
        # Get 3D triangle
        v0, v1, v2 = mesh.get_face_vertices_3d(face_id)

        # Get UV triangle
        uv0, uv1, uv2 = uvs.get_face_vertices_2d(face_id)

        # Compute Jacobian of the 3D->2D mapping
        J = compute_jacobian(v0, v1, v2, uv0, uv1, uv2)

        # Singular values
        s1, s2 = np.linalg.svd(J, compute_uv=False)

        # L² stretch
        L2 = np.sqrt((s1**2 + s2**2) / 2)
        stretches.append(L2)

    return stretches

def max_stretch(mesh, uvs, clusters):
    """Maximum stretch across all faces."""
    all_stretches = []
    for cluster in clusters:
        all_stretches.extend(compute_L2_stretch(mesh, uvs, cluster))
    return max(all_stretches)

def mean_stretch(mesh, uvs, clusters):
    """Area-weighted mean stretch."""
    total_stretch = 0.0
    total_area = 0.0

    for cluster in clusters:
        for face_id in cluster:
            area = mesh.get_face_area(face_id)
            stretch = compute_L2_stretch(mesh, uvs, {face_id})[0]
            total_stretch += stretch * area
            total_area += area

    return total_stretch / total_area

# Interpretation:
# 1.0 : Perfect (no distortion)
# 1.0-1.1 : Excellent
# 1.1-1.2 : Good
# 1.2-1.5 : Acceptable
# > 1.5 : Poor (visible stretching)
```

### 2.2 Angle Distortion (Conformal Error)

Measures how well angles are preserved.

```python
def angle_distortion(mesh, uvs, face_id):
    """
    Sum of absolute angle differences between 3D and UV.
    """
    # 3D angles
    angles_3d = mesh.get_face_angles(face_id)

    # UV angles
    angles_uv = uvs.get_face_angles(face_id)

    # Total absolute difference
    distortion = sum(abs(a3d - auv) for a3d, auv in zip(angles_3d, angles_uv))

    return distortion  # In radians

def max_angle_distortion(mesh, uvs, clusters):
    """Maximum angle distortion in degrees."""
    max_dist = 0.0
    for cluster in clusters:
        for face_id in cluster:
            dist = angle_distortion(mesh, uvs, face_id)
            max_dist = max(max_dist, np.degrees(dist))
    return max_dist

# Interpretation (degrees):
# < 5° : Excellent (nearly conformal)
# 5-15° : Good
# 15-30° : Acceptable
# > 30° : Poor
```

### 2.3 Area Distortion

Measures how well areas are preserved.

```python
def area_distortion_ratio(mesh, uvs, face_id):
    """
    Ratio of UV area to 3D area.

    1.0 = perfect preservation
    """
    area_3d = mesh.get_face_area(face_id)
    area_uv = uvs.get_face_area(face_id)

    # Normalize UV to same total area as 3D
    scale = mesh.total_area() / uvs.total_area()
    area_uv_scaled = area_uv * scale

    ratio = area_uv_scaled / area_3d
    return ratio

def area_distortion_variance(mesh, uvs, clusters):
    """
    Variance in area ratios.

    0 = all faces scaled equally (ideal)
    Higher = uneven scaling
    """
    ratios = []
    for cluster in clusters:
        for face_id in cluster:
            ratios.append(area_distortion_ratio(mesh, uvs, face_id))

    return np.var(ratios)
```

---

## 3. Cluster Quality Metrics

### 3.1 Cluster Size Distribution

```python
def cluster_size_stats(clusters):
    """
    Statistics about cluster sizes.
    """
    sizes = [len(c) for c in clusters]

    return {
        'min': min(sizes),
        'max': max(sizes),
        'mean': np.mean(sizes),
        'std': np.std(sizes),
        'count': len(clusters),
    }

def cluster_balance_score(clusters):
    """
    How balanced are cluster sizes?

    1.0 = all same size (ideal balance)
    Lower = more imbalanced
    """
    sizes = [len(c) for c in clusters]
    mean = np.mean(sizes)

    if mean == 0:
        return 0

    deviations = [abs(s - mean) / mean for s in sizes]
    avg_deviation = np.mean(deviations)

    return 1.0 - min(avg_deviation, 1.0)
```

### 3.2 Cluster Compactness

Compact clusters are easier to texture.

```python
def cluster_compactness(mesh, cluster):
    """
    Ratio of cluster area to convex hull area.

    1.0 = perfectly compact (convex)
    Lower = more irregular shape
    """
    # Compute UV bounding convex hull
    uv_points = [mesh.get_face_centroid_uv(f) for f in cluster]
    hull = ConvexHull(uv_points)

    cluster_area = sum(mesh.get_face_area_uv(f) for f in cluster)
    hull_area = hull.area

    return cluster_area / hull_area

def average_compactness(mesh, clusters):
    """Average compactness across all clusters."""
    compactnesses = [cluster_compactness(mesh, c) for c in clusters]
    return np.mean(compactnesses)
```

### 3.3 Cluster Boundary Complexity

```python
def boundary_complexity(mesh, cluster):
    """
    Ratio of boundary length to cluster area.

    Lower = simpler boundary (more square-like)
    Higher = complex boundary (more jagged)
    """
    boundary_length = compute_cluster_boundary_length(mesh, cluster)
    area = sum(mesh.get_face_area(f) for f in cluster)

    # Normalize: circle has ratio 2*sqrt(pi/area)
    # Square has ratio 4/sqrt(area)
    normalized = boundary_length / np.sqrt(area)

    return normalized
```

---

## 4. UV Space Efficiency

### 4.1 UV Space Utilization

```python
def uv_utilization(uvs, clusters):
    """
    What fraction of the UV square is actually used?

    Higher is better (efficient texture use).
    """
    # Total area of all UV faces
    used_area = sum(
        uvs.get_face_area(f)
        for cluster in clusters
        for f in cluster
    )

    # UV space is typically [0,1] x [0,1]
    total_area = 1.0

    return used_area / total_area

# Interpretation:
# > 0.85 : Excellent
# 0.70-0.85 : Good
# 0.50-0.70 : Acceptable
# < 0.50 : Poor (wasted texture space)
```

### 4.2 Packing Efficiency

```python
def packing_efficiency(uvs, clusters):
    """
    How well are UV islands packed?

    Ratio of used area to bounding box area.
    """
    # Get bounding box of all UV coordinates
    all_uvs = []
    for cluster in clusters:
        for face_id in cluster:
            all_uvs.extend(uvs.get_face_vertices(face_id))

    min_u = min(uv[0] for uv in all_uvs)
    max_u = max(uv[0] for uv in all_uvs)
    min_v = min(uv[1] for uv in all_uvs)
    max_v = max(uv[1] for uv in all_uvs)

    bbox_area = (max_u - min_u) * (max_v - min_v)
    used_area = sum(
        uvs.get_face_area(f)
        for cluster in clusters
        for f in cluster
    )

    return used_area / bbox_area
```

### 4.3 Island Padding Score

```python
def island_padding_score(uvs, clusters, min_padding=0.01):
    """
    Check if islands have sufficient padding (gap between them).

    Returns fraction of island pairs with adequate padding.
    """
    adequate_pairs = 0
    total_pairs = 0

    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i >= j:
                continue

            total_pairs += 1
            min_distance = compute_min_distance_between_clusters(uvs, c1, c2)

            if min_distance >= min_padding:
                adequate_pairs += 1

    return adequate_pairs / total_pairs if total_pairs > 0 else 1.0
```

---

## 5. Composite Quality Score

### 5.1 Weighted Overall Score

```python
def overall_quality_score(mesh, uvs, clusters, weights=None):
    """
    Compute weighted combination of all quality metrics.

    Returns score from 0 (worst) to 1 (best).
    """
    if weights is None:
        weights = {
            'interior_edges': 0.25,
            'stretch': 0.20,
            'angle_distortion': 0.15,
            'visibility': 0.15,
            'utilization': 0.10,
            'compactness': 0.10,
            'balance': 0.05,
        }

    scores = {}

    # Interior edge ratio (already 0-1)
    scores['interior_edges'] = interior_edge_ratio(mesh, clusters)

    # Stretch (convert to 0-1, 1.0 = stretch of 1.0, 0 = stretch > 2.0)
    stretch = mean_stretch(mesh, uvs, clusters)
    scores['stretch'] = max(0, 1.0 - (stretch - 1.0))

    # Angle distortion (convert to 0-1)
    max_angle = max_angle_distortion(mesh, uvs, clusters)
    scores['angle_distortion'] = max(0, 1.0 - max_angle / 60.0)

    # Visibility score (normalize and invert)
    vis = visibility_weighted_seam_score(mesh, clusters)
    max_vis = total_seam_length(mesh, clusters)  # Upper bound
    scores['visibility'] = 1.0 - (vis / max_vis) if max_vis > 0 else 1.0

    # UV utilization
    scores['utilization'] = uv_utilization(uvs, clusters)

    # Compactness
    scores['compactness'] = average_compactness(mesh, clusters)

    # Cluster balance
    scores['balance'] = cluster_balance_score(clusters)

    # Weighted sum
    total = sum(weights[k] * scores[k] for k in weights)

    return total, scores
```

### 5.2 Quality Grade

```python
def quality_grade(score):
    """Convert numeric score to letter grade."""
    if score >= 0.95:
        return 'A+'
    elif score >= 0.90:
        return 'A'
    elif score >= 0.85:
        return 'B+'
    elif score >= 0.80:
        return 'B'
    elif score >= 0.75:
        return 'C+'
    elif score >= 0.70:
        return 'C'
    elif score >= 0.60:
        return 'D'
    else:
        return 'F'
```

---

## 6. Quality Report Template

```python
def generate_quality_report(mesh, uvs, clusters):
    """
    Generate comprehensive quality report.
    """
    report = []

    report.append("=" * 60)
    report.append("POLYGON UNWRAPPING QUALITY REPORT")
    report.append("=" * 60)

    # Basic stats
    report.append(f"\nMesh: {len(mesh.faces)} faces, {len(mesh.edges)} edges")
    report.append(f"Clusters: {len(clusters)}")

    # Edge metrics
    report.append("\n--- EDGE CONNECTIVITY ---")
    ie_ratio = interior_edge_ratio(mesh, clusters)
    report.append(f"Interior edge ratio: {ie_ratio:.1%}")
    report.append(f"Total seam length: {total_seam_length(mesh, clusters):.2f}")

    # Distortion metrics
    report.append("\n--- DISTORTION ---")
    report.append(f"Max stretch (L²): {max_stretch(mesh, uvs, clusters):.3f}")
    report.append(f"Mean stretch: {mean_stretch(mesh, uvs, clusters):.3f}")
    report.append(f"Max angle distortion: {max_angle_distortion(mesh, uvs, clusters):.1f}°")

    # Cluster stats
    report.append("\n--- CLUSTER QUALITY ---")
    stats = cluster_size_stats(clusters)
    report.append(f"Cluster sizes: {stats['min']}-{stats['max']} (mean: {stats['mean']:.1f})")
    report.append(f"Balance score: {cluster_balance_score(clusters):.2f}")
    report.append(f"Avg compactness: {average_compactness(mesh, clusters):.2f}")

    # UV efficiency
    report.append("\n--- UV EFFICIENCY ---")
    report.append(f"UV utilization: {uv_utilization(uvs, clusters):.1%}")
    report.append(f"Packing efficiency: {packing_efficiency(uvs, clusters):.1%}")

    # Overall score
    report.append("\n--- OVERALL ---")
    total, scores = overall_quality_score(mesh, uvs, clusters)
    grade = quality_grade(total)
    report.append(f"Quality score: {total:.2f} (Grade: {grade})")

    report.append("\nComponent scores:")
    for key, value in scores.items():
        report.append(f"  {key}: {value:.2f}")

    report.append("=" * 60)

    return "\n".join(report)
```

### Example Output

```
============================================================
POLYGON UNWRAPPING QUALITY REPORT
============================================================

Mesh: 150 faces, 225 edges
Clusters: 5

--- EDGE CONNECTIVITY ---
Interior edge ratio: 89.3%
Total seam length: 4.52

--- DISTORTION ---
Max stretch (L²): 1.18
Mean stretch: 1.05
Max angle distortion: 12.3°

--- CLUSTER QUALITY ---
Cluster sizes: 14-54 (mean: 30.0)
Balance score: 0.72
Avg compactness: 0.81

--- UV EFFICIENCY ---
UV utilization: 76.2%
Packing efficiency: 82.5%

--- OVERALL ---
Quality score: 0.84 (Grade: B)

Component scores:
  interior_edges: 0.89
  stretch: 0.95
  angle_distortion: 0.80
  visibility: 0.78
  utilization: 0.76
  compactness: 0.81
  balance: 0.72
============================================================
```

---

## 7. Benchmarking & Comparison

```python
def compare_unwrapping_methods(mesh, methods):
    """
    Compare different unwrapping methods on same mesh.
    """
    results = []

    for method_name, method_func in methods.items():
        clusters = method_func(mesh)
        uvs = flatten_clusters(mesh, clusters)

        score, details = overall_quality_score(mesh, uvs, clusters)

        results.append({
            'method': method_name,
            'clusters': len(clusters),
            'interior_edges': details['interior_edges'],
            'stretch': details['stretch'],
            'score': score,
            'grade': quality_grade(score),
        })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results
```
