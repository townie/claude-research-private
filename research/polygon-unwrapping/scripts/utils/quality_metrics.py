"""
Quality metrics for evaluating polygon unwrapping results.

Provides metrics for:
- Edge connectivity (interior edge ratio, seam length)
- Distortion (stretch, angle preservation)
- Cluster quality (size balance, compactness)
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass

from .mesh import Mesh
from .dual_graph import DualGraph


@dataclass
class QualityReport:
    """Comprehensive quality report for unwrapping results."""
    # Edge metrics
    interior_edge_ratio: float
    total_seam_length: float
    normalized_seam_length: float

    # Cluster metrics
    num_clusters: int
    cluster_sizes: List[int]
    cluster_balance: float
    min_cluster_size: int
    max_cluster_size: int

    # Distortion metrics
    max_distortion: float
    mean_distortion: float

    # Overall
    overall_score: float
    grade: str


def compute_interior_edge_ratio(mesh: Mesh, clusters: List[Set[int]]) -> float:
    """
    Compute ratio of interior edges to total interior edges.

    Interior edge = edge connecting two faces in the same cluster.
    """
    dual = DualGraph(mesh)

    interior_count = 0
    total_count = len(dual.edges)

    # Build face-to-cluster mapping
    face_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for face_id in cluster:
            face_to_cluster[face_id] = i

    for edge in dual.edges.values():
        c1 = face_to_cluster.get(edge.face1_id, -1)
        c2 = face_to_cluster.get(edge.face2_id, -2)
        if c1 == c2:
            interior_count += 1

    return interior_count / total_count if total_count > 0 else 0


def compute_seam_length(mesh: Mesh, clusters: List[Set[int]]) -> Tuple[float, float]:
    """
    Compute total and normalized seam length.

    Returns (total_seam_length, normalized_seam_length).
    """
    dual = DualGraph(mesh)

    # Build face-to-cluster mapping
    face_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for face_id in cluster:
            face_to_cluster[face_id] = i

    seam_length = 0.0
    total_length = 0.0

    for edge in dual.edges.values():
        edge_length = mesh.get_edge_length(edge.mesh_edge)
        total_length += edge_length

        c1 = face_to_cluster.get(edge.face1_id, -1)
        c2 = face_to_cluster.get(edge.face2_id, -2)

        if c1 != c2:
            seam_length += edge_length

    normalized = seam_length / total_length if total_length > 0 else 0

    return seam_length, normalized


def compute_cluster_balance(clusters: List[Set[int]]) -> float:
    """
    Compute how balanced cluster sizes are.

    1.0 = perfectly balanced (all same size)
    0.0 = highly imbalanced
    """
    if not clusters:
        return 0.0

    sizes = [len(c) for c in clusters]
    mean_size = np.mean(sizes)

    if mean_size == 0:
        return 0.0

    # Coefficient of variation
    cv = np.std(sizes) / mean_size

    # Convert to 0-1 score (lower CV = higher score)
    return max(0, 1.0 - cv)


def estimate_cluster_distortion(mesh: Mesh, cluster: Set[int]) -> float:
    """
    Estimate distortion for flattening a cluster.

    Based on normal variance (proxy for Gaussian curvature).
    """
    if len(cluster) <= 1:
        return 0.0

    normals = [mesh.get_face_normal(f) for f in cluster]
    avg_normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(avg_normal)

    if norm > 1e-10:
        avg_normal = avg_normal / norm

    # Compute variance from average normal
    deviations = [1 - np.dot(n, avg_normal) for n in normals]

    return min(1.0, np.mean(deviations) * 2)


def compute_distortion_stats(mesh: Mesh, clusters: List[Set[int]]) -> Tuple[float, float]:
    """
    Compute max and mean distortion across clusters.

    Returns (max_distortion, mean_distortion).
    """
    if not clusters:
        return 0.0, 0.0

    distortions = [estimate_cluster_distortion(mesh, c) for c in clusters]

    return max(distortions), np.mean(distortions)


def compute_overall_score(
    interior_edge_ratio: float,
    normalized_seam_length: float,
    cluster_balance: float,
    mean_distortion: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute weighted overall quality score.

    Higher is better (0-1 scale).
    """
    if weights is None:
        weights = {
            'interior_edges': 0.35,
            'seam_length': 0.25,
            'balance': 0.15,
            'distortion': 0.25,
        }

    scores = {
        'interior_edges': interior_edge_ratio,
        'seam_length': 1.0 - normalized_seam_length,
        'balance': cluster_balance,
        'distortion': 1.0 - mean_distortion,
    }

    total = sum(weights[k] * scores[k] for k in weights)
    return total


def score_to_grade(score: float) -> str:
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


def generate_quality_report(mesh: Mesh, clusters: List[Set[int]]) -> QualityReport:
    """
    Generate comprehensive quality report for clustering results.
    """
    # Edge metrics
    interior_ratio = compute_interior_edge_ratio(mesh, clusters)
    seam_length, normalized_seam = compute_seam_length(mesh, clusters)

    # Cluster metrics
    sizes = [len(c) for c in clusters]
    balance = compute_cluster_balance(clusters)

    # Distortion
    max_dist, mean_dist = compute_distortion_stats(mesh, clusters)

    # Overall
    overall = compute_overall_score(
        interior_ratio, normalized_seam, balance, mean_dist
    )
    grade = score_to_grade(overall)

    return QualityReport(
        interior_edge_ratio=interior_ratio,
        total_seam_length=seam_length,
        normalized_seam_length=normalized_seam,
        num_clusters=len(clusters),
        cluster_sizes=sizes,
        cluster_balance=balance,
        min_cluster_size=min(sizes) if sizes else 0,
        max_cluster_size=max(sizes) if sizes else 0,
        max_distortion=max_dist,
        mean_distortion=mean_dist,
        overall_score=overall,
        grade=grade
    )


def format_quality_report(report: QualityReport) -> str:
    """Format quality report as readable string."""
    lines = [
        "=" * 50,
        "POLYGON UNWRAPPING QUALITY REPORT",
        "=" * 50,
        "",
        "--- EDGE CONNECTIVITY ---",
        f"Interior edge ratio:    {report.interior_edge_ratio:.1%}",
        f"Total seam length:      {report.total_seam_length:.2f}",
        f"Normalized seam length: {report.normalized_seam_length:.1%}",
        "",
        "--- CLUSTERS ---",
        f"Number of clusters:     {report.num_clusters}",
        f"Cluster sizes:          {report.min_cluster_size} - {report.max_cluster_size}",
        f"Balance score:          {report.cluster_balance:.2f}",
        "",
        "--- DISTORTION ---",
        f"Max distortion:         {report.max_distortion:.3f}",
        f"Mean distortion:        {report.mean_distortion:.3f}",
        "",
        "--- OVERALL ---",
        f"Quality score:          {report.overall_score:.2f}",
        f"Grade:                  {report.grade}",
        "=" * 50,
    ]
    return "\n".join(lines)


def compare_clustering_methods(
    mesh: Mesh,
    methods: Dict[str, List[Set[int]]]
) -> str:
    """
    Compare multiple clustering methods on the same mesh.

    Parameters:
        mesh: The mesh
        methods: Dict mapping method name to clusters

    Returns:
        Formatted comparison table
    """
    reports = {}
    for name, clusters in methods.items():
        reports[name] = generate_quality_report(mesh, clusters)

    # Build comparison table
    lines = [
        "=" * 70,
        "CLUSTERING METHOD COMPARISON",
        "=" * 70,
        "",
        f"{'Method':<20} {'Clusters':<10} {'Int.Edges':<12} {'Distortion':<12} {'Score':<8} {'Grade':<6}",
        "-" * 70,
    ]

    for name, report in sorted(reports.items(), key=lambda x: -x[1].overall_score):
        lines.append(
            f"{name:<20} {report.num_clusters:<10} "
            f"{report.interior_edge_ratio:.1%}{'':>4} "
            f"{report.mean_distortion:.3f}{'':>6} "
            f"{report.overall_score:.2f}{'':>4} {report.grade:<6}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)
