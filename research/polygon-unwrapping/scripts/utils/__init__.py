"""Utility modules for polygon unwrapping."""

from .mesh import Mesh, Face, Vertex, Edge, create_cube, create_pyramid, create_cylinder, create_simple_dog
from .dual_graph import DualGraph, DualEdge, UnionFind
from .edge_weights import (
    dihedral_weight, edge_length_weight, curvature_continuity_weight,
    feature_edge_weight, visibility_weight, composite_weight, pet_weight,
    apply_dihedral_weights, apply_composite_weights, apply_pet_weights,
    WeightConfig
)
from .quality_metrics import (
    compute_interior_edge_ratio, compute_seam_length, compute_cluster_balance,
    estimate_cluster_distortion, generate_quality_report, format_quality_report,
    compare_clustering_methods, QualityReport
)
from .optimization import (
    optimize_boundaries, merge_small_clusters, split_high_distortion_clusters,
    simulated_annealing, full_optimization_pipeline
)

__all__ = [
    # Mesh
    'Mesh', 'Face', 'Vertex', 'Edge',
    'create_cube', 'create_pyramid', 'create_cylinder', 'create_simple_dog',

    # Dual Graph
    'DualGraph', 'DualEdge', 'UnionFind',

    # Edge Weights
    'dihedral_weight', 'edge_length_weight', 'curvature_continuity_weight',
    'feature_edge_weight', 'visibility_weight', 'composite_weight', 'pet_weight',
    'apply_dihedral_weights', 'apply_composite_weights', 'apply_pet_weights',
    'WeightConfig',

    # Quality Metrics
    'compute_interior_edge_ratio', 'compute_seam_length', 'compute_cluster_balance',
    'estimate_cluster_distortion', 'generate_quality_report', 'format_quality_report',
    'compare_clustering_methods', 'QualityReport',

    # Optimization
    'optimize_boundaries', 'merge_small_clusters', 'split_high_distortion_clusters',
    'simulated_annealing', 'full_optimization_pipeline',
]
