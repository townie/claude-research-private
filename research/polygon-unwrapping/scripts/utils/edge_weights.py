"""
Edge weight functions for polygon unwrapping.

Higher weight = prefer to keep edge connected (interior)
Lower weight = prefer to cut edge (seam)
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional, List
from dataclasses import dataclass

from .mesh import Mesh
from .dual_graph import DualGraph


@dataclass
class WeightConfig:
    """Configuration for composite weight functions."""
    dihedral_weight: float = 0.4
    length_weight: float = 0.2
    curvature_weight: float = 0.2
    feature_weight: float = 0.2

    # Thresholds
    feature_angle_threshold: float = 30.0  # degrees
    dihedral_exponent: float = 1.0  # 1=linear, 2=quadratic falloff


# Individual weight functions

def dihedral_weight(mesh: Mesh, face1_id: int, face2_id: int,
                    exponent: float = 1.0) -> float:
    """
    Weight based on dihedral angle between faces.

    180° (coplanar) -> 1.0
    90° (perpendicular) -> 0.5
    0° (folded back) -> 0.0
    """
    dihedral = mesh.compute_dihedral_angle(face1_id, face2_id)
    normalized = dihedral / 180.0

    if exponent != 1.0:
        normalized = normalized ** exponent

    return normalized


def edge_length_weight(mesh: Mesh, edge: Tuple[int, int],
                       normalize: bool = True) -> float:
    """
    Weight based on edge length.

    Longer edges = higher weight (more visible if cut).
    """
    length = mesh.get_edge_length(edge)

    if normalize:
        # Normalize by average edge length
        all_lengths = [mesh.get_edge_length(e) for e in mesh.get_interior_edges()]
        avg_length = np.mean(all_lengths) if all_lengths else 1.0
        length = length / avg_length

    # Clamp to reasonable range
    return min(length, 2.0)


def curvature_continuity_weight(mesh: Mesh, face1_id: int, face2_id: int) -> float:
    """
    Weight based on curvature continuity across the edge.

    Similar curvature on both sides -> high weight
    Curvature discontinuity -> low weight
    """
    # Use normal difference as proxy for curvature change
    n1 = mesh.get_face_normal(face1_id)
    n2 = mesh.get_face_normal(face2_id)

    # Dot product: 1 = same direction, -1 = opposite
    similarity = np.dot(n1, n2)

    # Map from [-1, 1] to [0, 1]
    weight = (similarity + 1.0) / 2.0

    return weight


def feature_edge_weight(mesh: Mesh, face1_id: int, face2_id: int,
                        threshold: float = 30.0) -> float:
    """
    Detect feature edges (sharp creases).

    Returns low weight for sharp edges (should cut here).
    """
    dihedral = mesh.compute_dihedral_angle(face1_id, face2_id)

    # Sharp edge = low dihedral angle = feature
    if dihedral < 180.0 - threshold:
        return 0.1  # Prefer to cut
    return 1.0


def area_ratio_weight(mesh: Mesh, face1_id: int, face2_id: int) -> float:
    """
    Prefer edges between similarly-sized faces.
    """
    a1 = mesh.get_face_area(face1_id)
    a2 = mesh.get_face_area(face2_id)

    if max(a1, a2) < 1e-10:
        return 1.0

    ratio = min(a1, a2) / max(a1, a2)
    return ratio


def visibility_weight(mesh: Mesh, edge: Tuple[int, int],
                      camera_positions: Optional[List[np.ndarray]] = None) -> float:
    """
    Weight based on edge visibility.

    Hidden edges (on underside, inner surfaces) get lower weight.
    """
    if camera_positions is None:
        # Default: views from 6 cardinal directions
        camera_positions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
        ]

    edge_center = mesh.get_edge_center(edge)
    mesh_center = mesh.center()

    # Outward direction from mesh center
    outward = edge_center - mesh_center
    norm = np.linalg.norm(outward)
    if norm > 1e-10:
        outward = outward / norm
    else:
        outward = np.array([0, 1, 0])

    # Check visibility from each camera
    max_visibility = 0.0
    for cam_pos in camera_positions:
        cam_dir = cam_pos / np.linalg.norm(cam_pos)
        visibility = max(0, np.dot(outward, cam_dir))
        max_visibility = max(max_visibility, visibility)

    # Invert: hidden edges should have LOW weight (prefer to cut there)
    return 1.0 - 0.5 * max_visibility


def underside_weight(mesh: Mesh, edge: Tuple[int, int],
                     up_axis: int = 1) -> float:
    """
    Give lower weight to edges on the underside of the mesh.

    Good for characters/pets where belly seams are less visible.
    """
    edge_center = mesh.get_edge_center(edge)
    mesh_center = mesh.center()

    # Check if edge is below center in up axis
    if edge_center[up_axis] < mesh_center[up_axis]:
        return 0.5  # Underside - prefer to cut here
    return 1.0


# Composite weight functions

def composite_weight(mesh: Mesh, face1_id: int, face2_id: int,
                     edge: Tuple[int, int],
                     config: Optional[WeightConfig] = None) -> float:
    """
    Weighted combination of multiple weight functions.
    """
    if config is None:
        config = WeightConfig()

    w_dihedral = dihedral_weight(mesh, face1_id, face2_id, config.dihedral_exponent)
    w_length = edge_length_weight(mesh, edge)
    w_curvature = curvature_continuity_weight(mesh, face1_id, face2_id)
    w_feature = feature_edge_weight(mesh, face1_id, face2_id, config.feature_angle_threshold)

    total = (
        config.dihedral_weight * w_dihedral +
        config.length_weight * w_length +
        config.curvature_weight * w_curvature +
        config.feature_weight * w_feature
    )

    return total


def multiplicative_weight(mesh: Mesh, face1_id: int, face2_id: int,
                          edge: Tuple[int, int]) -> float:
    """
    Multiply weights - any low weight vetoes the edge.
    """
    w1 = dihedral_weight(mesh, face1_id, face2_id)
    w2 = feature_edge_weight(mesh, face1_id, face2_id)
    w3 = curvature_continuity_weight(mesh, face1_id, face2_id)

    return w1 * w2 * w3


def pet_weight(mesh: Mesh, face1_id: int, face2_id: int,
               edge: Tuple[int, int]) -> float:
    """
    Weight function optimized for pet meshes (100-200 polys).

    - Prioritizes dihedral angle
    - Prefers cuts on underside
    - Respects feature edges
    """
    w_dihedral = dihedral_weight(mesh, face1_id, face2_id)
    w_feature = feature_edge_weight(mesh, face1_id, face2_id, threshold=45.0)
    w_underside = underside_weight(mesh, edge)

    # Multiplicative to ensure feature edges are cut
    return w_dihedral * w_feature * w_underside


# Weight application to dual graph

def apply_weights(dual_graph: DualGraph,
                  weight_func: Callable,
                  **kwargs) -> None:
    """
    Apply a weight function to all edges in a dual graph.
    """
    mesh = dual_graph.mesh

    for edge in dual_graph.edges.values():
        weight = weight_func(
            mesh,
            edge.face1_id,
            edge.face2_id,
            edge.mesh_edge,
            **kwargs
        )
        edge.weight = weight


def apply_dihedral_weights(dual_graph: DualGraph, exponent: float = 1.0) -> None:
    """Convenience function to apply dihedral weights."""
    for edge in dual_graph.edges.values():
        edge.weight = dihedral_weight(
            dual_graph.mesh,
            edge.face1_id,
            edge.face2_id,
            exponent
        )


def apply_composite_weights(dual_graph: DualGraph,
                            config: Optional[WeightConfig] = None) -> None:
    """Convenience function to apply composite weights."""
    for edge in dual_graph.edges.values():
        edge.weight = composite_weight(
            dual_graph.mesh,
            edge.face1_id,
            edge.face2_id,
            edge.mesh_edge,
            config
        )


def apply_pet_weights(dual_graph: DualGraph) -> None:
    """Convenience function to apply pet-optimized weights."""
    for edge in dual_graph.edges.values():
        edge.weight = pet_weight(
            dual_graph.mesh,
            edge.face1_id,
            edge.face2_id,
            edge.mesh_edge
        )


# Weight statistics

def get_weight_statistics(dual_graph: DualGraph) -> Dict:
    """Get statistics about edge weights."""
    weights = [e.weight for e in dual_graph.edges.values()]

    if not weights:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

    return {
        'min': min(weights),
        'max': max(weights),
        'mean': np.mean(weights),
        'std': np.std(weights),
        'count': len(weights),
    }
