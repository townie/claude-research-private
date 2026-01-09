"""
UV Flattening algorithms for polygon unwrapping.

Provides methods to flatten 3D mesh clusters into 2D UV coordinates:
- Planar projection (simple, fast)
- Conformal mapping via LSCM (angle-preserving)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from scripts.utils.mesh import Mesh


class FlattenMethod(Enum):
    PLANAR = "planar"
    CONFORMAL = "conformal"


@dataclass
class ClusterUV:
    """UV coordinates for a flattened cluster."""
    cluster_id: int
    face_ids: List[int]
    # Maps original vertex_id -> (u, v) in this cluster's local UV space
    uv_coords: Dict[int, Tuple[float, float]]
    # UV faces using local vertex indices for this cluster
    uv_faces: List[List[int]]
    # Mapping from local UV vertex index to original vertex_id
    local_to_global: Dict[int, int]
    # Bounding box in UV space
    bbox_min: Tuple[float, float] = (0.0, 0.0)
    bbox_max: Tuple[float, float] = (1.0, 1.0)
    # Method used for flattening
    method: FlattenMethod = FlattenMethod.PLANAR

    @property
    def width(self) -> float:
        return self.bbox_max[0] - self.bbox_min[0]

    @property
    def height(self) -> float:
        return self.bbox_max[1] - self.bbox_min[1]

    def get_uv_vertices_list(self) -> List[Tuple[float, float]]:
        """Get UV vertices as ordered list matching local indices."""
        result = [(0.0, 0.0)] * len(self.local_to_global)
        for local_idx, global_id in self.local_to_global.items():
            result[local_idx] = self.uv_coords[global_id]
        return result


def _extract_cluster_submesh(mesh: Mesh, cluster: Set[int]) -> Tuple[
    np.ndarray,  # vertices (N, 3)
    List[List[int]],  # faces with local vertex indices
    Dict[int, int],  # global_to_local vertex mapping
    Dict[int, int],  # local_to_global vertex mapping
]:
    """Extract submesh for a cluster with local vertex indexing."""
    # Collect all vertices used by cluster faces
    global_vertices = set()
    for face_id in cluster:
        face = mesh.faces[face_id]
        global_vertices.update(face.vertex_ids)

    # Create mapping
    global_to_local = {gid: lid for lid, gid in enumerate(sorted(global_vertices))}
    local_to_global = {lid: gid for gid, lid in global_to_local.items()}

    # Build vertex array
    vertices = np.array([
        mesh.vertices[gid].position
        for gid in sorted(global_vertices)
    ])

    # Build faces with local indices
    local_faces = []
    for face_id in cluster:
        face = mesh.faces[face_id]
        local_face = [global_to_local[vid] for vid in face.vertex_ids]
        local_faces.append(local_face)

    return vertices, local_faces, global_to_local, local_to_global


def _find_boundary_vertices(faces: List[List[int]], n_vertices: int) -> List[int]:
    """Find boundary vertices of a mesh in order."""
    # Count edge occurrences
    edge_count = {}
    edge_to_next = {}

    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            # Store directed edge for boundary traversal
            edge_to_next[(v1, v2)] = v2

    # Boundary edges appear exactly once
    boundary_edges = {e for e, c in edge_count.items() if c == 1}

    if not boundary_edges:
        # No boundary (closed mesh) - return empty
        return []

    # Build adjacency for boundary
    boundary_adj = {}
    for e in boundary_edges:
        v1, v2 = e
        boundary_adj.setdefault(v1, []).append(v2)
        boundary_adj.setdefault(v2, []).append(v1)

    # Traverse boundary in order
    start = min(boundary_adj.keys())
    boundary = [start]
    visited = {start}
    current = start

    while True:
        neighbors = boundary_adj.get(current, [])
        next_v = None
        for n in neighbors:
            if n not in visited:
                next_v = n
                break

        if next_v is None:
            break

        boundary.append(next_v)
        visited.add(next_v)
        current = next_v

    return boundary


def planar_projection(mesh: Mesh, cluster: Set[int]) -> ClusterUV:
    """
    Flatten cluster using planar projection onto best-fit plane.

    Uses PCA to find the plane that best fits the cluster vertices,
    then projects onto the first two principal components.

    Args:
        mesh: The source mesh
        cluster: Set of face IDs in this cluster

    Returns:
        ClusterUV with 2D coordinates
    """
    vertices, local_faces, global_to_local, local_to_global = \
        _extract_cluster_submesh(mesh, cluster)

    if len(vertices) < 3:
        # Degenerate case
        uv_coords = {local_to_global[i]: (0.0, 0.0) for i in range(len(vertices))}
        return ClusterUV(
            cluster_id=min(cluster),
            face_ids=list(cluster),
            uv_coords=uv_coords,
            uv_faces=local_faces,
            local_to_global=local_to_global,
            method=FlattenMethod.PLANAR,
        )

    # Center the vertices
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # PCA to find best-fit plane
    # Use SVD: centered = U @ S @ Vt
    # The first two rows of Vt are the principal axes
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Project onto first two principal components
        uv_2d = centered @ Vt[:2].T
    except np.linalg.LinAlgError:
        # Fallback to XY projection if SVD fails
        uv_2d = centered[:, :2]

    # Normalize to [0, 1] range with some padding
    uv_min = uv_2d.min(axis=0)
    uv_max = uv_2d.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.where(uv_range < 1e-10, 1.0, uv_range)  # Prevent division by zero

    uv_normalized = (uv_2d - uv_min) / uv_range

    # Build UV coordinate dictionary
    uv_coords = {}
    for local_idx in range(len(vertices)):
        global_id = local_to_global[local_idx]
        uv_coords[global_id] = (float(uv_normalized[local_idx, 0]),
                                 float(uv_normalized[local_idx, 1]))

    # Compute actual bounding box
    bbox_min = (float(uv_normalized[:, 0].min()), float(uv_normalized[:, 1].min()))
    bbox_max = (float(uv_normalized[:, 0].max()), float(uv_normalized[:, 1].max()))

    return ClusterUV(
        cluster_id=min(cluster),
        face_ids=list(cluster),
        uv_coords=uv_coords,
        uv_faces=local_faces,
        local_to_global=local_to_global,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        method=FlattenMethod.PLANAR,
    )


def conformal_flatten(mesh: Mesh, cluster: Set[int]) -> ClusterUV:
    """
    Flatten cluster using LSCM (Least Squares Conformal Mapping).

    This method preserves angles locally, making it ideal for
    texturing and paper craft where angle accuracy matters.

    Args:
        mesh: The source mesh
        cluster: Set of face IDs in this cluster

    Returns:
        ClusterUV with 2D coordinates
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
    except ImportError:
        # Fall back to planar if scipy not available
        print("Warning: scipy not available, falling back to planar projection")
        return planar_projection(mesh, cluster)

    vertices, local_faces, global_to_local, local_to_global = \
        _extract_cluster_submesh(mesh, cluster)

    n_verts = len(vertices)
    n_faces = len(local_faces)

    if n_verts < 3 or n_faces < 1:
        return planar_projection(mesh, cluster)

    # Find boundary vertices
    boundary = _find_boundary_vertices(local_faces, n_verts)

    if len(boundary) < 2:
        # No proper boundary, fall back to planar
        return planar_projection(mesh, cluster)

    # Pin two boundary vertices to fix the solution
    # Choose vertices far apart on boundary
    pin1 = boundary[0]
    pin2 = boundary[len(boundary) // 2]

    # Fixed UV positions for pinned vertices
    pin1_uv = np.array([0.0, 0.0])
    pin2_uv = np.array([1.0, 0.0])

    # Build LSCM system
    # For each triangle, we want to minimize the conformal energy
    # This leads to a sparse linear system

    # We solve for UV coordinates of all vertices except pinned ones
    free_verts = [i for i in range(n_verts) if i not in (pin1, pin2)]
    free_to_idx = {v: i for i, v in enumerate(free_verts)}
    n_free = len(free_verts)

    if n_free == 0:
        # Only 2 vertices, trivial case
        uv_coords = {}
        for local_idx in range(n_verts):
            global_id = local_to_global[local_idx]
            if local_idx == pin1:
                uv_coords[global_id] = tuple(pin1_uv)
            elif local_idx == pin2:
                uv_coords[global_id] = tuple(pin2_uv)
        return ClusterUV(
            cluster_id=min(cluster),
            face_ids=list(cluster),
            uv_coords=uv_coords,
            uv_faces=local_faces,
            local_to_global=local_to_global,
            method=FlattenMethod.CONFORMAL,
        )

    # Build sparse matrix for LSCM
    # Each triangle contributes 2 rows (real and imaginary parts of conformality)
    rows = []
    cols = []
    data = []
    rhs = np.zeros(2 * n_faces)

    for face_idx, face in enumerate(local_faces):
        if len(face) != 3:
            continue  # Skip non-triangles

        v0, v1, v2 = face
        p0 = vertices[v0]
        p1 = vertices[v1]
        p2 = vertices[v2]

        # Edge vectors
        e1 = p1 - p0
        e2 = p2 - p0

        # Triangle area (for weighting)
        area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        if area < 1e-10:
            continue

        # Local 2D coordinates in triangle plane
        # x-axis along e1
        e1_len = np.linalg.norm(e1)
        if e1_len < 1e-10:
            continue

        x_axis = e1 / e1_len
        normal = np.cross(e1, e2)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-10:
            continue
        normal = normal / normal_len
        y_axis = np.cross(normal, x_axis)

        # Project vertices to local 2D
        local_2d = np.array([
            [0, 0],
            [e1_len, 0],
            [np.dot(e2, x_axis), np.dot(e2, y_axis)]
        ])

        # LSCM coefficients
        # For conformal mapping: (u1-u0) + i(v1-v0) = lambda * ((x1-x0) + i(y1-y0))
        # This gives us constraints on u, v coordinates

        s = np.sqrt(area)

        # Coefficients for gradient in local coordinates
        # d_x = (y2-y0, y0-y1, y1-y2) / (2*area)
        # d_y = (x0-x2, x1-x0, x2-x1) / (2*area)

        x0, y0 = local_2d[0]
        x1, y1 = local_2d[1]
        x2, y2 = local_2d[2]

        denom = 2 * area
        w = [
            complex(x2 - x1, y2 - y1) / denom,
            complex(x0 - x2, y0 - y2) / denom,
            complex(x1 - x0, y1 - y0) / denom,
        ]

        row_re = 2 * face_idx
        row_im = 2 * face_idx + 1

        for local_v, coef in zip(face, w):
            if local_v == pin1:
                # Move to RHS
                rhs[row_re] -= coef.real * pin1_uv[0] - coef.imag * pin1_uv[1]
                rhs[row_im] -= coef.real * pin1_uv[1] + coef.imag * pin1_uv[0]
            elif local_v == pin2:
                rhs[row_re] -= coef.real * pin2_uv[0] - coef.imag * pin2_uv[1]
                rhs[row_im] -= coef.real * pin2_uv[1] + coef.imag * pin2_uv[0]
            else:
                col_u = free_to_idx[local_v]
                col_v = n_free + free_to_idx[local_v]

                # Real part: coef.real * u - coef.imag * v
                rows.extend([row_re, row_re])
                cols.extend([col_u, col_v])
                data.extend([coef.real, -coef.imag])

                # Imaginary part: coef.real * v + coef.imag * u
                rows.extend([row_im, row_im])
                cols.extend([col_v, col_u])
                data.extend([coef.real, coef.imag])

    if len(rows) == 0:
        return planar_projection(mesh, cluster)

    # Build sparse matrix and solve
    A = sparse.csr_matrix((data, (rows, cols)), shape=(2 * n_faces, 2 * n_free))

    try:
        result = lsqr(A, rhs, atol=1e-8, btol=1e-8)
        uv_solution = result[0]
    except Exception:
        return planar_projection(mesh, cluster)

    # Extract UV coordinates
    uv_all = np.zeros((n_verts, 2))
    uv_all[pin1] = pin1_uv
    uv_all[pin2] = pin2_uv

    for free_v, idx in free_to_idx.items():
        uv_all[free_v, 0] = uv_solution[idx]
        uv_all[free_v, 1] = uv_solution[n_free + idx]

    # Normalize to [0, 1] range
    uv_min = uv_all.min(axis=0)
    uv_max = uv_all.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.where(uv_range < 1e-10, 1.0, uv_range)

    uv_normalized = (uv_all - uv_min) / uv_range

    # Build result
    uv_coords = {}
    for local_idx in range(n_verts):
        global_id = local_to_global[local_idx]
        uv_coords[global_id] = (float(uv_normalized[local_idx, 0]),
                                 float(uv_normalized[local_idx, 1]))

    bbox_min = (float(uv_normalized[:, 0].min()), float(uv_normalized[:, 1].min()))
    bbox_max = (float(uv_normalized[:, 0].max()), float(uv_normalized[:, 1].max()))

    return ClusterUV(
        cluster_id=min(cluster),
        face_ids=list(cluster),
        uv_coords=uv_coords,
        uv_faces=local_faces,
        local_to_global=local_to_global,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        method=FlattenMethod.CONFORMAL,
    )


def flatten_cluster(
    mesh: Mesh,
    cluster: Set[int],
    method: str = "conformal"
) -> ClusterUV:
    """
    Flatten a cluster of faces to 2D UV coordinates.

    Args:
        mesh: The source mesh
        cluster: Set of face IDs to flatten
        method: "conformal" (angle-preserving) or "planar" (simple projection)

    Returns:
        ClusterUV with 2D coordinates
    """
    if method == "conformal":
        return conformal_flatten(mesh, cluster)
    elif method == "planar":
        return planar_projection(mesh, cluster)
    else:
        raise ValueError(f"Unknown flattening method: {method}. Use 'conformal' or 'planar'.")


def flatten_all_clusters(
    mesh: Mesh,
    clusters: List[Set[int]],
    method: str = "conformal",
    preserve_scale: bool = True
) -> List[ClusterUV]:
    """
    Flatten all clusters in a mesh.

    Args:
        mesh: The source mesh
        clusters: List of clusters (each a set of face IDs)
        method: "conformal" or "planar"
        preserve_scale: If True, all clusters use the same global scale to
                       preserve relative sizes (for paper models)

    Returns:
        List of ClusterUV objects
    """
    results = []
    for i, cluster in enumerate(clusters):
        cluster_uv = flatten_cluster(mesh, cluster, method)
        cluster_uv.cluster_id = i
        results.append(cluster_uv)

    if preserve_scale and len(results) > 0:
        # Rescale all clusters to use actual 3D units
        # This ensures a 1:1 mapping between 3D size and UV size
        results = _rescale_clusters_to_3d_units(mesh, clusters, results)

    return results


def _rescale_clusters_to_3d_units(
    mesh: Mesh,
    clusters: List[Set[int]],
    clusters_uv: List[ClusterUV]
) -> List[ClusterUV]:
    """
    Rescale UV coordinates to match actual 3D edge lengths.

    This ensures all clusters use the same scale factor, preserving
    relative sizes between clusters for proper paper model assembly.
    """
    rescaled_results = []

    for cluster_idx, (cluster, cuv) in enumerate(zip(clusters, clusters_uv)):
        # Calculate scale factor by comparing UV edge lengths to 3D edge lengths
        total_uv_length = 0.0
        total_3d_length = 0.0
        edge_count = 0

        for face in cuv.uv_faces:
            for i in range(len(face)):
                local_idx1 = face[i]
                local_idx2 = face[(i + 1) % len(face)]

                # Get UV edge length
                global_v1 = cuv.local_to_global.get(local_idx1)
                global_v2 = cuv.local_to_global.get(local_idx2)

                if global_v1 is None or global_v2 is None:
                    continue
                if global_v1 not in cuv.uv_coords or global_v2 not in cuv.uv_coords:
                    continue

                uv1 = cuv.uv_coords[global_v1]
                uv2 = cuv.uv_coords[global_v2]
                uv_dx = uv2[0] - uv1[0]
                uv_dy = uv2[1] - uv1[1]
                uv_len = np.sqrt(uv_dx * uv_dx + uv_dy * uv_dy)

                # Get 3D edge length
                if global_v1 in mesh.vertices and global_v2 in mesh.vertices:
                    p1 = mesh.vertices[global_v1].position
                    p2 = mesh.vertices[global_v2].position
                    d = p2 - p1
                    len_3d = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

                    if uv_len > 1e-10 and len_3d > 1e-10:
                        total_uv_length += uv_len
                        total_3d_length += len_3d
                        edge_count += 1

        # Scale factor to convert UV to 3D units
        if edge_count > 0 and total_uv_length > 0:
            scale = total_3d_length / total_uv_length
        else:
            scale = 1.0

        # Apply scale to UV coordinates
        new_uv_coords = {}
        for global_id, (u, v) in cuv.uv_coords.items():
            new_uv_coords[global_id] = (u * scale, v * scale)

        # Update bounding box
        if new_uv_coords:
            all_u = [uv[0] for uv in new_uv_coords.values()]
            all_v = [uv[1] for uv in new_uv_coords.values()]
            new_bbox_min = (min(all_u), min(all_v))
            new_bbox_max = (max(all_u), max(all_v))
        else:
            new_bbox_min = (0.0, 0.0)
            new_bbox_max = (1.0, 1.0)

        # Create rescaled ClusterUV
        rescaled = ClusterUV(
            cluster_id=cuv.cluster_id,
            face_ids=cuv.face_ids,
            uv_coords=new_uv_coords,
            uv_faces=cuv.uv_faces,
            local_to_global=cuv.local_to_global,
            bbox_min=new_bbox_min,
            bbox_max=new_bbox_max,
            method=cuv.method,
        )
        rescaled_results.append(rescaled)

    return rescaled_results
