"""
FastAPI server for UV visualization dashboard.

Provides REST API for mesh loading, clustering, UV flattening, and export.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import project modules
from scripts.utils.mesh import Mesh, create_cube, create_pyramid, create_cylinder, create_simple_dog
from scripts.utils.dual_graph import DualGraph
from scripts.utils.edge_weights import apply_dihedral_weights, apply_pet_weights
from scripts.utils.uv_flatten import flatten_all_clusters, FlattenMethod
from scripts.utils.uv_packer import pack_uv_islands, export_print_svg, export_print_pdf
from scripts.algorithms.maximum_spanning_forest import maximum_spanning_forest
from scripts.algorithms.greedy_region_growing import greedy_region_growing
from scripts.algorithms.hierarchical_clustering import hierarchical_clustering


app = FastAPI(
    title="UV Visualization Dashboard",
    description="Interactive 3D/2D UV unwrapping visualization",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (simple for demo)
sessions: Dict[str, Dict[str, Any]] = {}
current_session_id = "default"


class AlgorithmType(str, Enum):
    MSF = "msf"
    GREEDY = "greedy"
    HIERARCHICAL = "hierarchical"


class FlattenMethodType(str, Enum):
    CONFORMAL = "conformal"
    PLANAR = "planar"


class ClusterRequest(BaseModel):
    algorithm: AlgorithmType = AlgorithmType.MSF
    max_distortion: float = 0.3


class FlattenRequest(BaseModel):
    method: FlattenMethodType = FlattenMethodType.CONFORMAL


def get_session(session_id: str = "default") -> Dict[str, Any]:
    """Get or create a session."""
    if session_id not in sessions:
        sessions[session_id] = {
            "mesh": None,
            "clusters": None,
            "clusters_uv": None,
            "mesh_name": None,
        }
    return sessions[session_id]


def mesh_to_json(mesh: Mesh) -> Dict:
    """Convert mesh to JSON-serializable format."""
    vertices = []
    for vid in sorted(mesh.vertices.keys()):
        v = mesh.vertices[vid]
        vertices.append(v.position.tolist())

    faces = []
    face_ids = []
    for fid in sorted(mesh.faces.keys()):
        f = mesh.faces[fid]
        faces.append(f.vertex_ids)
        face_ids.append(fid)

    # Compute normals
    normals = []
    for fid in sorted(mesh.faces.keys()):
        normal = mesh.get_face_normal(fid)
        normals.append(normal.tolist() if normal is not None else [0, 0, 1])

    return {
        "vertices": vertices,
        "faces": faces,
        "face_ids": face_ids,
        "normals": normals,
    }


def compute_cluster_metrics(mesh: Mesh, cluster: set, cluster_uv) -> dict:
    """
    Compute geometric metrics for a cluster.

    Returns dict with:
        - face_count: number of faces in cluster
        - surface_area_3d: sum of face areas in 3D mesh units
        - physical_width: width when printed (UV width × scale_factor)
        - physical_height: height when printed (UV height × scale_factor)
    """
    # Sum face areas from the 3D mesh
    surface_area = 0.0
    for face_id in cluster:
        if face_id in mesh.faces:
            surface_area += mesh.get_face_area(face_id)

    # Get physical dimensions using scale_factor
    if cluster_uv is not None:
        scale_factor = compute_cluster_scale(mesh, cluster, cluster_uv)
        physical_width = cluster_uv.width * scale_factor
        physical_height = cluster_uv.height * scale_factor
    else:
        scale_factor = 1.0
        physical_width = 0.0
        physical_height = 0.0

    return {
        "face_count": len(cluster),
        "surface_area_3d": float(surface_area),
        "physical_width": float(physical_width),
        "physical_height": float(physical_height),
        "scale_factor": float(scale_factor),
    }


def compute_cluster_scale(mesh: Mesh, cluster: set, cluster_uv) -> float:
    """
    Compute the scale factor for a cluster by comparing UV edge lengths to 3D edge lengths.
    Returns the ratio: 3D_length / UV_length
    This scale factor should be applied to UV coords to get real-world dimensions.
    """
    import numpy as np

    if cluster_uv is None:
        return 1.0

    total_uv_length = 0.0
    total_3d_length = 0.0
    edge_count = 0

    # Iterate through faces in the cluster
    for face_idx, uv_face in enumerate(cluster_uv.uv_faces):
        for i in range(len(uv_face)):
            local_idx1 = uv_face[i]
            local_idx2 = uv_face[(i + 1) % len(uv_face)]

            # Get UV edge length
            uv1 = cluster_uv.uv_coords.get(cluster_uv.local_to_global.get(local_idx1, local_idx1), (0, 0))
            uv2 = cluster_uv.uv_coords.get(cluster_uv.local_to_global.get(local_idx2, local_idx2), (0, 0))

            # Use the uv_vertices list which is properly indexed
            if local_idx1 < len(cluster_uv.get_uv_vertices_list()) and local_idx2 < len(cluster_uv.get_uv_vertices_list()):
                uv1 = cluster_uv.get_uv_vertices_list()[local_idx1]
                uv2 = cluster_uv.get_uv_vertices_list()[local_idx2]

            uv_dx = uv2[0] - uv1[0]
            uv_dy = uv2[1] - uv1[1]
            uv_len = np.sqrt(uv_dx * uv_dx + uv_dy * uv_dy)

            # Get 3D edge length
            global_v1 = cluster_uv.local_to_global.get(local_idx1)
            global_v2 = cluster_uv.local_to_global.get(local_idx2)

            if global_v1 is not None and global_v2 is not None:
                if global_v1 in mesh.vertices and global_v2 in mesh.vertices:
                    p1 = mesh.vertices[global_v1].position
                    p2 = mesh.vertices[global_v2].position
                    d = p2 - p1
                    len_3d = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

                    if uv_len > 0.0001 and len_3d > 0.0001:
                        total_uv_length += uv_len
                        total_3d_length += len_3d
                        edge_count += 1

    if edge_count > 0 and total_uv_length > 0:
        return total_3d_length / total_uv_length
    return 1.0


def clusters_to_json(mesh: Mesh, clusters: List, clusters_uv: List = None) -> List[Dict]:
    """Convert clusters and UV data to JSON, including seam edge labels."""
    import colorsys

    result = []
    n_clusters = len(clusters)

    # Build face -> cluster mapping
    face_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for fid in cluster:
            face_to_cluster[fid] = cluster_idx

    # Build edge -> faces mapping for seam detection
    edge_to_faces = {}
    for fid, face in mesh.faces.items():
        verts = face.vertex_ids
        for i in range(len(verts)):
            v1, v2 = verts[i], verts[(i + 1) % len(verts)]
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((fid, i))  # face id and edge index within face

    for i, cluster in enumerate(clusters):
        # Generate color
        hue = i / max(n_clusters, 1)
        sat = 0.7 + 0.3 * (i % 2)
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, sat)
        color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        cluster_data = {
            "id": i,
            "face_ids": list(cluster),
            "color": color,
        }

        # Add UV data if available
        if clusters_uv and i < len(clusters_uv):
            cuv = clusters_uv[i]
            cluster_data["uv_vertices"] = cuv.get_uv_vertices_list()
            cluster_data["uv_faces"] = cuv.uv_faces
            cluster_data["local_to_global"] = {str(k): v for k, v in cuv.local_to_global.items()}
            cluster_data["bbox"] = {
                "min": list(cuv.bbox_min),
                "max": list(cuv.bbox_max),
            }

            # Compute scale factor to match 3D dimensions
            # This is the ratio: 3D_edge_length / UV_edge_length
            scale_factor = compute_cluster_scale(mesh, cluster, cuv)
            cluster_data["scale_factor"] = scale_factor

            # Compute cluster metrics (surface area, physical dimensions)
            cluster_data["metrics"] = compute_cluster_metrics(mesh, cluster, cuv)

            # Compute seam edges - done after all clusters processed
            pass  # Will be added below

        result.append(cluster_data)

    # Second pass: compute seam edges with proper edge numbering
    # First assign edge numbers to all seam edges
    edge_to_cluster_edge = {}  # mesh_edge -> {cluster_id: edge_num}

    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            continue
        edge_counter = 0
        processed_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_edges:
                    continue
                processed_edges.add(mesh_edge)

                # Check if seam edge
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                is_seam = False
                for other_fid, _ in faces_on_edge:
                    other_c = face_to_cluster.get(other_fid)
                    if other_c is not None and other_c != cluster_idx:
                        is_seam = True
                        break

                if is_seam:
                    if mesh_edge not in edge_to_cluster_edge:
                        edge_to_cluster_edge[mesh_edge] = {}
                    edge_to_cluster_edge[mesh_edge][cluster_idx] = edge_counter
                    edge_counter += 1

    # Now add seam edges to each cluster with proper labels
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            continue

        cuv = clusters_uv[cluster_idx]
        seam_edges = []
        processed_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_edges:
                    continue
                processed_edges.add(mesh_edge)

                # Check if connects to another cluster
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                other_cluster = None

                for other_fid, _ in faces_on_edge:
                    other_c = face_to_cluster.get(other_fid)
                    if other_c is not None and other_c != cluster_idx:
                        other_cluster = other_c
                        break

                if other_cluster is not None:
                    uv1 = cuv.uv_coords.get(v1, (0, 0))
                    uv2 = cuv.uv_coords.get(v2, (0, 0))

                    # Get edge numbers
                    my_edge_num = edge_to_cluster_edge.get(mesh_edge, {}).get(cluster_idx, 0)
                    their_edge_num = edge_to_cluster_edge.get(mesh_edge, {}).get(other_cluster, 0)

                    seam_edges.append({
                        "edge_id": my_edge_num,
                        "my_edge": my_edge_num,
                        "their_edge": their_edge_num,
                        "connects_to": f"C{other_cluster}",
                        "label": f"E{my_edge_num}↔C{other_cluster}-E{their_edge_num}",
                        "v1": list(uv1),
                        "v2": list(uv2),
                    })

        result[cluster_idx]["seam_edges"] = seam_edges

    # Third pass: compute interior fold edges
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            continue

        cuv = clusters_uv[cluster_idx]
        fold_edges = []
        processed_interior_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_interior_edges:
                    continue
                processed_interior_edges.add(mesh_edge)

                # Get faces sharing this edge
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                faces_in_cluster = [f for f, _ in faces_on_edge if f in cluster]

                # Interior edge: shared by exactly 2 faces within this cluster
                if len(faces_in_cluster) == 2:
                    face1_id, face2_id = faces_in_cluster[0], faces_in_cluster[1]

                    # Compute fold type
                    fold_type, dihedral_angle = mesh.compute_fold_type(face1_id, face2_id)

                    # Skip flat edges (no visible fold needed)
                    if fold_type == 'flat':
                        continue

                    # Get UV coordinates for this edge
                    uv1 = cuv.uv_coords.get(v1, (0, 0))
                    uv2 = cuv.uv_coords.get(v2, (0, 0))

                    fold_edges.append({
                        "v1": list(uv1),
                        "v2": list(uv2),
                        "fold_type": fold_type,  # 'valley' or 'mountain'
                        "dihedral_angle": round(dihedral_angle, 1),
                    })

        result[cluster_idx]["fold_edges"] = fold_edges

    return result


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Redirect to frontend."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UV Dashboard</title>
        <meta http-equiv="refresh" content="0; url=/static/index.html">
    </head>
    <body>
        <p>Redirecting to <a href="/static/index.html">dashboard</a>...</p>
    </body>
    </html>
    """)


@app.get("/api/meshes")
async def list_meshes():
    """List available built-in meshes."""
    return {
        "meshes": [
            {"id": "cube", "name": "Cube", "description": "Simple cube"},
            {"id": "pyramid", "name": "Pyramid", "description": "Four-sided pyramid"},
            {"id": "cylinder", "name": "Cylinder", "description": "8-sided cylinder"},
            {"id": "dog", "name": "Dog", "description": "Simple dog shape"},
            {"id": "dog_detailed", "name": "Dog (Detailed)", "description": "Detailed dog shape"},
        ]
    }


@app.post("/api/meshes/load/{mesh_id}")
async def load_builtin_mesh(mesh_id: str):
    """Load a built-in mesh."""
    creators = {
        'cube': lambda: create_cube(1.0),
        'pyramid': lambda: create_pyramid(1.0, 1.5),
        'cylinder': lambda: create_cylinder(0.5, 2.0, 8),
        'dog': lambda: create_simple_dog(detail=1),
        'dog_detailed': lambda: create_simple_dog(detail=2),
    }

    if mesh_id not in creators:
        raise HTTPException(status_code=404, detail=f"Unknown mesh: {mesh_id}")

    session = get_session()
    session["mesh"] = creators[mesh_id]()
    session["mesh_name"] = mesh_id
    session["clusters"] = None
    session["clusters_uv"] = None

    return {
        "status": "ok",
        "mesh_name": mesh_id,
        "vertices": len(session["mesh"].vertices),
        "faces": len(session["mesh"].faces),
    }


@app.post("/api/meshes/upload")
async def upload_mesh(file: UploadFile = File(...)):
    """Upload an OBJ or STL file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".obj", ".stl"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use .obj or .stl")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        mesh = Mesh.load(tmp_path)
        session = get_session()
        session["mesh"] = mesh
        session["mesh_name"] = file.filename
        session["clusters"] = None
        session["clusters_uv"] = None

        return {
            "status": "ok",
            "mesh_name": file.filename,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }
    finally:
        os.unlink(tmp_path)


@app.get("/api/mesh")
async def get_mesh():
    """Get current mesh data."""
    session = get_session()
    if session["mesh"] is None:
        raise HTTPException(status_code=404, detail="No mesh loaded")

    return {
        "mesh_name": session["mesh_name"],
        "data": mesh_to_json(session["mesh"]),
    }


@app.post("/api/cluster")
async def run_clustering(request: ClusterRequest):
    """Run clustering algorithm on current mesh."""
    session = get_session()
    if session["mesh"] is None:
        raise HTTPException(status_code=404, detail="No mesh loaded")

    mesh = session["mesh"]

    # Apply edge weights
    dual = DualGraph(mesh)
    if session["mesh_name"] in ["dog", "dog_detailed"]:
        apply_pet_weights(dual)
    else:
        apply_dihedral_weights(dual)

    # Run algorithm
    if request.algorithm == AlgorithmType.MSF:
        result = maximum_spanning_forest(mesh, request.max_distortion)
    elif request.algorithm == AlgorithmType.GREEDY:
        result = greedy_region_growing(mesh, request.max_distortion)
    else:
        result = hierarchical_clustering(mesh, request.max_distortion)

    session["clusters"] = result.clusters
    session["clusters_uv"] = None  # Reset UV when clustering changes

    return {
        "status": "ok",
        "algorithm": request.algorithm,
        "num_clusters": len(result.clusters),
        "interior_edge_ratio": result.interior_edge_ratio,
    }


@app.post("/api/flatten")
async def flatten_clusters(request: FlattenRequest):
    """Flatten clusters to UV coordinates."""
    session = get_session()
    if session["mesh"] is None:
        raise HTTPException(status_code=404, detail="No mesh loaded")
    if session["clusters"] is None:
        raise HTTPException(status_code=400, detail="No clusters. Run clustering first.")

    clusters_uv = flatten_all_clusters(
        session["mesh"],
        session["clusters"],
        method=request.method.value
    )
    session["clusters_uv"] = clusters_uv

    return {
        "status": "ok",
        "method": request.method,
        "num_clusters": len(clusters_uv),
    }


@app.get("/api/export/json")
async def export_json():
    """Export current scene as JSON."""
    session = get_session()
    if session["mesh"] is None:
        raise HTTPException(status_code=404, detail="No mesh loaded")

    # Build seam edges
    seam_edges = []
    if session["clusters"]:
        mesh = session["mesh"]
        cluster_map = {}
        for i, cluster in enumerate(session["clusters"]):
            for fid in cluster:
                cluster_map[fid] = i

        for fid, face in mesh.faces.items():
            for neighbor_id in mesh.get_adjacent_faces(fid):
                if cluster_map.get(fid, -1) != cluster_map.get(neighbor_id, -2):
                    # This is a seam edge
                    shared = mesh.get_shared_edge(fid, neighbor_id)
                    if shared:
                        seam_edges.append(list(shared))

    return {
        "mesh": mesh_to_json(session["mesh"]),
        "clusters": clusters_to_json(
            session["mesh"],
            session["clusters"] or [],
            session["clusters_uv"]
        ),
        "seam_edges": seam_edges,
    }


@app.get("/api/export/svg")
async def export_svg(
    page_width: float = Query(default=8.5, description="Page width in inches (Letter=8.5)"),
    page_height: float = Query(default=11.0, description="Page height in inches (Letter=11)"),
):
    """Export UV layout as printable SVG."""
    session = get_session()
    if session["clusters_uv"] is None:
        raise HTTPException(status_code=400, detail="No UV data. Run flatten first.")

    # Pack clusters
    packed = pack_uv_islands(session["clusters_uv"])

    # Generate SVG to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", mode="w") as tmp:
        export_print_svg(
            packed,
            tmp.name,
            page_width=page_width,
            page_height=page_height,
        )
        tmp_path = tmp.name

    return FileResponse(
        tmp_path,
        media_type="image/svg+xml",
        filename="uv_layout.svg"
    )


@app.get("/api/export/pdf")
async def export_pdf(
    page_width: float = Query(default=8.5, description="Page width in inches (Letter=8.5)"),
    page_height: float = Query(default=11.0, description="Page height in inches (Letter=11)"),
    show_cut_lines: bool = Query(default=True, description="Show cut lines around shapes"),
    show_fold_tabs: bool = Query(default=True, description="Show fold tabs for assembly"),
):
    """Export UV layout as printer-ready PDF for Letter paper (8.5x11 inches)."""
    session = get_session()
    if session["clusters_uv"] is None:
        raise HTTPException(status_code=400, detail="No UV data. Run flatten first.")

    # Pack clusters
    packed = pack_uv_islands(session["clusters_uv"])

    # Generate PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    export_print_pdf(
        packed,
        tmp_path,
        page_width=page_width,
        page_height=page_height,
        show_cut_lines=show_cut_lines,
        show_fold_tabs=show_fold_tabs,
    )

    return FileResponse(
        tmp_path,
        media_type="application/pdf",
        filename="uv_layout.pdf"
    )


class ClusterPosition(BaseModel):
    cluster_id: int
    x: float  # normalized 0-1
    y: float  # normalized 0-1
    page: int = 0  # page number (0-indexed)
    scale_factor: float = 1.0  # 3D units per UV unit
    width_inches: float = 0  # actual width in inches
    height_inches: float = 0  # actual height in inches


class CustomLayoutRequest(BaseModel):
    positions: List[ClusterPosition]
    page_width: float = 8.5
    page_height: float = 11.0
    show_cut_lines: bool = True
    show_fold_tabs: bool = True
    print_scale: float = 1.0  # inches per 3D unit


@app.post("/api/export/pdf/custom")
async def export_pdf_custom(request: CustomLayoutRequest):
    """Export UV layout as multi-page PDF with custom cluster positions."""
    session = get_session()
    if session["clusters_uv"] is None:
        raise HTTPException(status_code=400, detail="No UV data. Run flatten first.")
    if session["mesh"] is None:
        raise HTTPException(status_code=400, detail="No mesh loaded.")
    if session["clusters"] is None:
        raise HTTPException(status_code=400, detail="No clusters. Run clustering first.")

    clusters_uv = session["clusters_uv"]
    mesh = session["mesh"]
    clusters = session["clusters"]

    # Compute seam edges for each cluster
    cluster_seam_edges = compute_seam_edges(mesh, clusters, clusters_uv)

    # Compute fold edges for each cluster
    cluster_fold_edges = compute_fold_edges(mesh, clusters, clusters_uv)

    # Group positions by page
    pages = {}
    for pos in request.positions:
        if pos.page not in pages:
            pages[pos.page] = []
        pages[pos.page].append(pos)

    # Generate multi-page PDF
    from scripts.utils.uv_packer import export_multipage_pdf

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    # Compute scale factors for each cluster
    cluster_scales = {}
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx < len(clusters_uv):
            cluster_scales[cluster_idx] = compute_cluster_scale(mesh, cluster, clusters_uv[cluster_idx])

    export_multipage_pdf(
        clusters_uv,
        pages,
        tmp_path,
        page_width=request.page_width,
        page_height=request.page_height,
        show_cut_lines=request.show_cut_lines,
        show_fold_tabs=request.show_fold_tabs,
        seam_edges=cluster_seam_edges,
        fold_edges=cluster_fold_edges,
        print_scale=request.print_scale,
        cluster_scales=cluster_scales,
    )

    return FileResponse(
        tmp_path,
        media_type="application/pdf",
        filename="uv_layout.pdf"
    )


def compute_seam_edges(mesh: Mesh, clusters: List, clusters_uv: List) -> Dict[int, List]:
    """Compute seam edges with connection labels showing edge number pairings."""
    # Build face -> cluster mapping
    face_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for fid in cluster:
            face_to_cluster[fid] = cluster_idx

    # Build edge -> faces mapping
    edge_to_faces = {}
    for fid, face in mesh.faces.items():
        verts = face.vertex_ids
        for i in range(len(verts)):
            v1, v2 = verts[i], verts[(i + 1) % len(verts)]
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((fid, i))

    # First pass: assign edge numbers to all boundary/seam edges per cluster
    # and build a mapping of mesh_edge -> (cluster, edge_num)
    edge_to_cluster_edge = {}  # mesh_edge -> {cluster_id: edge_num}
    cluster_edge_counters = {}

    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            continue

        cluster_edge_counters[cluster_idx] = 0
        processed_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_edges:
                    continue
                processed_edges.add(mesh_edge)

                # Check if this is a seam edge (connects to another cluster)
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                is_seam = False
                for other_fid, _ in faces_on_edge:
                    other_c = face_to_cluster.get(other_fid)
                    if other_c is not None and other_c != cluster_idx:
                        is_seam = True
                        break

                if is_seam:
                    edge_num = cluster_edge_counters[cluster_idx]
                    cluster_edge_counters[cluster_idx] += 1

                    if mesh_edge not in edge_to_cluster_edge:
                        edge_to_cluster_edge[mesh_edge] = {}
                    edge_to_cluster_edge[mesh_edge][cluster_idx] = edge_num

    # Second pass: build seam edge data with paired edge numbers
    result = {}

    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            continue

        cuv = clusters_uv[cluster_idx]
        seam_edges = []
        processed_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_edges:
                    continue
                processed_edges.add(mesh_edge)

                # Check if connects to another cluster
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                other_cluster = None

                for other_fid, _ in faces_on_edge:
                    other_c = face_to_cluster.get(other_fid)
                    if other_c is not None and other_c != cluster_idx:
                        other_cluster = other_c
                        break

                if other_cluster is not None:
                    uv1 = cuv.uv_coords.get(v1, (0, 0))
                    uv2 = cuv.uv_coords.get(v2, (0, 0))

                    # Get edge numbers for both sides
                    my_edge_num = edge_to_cluster_edge.get(mesh_edge, {}).get(cluster_idx, 0)
                    their_edge_num = edge_to_cluster_edge.get(mesh_edge, {}).get(other_cluster, 0)

                    seam_edges.append({
                        "edge_id": my_edge_num,
                        "my_edge": my_edge_num,
                        "their_edge": their_edge_num,
                        "connects_to": other_cluster,
                        "label": f"E{my_edge_num}↔C{other_cluster}-E{their_edge_num}",
                        "v1": list(uv1),
                        "v2": list(uv2),
                    })

        result[cluster_idx] = seam_edges

    return result


def compute_fold_edges(mesh: Mesh, clusters: List, clusters_uv: List) -> Dict[int, List]:
    """Compute interior fold edges with fold type (valley/mountain) for each cluster."""
    # Build face -> cluster mapping
    face_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for fid in cluster:
            face_to_cluster[fid] = cluster_idx

    # Build edge -> faces mapping
    edge_to_faces = {}
    for fid, face in mesh.faces.items():
        verts = face.vertex_ids
        for i in range(len(verts)):
            v1, v2 = verts[i], verts[(i + 1) % len(verts)]
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((fid, i))

    result = {}

    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx >= len(clusters_uv):
            result[cluster_idx] = []
            continue

        cuv = clusters_uv[cluster_idx]
        fold_edges = []
        processed_edges = set()

        for fid in cluster:
            if fid not in mesh.faces:
                continue
            face = mesh.faces[fid]
            verts = face.vertex_ids

            for edge_idx in range(len(verts)):
                v1, v2 = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
                mesh_edge = tuple(sorted([v1, v2]))

                if mesh_edge in processed_edges:
                    continue
                processed_edges.add(mesh_edge)

                # Get faces sharing this edge
                faces_on_edge = edge_to_faces.get(mesh_edge, [])
                faces_in_cluster = [f for f, _ in faces_on_edge if f in cluster]

                # Interior edge: shared by exactly 2 faces within this cluster
                if len(faces_in_cluster) == 2:
                    face1_id, face2_id = faces_in_cluster[0], faces_in_cluster[1]

                    # Compute fold type
                    fold_type, dihedral_angle = mesh.compute_fold_type(face1_id, face2_id)

                    # Skip flat edges (no visible fold needed)
                    if fold_type == 'flat':
                        continue

                    # Get UV coordinates for this edge
                    uv1 = cuv.uv_coords.get(v1, (0, 0))
                    uv2 = cuv.uv_coords.get(v2, (0, 0))

                    fold_edges.append({
                        "v1": list(uv1),
                        "v2": list(uv2),
                        "fold_type": fold_type,
                        "dihedral_angle": round(dihedral_angle, 1),
                    })

        result[cluster_idx] = fold_edges

    return result


# Mount static files for frontend (if built)
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path), html=True), name="static")
else:
    # Serve from src during development
    frontend_src = Path(__file__).parent.parent / "frontend"
    if frontend_src.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_src), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
