"""
Core mesh data structures for polygon unwrapping.

Provides Mesh, Face, Vertex, and Edge classes with utilities
for adjacency computation and geometric queries.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


@dataclass
class Vertex:
    """A vertex in 3D space."""
    id: int
    position: np.ndarray  # (x, y, z)
    uv: Optional[np.ndarray] = None  # (u, v) for UV mapping

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Edge:
    """An edge connecting two vertices."""
    v1_id: int
    v2_id: int

    @property
    def key(self) -> Tuple[int, int]:
        """Canonical edge key (sorted vertex ids)."""
        return tuple(sorted([self.v1_id, self.v2_id]))

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key


@dataclass
class Face:
    """A polygonal face (triangle or quad)."""
    id: int
    vertex_ids: List[int]
    normal: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def get_edges(self) -> List[Tuple[int, int]]:
        """Get all edges of this face as sorted tuples."""
        edges = []
        n = len(self.vertex_ids)
        for i in range(n):
            edge = tuple(sorted([self.vertex_ids[i], self.vertex_ids[(i + 1) % n]]))
            edges.append(edge)
        return edges


class Mesh:
    """
    A polygonal mesh with vertices, faces, and computed adjacency.

    Supports:
    - Loading from OBJ files or programmatic construction
    - Face and edge adjacency queries
    - Geometric computations (normals, areas, angles)
    - UV coordinate storage
    """

    def __init__(self):
        self.vertices: Dict[int, Vertex] = {}
        self.faces: Dict[int, Face] = {}

        # Computed on demand
        self._edge_to_faces: Dict[Tuple[int, int], List[int]] = None
        self._face_adjacency: Dict[int, Set[int]] = None
        self._vertex_to_faces: Dict[int, Set[int]] = None
        self._face_normals: Dict[int, np.ndarray] = None
        self._face_areas: Dict[int, float] = None

    def add_vertex(self, id: int, position: np.ndarray) -> Vertex:
        """Add a vertex to the mesh."""
        vertex = Vertex(id=id, position=np.array(position, dtype=float))
        self.vertices[id] = vertex
        self._invalidate_cache()
        return vertex

    def add_face(self, id: int, vertex_ids: List[int]) -> Face:
        """Add a face to the mesh."""
        face = Face(id=id, vertex_ids=list(vertex_ids))
        self.faces[id] = face
        self._invalidate_cache()
        return face

    def _invalidate_cache(self):
        """Invalidate computed caches."""
        self._edge_to_faces = None
        self._face_adjacency = None
        self._vertex_to_faces = None
        self._face_normals = None
        self._face_areas = None

    def _compute_edge_to_faces(self):
        """Compute mapping from edges to faces."""
        if self._edge_to_faces is not None:
            return

        self._edge_to_faces = defaultdict(list)
        for face_id, face in self.faces.items():
            for edge in face.get_edges():
                self._edge_to_faces[edge].append(face_id)

    def _compute_face_adjacency(self):
        """Compute face-to-face adjacency via shared edges."""
        if self._face_adjacency is not None:
            return

        self._compute_edge_to_faces()
        self._face_adjacency = defaultdict(set)

        for edge, face_ids in self._edge_to_faces.items():
            if len(face_ids) == 2:
                f1, f2 = face_ids
                self._face_adjacency[f1].add(f2)
                self._face_adjacency[f2].add(f1)

    def _compute_vertex_to_faces(self):
        """Compute mapping from vertices to faces."""
        if self._vertex_to_faces is not None:
            return

        self._vertex_to_faces = defaultdict(set)
        for face_id, face in self.faces.items():
            for v_id in face.vertex_ids:
                self._vertex_to_faces[v_id].add(face_id)

    def get_adjacent_faces(self, face_id: int) -> Set[int]:
        """Get all faces adjacent to a given face."""
        self._compute_face_adjacency()
        return self._face_adjacency.get(face_id, set())

    def get_edge_faces(self, edge: Tuple[int, int]) -> List[int]:
        """Get faces sharing an edge."""
        self._compute_edge_to_faces()
        return self._edge_to_faces.get(tuple(sorted(edge)), [])

    def get_interior_edges(self) -> List[Tuple[int, int]]:
        """Get all interior edges (shared by exactly 2 faces)."""
        self._compute_edge_to_faces()
        return [edge for edge, faces in self._edge_to_faces.items()
                if len(faces) == 2]

    def get_boundary_edges(self) -> List[Tuple[int, int]]:
        """Get all boundary edges (belong to only 1 face)."""
        self._compute_edge_to_faces()
        return [edge for edge, faces in self._edge_to_faces.items()
                if len(faces) == 1]

    def compute_face_normal(self, face_id: int) -> np.ndarray:
        """Compute the normal vector for a face."""
        face = self.faces[face_id]

        if len(face.vertex_ids) < 3:
            return np.array([0, 1, 0])

        v0 = self.vertices[face.vertex_ids[0]].position
        v1 = self.vertices[face.vertex_ids[1]].position
        v2 = self.vertices[face.vertex_ids[2]].position

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)

        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm

        return normal

    def get_face_normal(self, face_id: int) -> np.ndarray:
        """Get (cached) face normal."""
        if self._face_normals is None:
            self._face_normals = {}

        if face_id not in self._face_normals:
            self._face_normals[face_id] = self.compute_face_normal(face_id)

        return self._face_normals[face_id]

    def compute_face_area(self, face_id: int) -> float:
        """Compute the area of a face."""
        face = self.faces[face_id]

        if len(face.vertex_ids) < 3:
            return 0.0

        # Triangulate and sum areas
        v0 = self.vertices[face.vertex_ids[0]].position
        total_area = 0.0

        for i in range(1, len(face.vertex_ids) - 1):
            v1 = self.vertices[face.vertex_ids[i]].position
            v2 = self.vertices[face.vertex_ids[i + 1]].position

            cross = np.cross(v1 - v0, v2 - v0)
            total_area += np.linalg.norm(cross) / 2

        return total_area

    def get_face_area(self, face_id: int) -> float:
        """Get (cached) face area."""
        if self._face_areas is None:
            self._face_areas = {}

        if face_id not in self._face_areas:
            self._face_areas[face_id] = self.compute_face_area(face_id)

        return self._face_areas[face_id]

    def get_face_centroid(self, face_id: int) -> np.ndarray:
        """Get the centroid of a face."""
        face = self.faces[face_id]
        positions = [self.vertices[v_id].position for v_id in face.vertex_ids]
        return np.mean(positions, axis=0)

    def compute_dihedral_angle(self, face1_id: int, face2_id: int) -> float:
        """
        Compute dihedral angle between two adjacent faces in degrees.
        Returns 180 for coplanar faces, 0 for faces folded back on each other.
        """
        n1 = self.get_face_normal(face1_id)
        n2 = self.get_face_normal(face2_id)

        # Angle between normals
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        # Dihedral is supplement
        dihedral = 180.0 - np.degrees(angle_rad)
        return dihedral

    def get_edge_length(self, edge: Tuple[int, int]) -> float:
        """Get the length of an edge."""
        v1 = self.vertices[edge[0]].position
        v2 = self.vertices[edge[1]].position
        return np.linalg.norm(v2 - v1)

    def get_edge_center(self, edge: Tuple[int, int]) -> np.ndarray:
        """Get the center point of an edge."""
        v1 = self.vertices[edge[0]].position
        v2 = self.vertices[edge[1]].position
        return (v1 + v2) / 2

    def get_shared_edge(self, face1_id: int, face2_id: int) -> Optional[Tuple[int, int]]:
        """Get the edge shared by two faces, if any."""
        edges1 = set(self.faces[face1_id].get_edges())
        edges2 = set(self.faces[face2_id].get_edges())
        shared = edges1 & edges2
        return list(shared)[0] if shared else None

    def compute_fold_type(self, face1_id: int, face2_id: int, flat_threshold: float = 5.0) -> Tuple[str, float]:
        """
        Determine the fold type between two adjacent faces.

        Returns:
            Tuple of (fold_type, dihedral_angle) where:
            - fold_type: 'valley' (fold in/convex), 'mountain' (fold out/concave), or 'flat'
            - dihedral_angle: angle in degrees (180 = flat, <180 = convex, >180 = concave)

        The fold direction is determined by checking which side of face1's plane
        the opposite vertex of face2 lies on relative to face1's normal.
        """
        # Get dihedral angle first
        dihedral = self.compute_dihedral_angle(face1_id, face2_id)

        # Check if effectively flat
        if abs(dihedral - 180.0) < flat_threshold:
            return ('flat', dihedral)

        # Get the shared edge
        shared_edge = self.get_shared_edge(face1_id, face2_id)
        if shared_edge is None:
            return ('flat', 180.0)

        # Get face vertices
        face1 = self.faces[face1_id]
        face2 = self.faces[face2_id]
        edge_set = set(shared_edge)

        # Find the opposite vertex of face2 (vertex not on shared edge)
        opp2_candidates = [v for v in face2.vertex_ids if v not in edge_set]
        if not opp2_candidates:
            return ('flat', dihedral)
        opp2_id = opp2_candidates[0]
        opp2_pos = self.vertices[opp2_id].position

        # Get face1's normal and centroid
        n1 = self.get_face_normal(face1_id)
        centroid1 = self.get_face_centroid(face1_id)

        # Vector from face1 centroid to face2's opposite vertex
        to_opp2 = opp2_pos - centroid1

        # Check which side of face1's plane opp2 is on
        dot_product = np.dot(to_opp2, n1)

        # If opp2 is on the opposite side from where n1 points -> convex -> valley fold
        # If opp2 is on the same side as n1 points -> concave -> mountain fold
        if dot_product < 0:
            # Convex edge: faces bend outward (like outside of cube)
            # Paper needs to fold INWARD (valley fold) to recreate this
            return ('valley', dihedral)
        else:
            # Concave edge: faces bend inward (like inside corner)
            # Paper needs to fold OUTWARD (mountain fold) to recreate this
            return ('mountain', 180.0 + (180.0 - dihedral))

    def total_area(self) -> float:
        """Get total surface area of the mesh."""
        return sum(self.get_face_area(f) for f in self.faces)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max)."""
        positions = [v.position for v in self.vertices.values()]
        return np.min(positions, axis=0), np.max(positions, axis=0)

    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        min_pt, max_pt = self.bounding_box()
        return (min_pt + max_pt) / 2

    @classmethod
    def from_obj(cls, filepath: str) -> 'Mesh':
        """Load mesh from OBJ file."""
        mesh = cls()

        with open(filepath, 'r') as f:
            vertex_id = 0
            face_id = 0

            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()

                if parts[0] == 'v':
                    # Vertex
                    pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                    mesh.add_vertex(vertex_id, pos)
                    vertex_id += 1

                elif parts[0] == 'f':
                    # Face (OBJ uses 1-based indexing)
                    vertex_ids = []
                    for p in parts[1:]:
                        # Handle v, v/vt, v/vt/vn, v//vn formats
                        v_idx = int(p.split('/')[0]) - 1
                        vertex_ids.append(v_idx)
                    mesh.add_face(face_id, vertex_ids)
                    face_id += 1

        return mesh

    @classmethod
    def from_stl(cls, filepath: str, merge_tolerance: float = 1e-6) -> 'Mesh':
        """
        Load mesh from STL file (ASCII or binary).

        STL files store triangles with duplicated vertices, so this method
        merges vertices that are within merge_tolerance distance.

        Args:
            filepath: Path to STL file
            merge_tolerance: Distance threshold for merging vertices
        """
        import struct

        mesh = cls()

        # Try to detect if binary or ASCII
        with open(filepath, 'rb') as f:
            header = f.read(80)
            # Check if it looks like ASCII (starts with "solid")
            try:
                header_str = header.decode('ascii').strip()
                is_ascii = header_str.startswith('solid')
            except UnicodeDecodeError:
                is_ascii = False

            # Additional check: ASCII files shouldn't have the triangle count
            # right after header in a way that makes sense as binary
            if is_ascii:
                f.seek(0)
                first_lines = f.read(1000).decode('ascii', errors='ignore')
                # If we see 'facet normal' it's definitely ASCII
                if 'facet normal' not in first_lines.lower():
                    is_ascii = False

        # Vertex deduplication using spatial hashing
        vertex_map = {}  # (rounded_x, rounded_y, rounded_z) -> vertex_id

        def get_or_create_vertex(pos):
            """Get existing vertex or create new one."""
            # Round position for hashing
            key = (
                round(pos[0] / merge_tolerance),
                round(pos[1] / merge_tolerance),
                round(pos[2] / merge_tolerance)
            )
            if key in vertex_map:
                return vertex_map[key]

            v_id = len(mesh.vertices)
            mesh.add_vertex(v_id, pos)
            vertex_map[key] = v_id
            return v_id

        if is_ascii:
            # Parse ASCII STL
            with open(filepath, 'r') as f:
                face_id = 0
                current_vertices = []

                for line in f:
                    line = line.strip().lower()

                    if line.startswith('vertex'):
                        parts = line.split()
                        pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                        v_id = get_or_create_vertex(pos)
                        current_vertices.append(v_id)

                    elif line.startswith('endfacet'):
                        if len(current_vertices) == 3:
                            mesh.add_face(face_id, current_vertices)
                            face_id += 1
                        current_vertices = []
        else:
            # Parse binary STL
            with open(filepath, 'rb') as f:
                # Skip 80-byte header
                f.read(80)

                # Read triangle count (4 bytes, little-endian)
                num_triangles = struct.unpack('<I', f.read(4))[0]

                for face_id in range(num_triangles):
                    # Skip normal (3 floats = 12 bytes)
                    f.read(12)

                    # Read 3 vertices (each 3 floats = 12 bytes)
                    vertex_ids = []
                    for _ in range(3):
                        x, y, z = struct.unpack('<3f', f.read(12))
                        v_id = get_or_create_vertex([x, y, z])
                        vertex_ids.append(v_id)

                    mesh.add_face(face_id, vertex_ids)

                    # Skip attribute byte count (2 bytes)
                    f.read(2)

        return mesh

    @classmethod
    def load(cls, filepath: str) -> 'Mesh':
        """
        Load mesh from file, auto-detecting format by extension.

        Supported formats: .obj, .stl
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.obj':
            return cls.from_obj(filepath)
        elif ext == '.stl':
            return cls.from_stl(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .obj, .stl")

    def to_obj(self, filepath: str):
        """Save mesh to OBJ file."""
        with open(filepath, 'w') as f:
            f.write("# Mesh exported by polygon-unwrapping\n")

            # Write vertices
            for v_id in sorted(self.vertices.keys()):
                v = self.vertices[v_id]
                f.write(f"v {v.position[0]} {v.position[1]} {v.position[2]}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face_id in sorted(self.faces.keys()):
                face = self.faces[face_id]
                indices = ' '.join(str(v + 1) for v in face.vertex_ids)
                f.write(f"f {indices}\n")

    def __repr__(self):
        return f"Mesh(vertices={len(self.vertices)}, faces={len(self.faces)})"


# Factory functions for common primitives

def create_cube(size: float = 1.0) -> Mesh:
    """Create a simple cube mesh."""
    mesh = Mesh()
    s = size / 2

    # Vertices
    positions = [
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Back
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],       # Front
    ]
    for i, pos in enumerate(positions):
        mesh.add_vertex(i, pos)

    # Faces (quads)
    faces = [
        [0, 1, 2, 3],  # Back
        [4, 7, 6, 5],  # Front
        [0, 4, 5, 1],  # Bottom
        [2, 6, 7, 3],  # Top
        [0, 3, 7, 4],  # Left
        [1, 5, 6, 2],  # Right
    ]
    for i, f in enumerate(faces):
        mesh.add_face(i, f)

    return mesh


def create_pyramid(base_size: float = 1.0, height: float = 1.5) -> Mesh:
    """Create a square pyramid mesh."""
    mesh = Mesh()
    s = base_size / 2

    # Base vertices
    mesh.add_vertex(0, [-s, 0, -s])
    mesh.add_vertex(1, [s, 0, -s])
    mesh.add_vertex(2, [s, 0, s])
    mesh.add_vertex(3, [-s, 0, s])
    # Apex
    mesh.add_vertex(4, [0, height, 0])

    # Faces
    mesh.add_face(0, [0, 1, 2, 3])  # Base
    mesh.add_face(1, [0, 4, 1])      # Side 1
    mesh.add_face(2, [1, 4, 2])      # Side 2
    mesh.add_face(3, [2, 4, 3])      # Side 3
    mesh.add_face(4, [3, 4, 0])      # Side 4

    return mesh


def create_cylinder(radius: float = 0.5, height: float = 2.0, segments: int = 8) -> Mesh:
    """Create a cylinder mesh."""
    mesh = Mesh()

    # Create vertices
    v_id = 0
    bottom_center = v_id
    mesh.add_vertex(v_id, [0, 0, 0])
    v_id += 1

    top_center = v_id
    mesh.add_vertex(v_id, [0, height, 0])
    v_id += 1

    bottom_ring = []
    top_ring = []

    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)

        bottom_ring.append(v_id)
        mesh.add_vertex(v_id, [x, 0, z])
        v_id += 1

        top_ring.append(v_id)
        mesh.add_vertex(v_id, [x, height, z])
        v_id += 1

    # Create faces
    f_id = 0

    # Bottom cap (triangles)
    for i in range(segments):
        next_i = (i + 1) % segments
        mesh.add_face(f_id, [bottom_center, bottom_ring[next_i], bottom_ring[i]])
        f_id += 1

    # Top cap (triangles)
    for i in range(segments):
        next_i = (i + 1) % segments
        mesh.add_face(f_id, [top_center, top_ring[i], top_ring[next_i]])
        f_id += 1

    # Side faces (quads)
    for i in range(segments):
        next_i = (i + 1) % segments
        mesh.add_face(f_id, [bottom_ring[i], bottom_ring[next_i],
                             top_ring[next_i], top_ring[i]])
        f_id += 1

    return mesh


def create_simple_dog(detail: int = 1) -> Mesh:
    """
    Create a very simple low-poly dog mesh.
    detail=1 gives ~80 faces, detail=2 gives ~150 faces.
    """
    mesh = Mesh()
    v_id = 0
    f_id = 0

    def add_box(center, size, start_v, start_f):
        """Add a box primitive."""
        cx, cy, cz = center
        sx, sy, sz = [s/2 for s in size]

        # 8 vertices
        positions = [
            [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
            [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
            [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
            [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
        ]

        for i, pos in enumerate(positions):
            mesh.add_vertex(start_v + i, pos)

        # 6 faces
        faces = [
            [0, 3, 2, 1], [4, 5, 6, 7],  # Front/Back
            [0, 1, 5, 4], [2, 3, 7, 6],  # Bottom/Top
            [0, 4, 7, 3], [1, 2, 6, 5],  # Left/Right
        ]

        for i, f in enumerate(faces):
            mesh.add_face(start_f + i, [start_v + v for v in f])

        return start_v + 8, start_f + 6

    # Body
    v_id, f_id = add_box([0, 0.5, 0], [2.0, 0.8, 0.8], v_id, f_id)

    # Head
    v_id, f_id = add_box([1.3, 0.7, 0], [0.6, 0.6, 0.6], v_id, f_id)

    # Snout
    v_id, f_id = add_box([1.7, 0.55, 0], [0.3, 0.25, 0.3], v_id, f_id)

    # Front left leg
    v_id, f_id = add_box([0.6, 0, -0.3], [0.2, 0.5, 0.2], v_id, f_id)

    # Front right leg
    v_id, f_id = add_box([0.6, 0, 0.3], [0.2, 0.5, 0.2], v_id, f_id)

    # Back left leg
    v_id, f_id = add_box([-0.6, 0, -0.3], [0.2, 0.5, 0.2], v_id, f_id)

    # Back right leg
    v_id, f_id = add_box([-0.6, 0, 0.3], [0.2, 0.5, 0.2], v_id, f_id)

    # Tail
    v_id, f_id = add_box([-1.2, 0.7, 0], [0.4, 0.15, 0.15], v_id, f_id)

    # Ears (if detail > 1, add more)
    if detail > 1:
        v_id, f_id = add_box([1.2, 1.1, -0.2], [0.1, 0.2, 0.1], v_id, f_id)
        v_id, f_id = add_box([1.2, 1.1, 0.2], [0.1, 0.2, 0.1], v_id, f_id)

    return mesh
