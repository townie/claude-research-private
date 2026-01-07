"""
Dual graph representation for mesh clustering.

The dual graph has:
- Nodes: faces of the original mesh
- Edges: connections between adjacent faces (shared edges in mesh)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
import numpy as np

from .mesh import Mesh


@dataclass
class DualEdge:
    """An edge in the dual graph (connection between two faces)."""
    face1_id: int
    face2_id: int
    mesh_edge: Tuple[int, int]  # The original mesh edge
    weight: float = 1.0

    @property
    def key(self) -> Tuple[int, int]:
        """Canonical edge key."""
        return tuple(sorted([self.face1_id, self.face2_id]))

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def other_face(self, face_id: int) -> int:
        """Get the other face connected by this edge."""
        if face_id == self.face1_id:
            return self.face2_id
        return self.face1_id


class DualGraph:
    """
    Dual graph of a mesh for clustering operations.

    Nodes are faces, edges represent face adjacency.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.edges: Dict[Tuple[int, int], DualEdge] = {}
        self.node_edges: Dict[int, Set[Tuple[int, int]]] = {}

        self._build_from_mesh()

    def _build_from_mesh(self):
        """Build dual graph from mesh adjacency."""
        # Initialize node edges
        for face_id in self.mesh.faces:
            self.node_edges[face_id] = set()

        # Create dual edges from mesh interior edges
        for mesh_edge in self.mesh.get_interior_edges():
            faces = self.mesh.get_edge_faces(mesh_edge)

            if len(faces) == 2:
                f1, f2 = faces
                dual_edge = DualEdge(
                    face1_id=f1,
                    face2_id=f2,
                    mesh_edge=mesh_edge,
                    weight=1.0
                )

                self.edges[dual_edge.key] = dual_edge
                self.node_edges[f1].add(dual_edge.key)
                self.node_edges[f2].add(dual_edge.key)

    def get_neighbors(self, face_id: int) -> Set[int]:
        """Get all faces adjacent to a given face."""
        neighbors = set()
        for edge_key in self.node_edges.get(face_id, set()):
            edge = self.edges[edge_key]
            neighbors.add(edge.other_face(face_id))
        return neighbors

    def get_edge(self, face1_id: int, face2_id: int) -> Optional[DualEdge]:
        """Get the edge between two faces, if it exists."""
        key = tuple(sorted([face1_id, face2_id]))
        return self.edges.get(key)

    def set_weight(self, face1_id: int, face2_id: int, weight: float):
        """Set the weight of an edge."""
        key = tuple(sorted([face1_id, face2_id]))
        if key in self.edges:
            self.edges[key].weight = weight

    def get_all_edges(self) -> List[DualEdge]:
        """Get all edges as a list."""
        return list(self.edges.values())

    def get_edges_sorted_by_weight(self, descending: bool = True) -> List[DualEdge]:
        """Get edges sorted by weight."""
        return sorted(
            self.edges.values(),
            key=lambda e: e.weight,
            reverse=descending
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_edges)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def __repr__(self):
        return f"DualGraph(nodes={self.num_nodes}, edges={self.num_edges})"


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> bool:
        """
        Union by rank.
        Returns True if x and y were in different sets.
        """
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def same_set(self, x, y) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def get_clusters(self) -> List[Set]:
        """Get all clusters as sets."""
        clusters = {}
        for element in self.parent:
            root = self.find(element)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(element)
        return list(clusters.values())
