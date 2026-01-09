#!/usr/bin/env python3
"""
Visualization tool for polygon unwrapping results.

Provides multiple visualization modes:
- 3D mesh with cluster coloring
- 2D UV layout preview
- Edge weight heatmap
- Quality metrics dashboard

Requires: matplotlib, numpy
Optional: plotly (for interactive 3D)
"""

import numpy as np
from typing import List, Set, Dict, Optional, Tuple
import colorsys

from scripts.utils.mesh import Mesh
from scripts.utils.dual_graph import DualGraph
from scripts.utils.quality_metrics import generate_quality_report, format_quality_report


def generate_cluster_colors(n_clusters: int) -> List[Tuple[float, float, float]]:
    """Generate visually distinct colors for clusters."""
    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate saturation
        lightness = 0.5 + 0.15 * ((i // 2) % 2)  # Alternate lightness
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def get_face_cluster_colors(mesh: Mesh, clusters: List[Set[int]]) -> Dict[int, Tuple[float, float, float]]:
    """Get color for each face based on cluster membership."""
    colors = generate_cluster_colors(len(clusters))
    face_colors = {}

    for i, cluster in enumerate(clusters):
        for face_id in cluster:
            face_colors[face_id] = colors[i]

    return face_colors


class MeshVisualizer:
    """
    Visualizer for mesh clustering results.

    Supports matplotlib (2D/3D) and optional plotly (interactive 3D).
    """

    def __init__(self, mesh: Mesh, clusters: Optional[List[Set[int]]] = None):
        self.mesh = mesh
        self.clusters = clusters or [set(mesh.faces.keys())]
        self.colors = get_face_cluster_colors(mesh, self.clusters)

    def visualize_3d_matplotlib(self, title: str = "Mesh Clusters",
                                 show_edges: bool = True,
                                 highlight_seams: bool = True,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize mesh in 3D using matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Collect face polygons
        polygons = []
        face_colors_list = []

        for face_id, face in self.mesh.faces.items():
            verts = [self.mesh.vertices[v].position for v in face.vertex_ids]
            polygons.append(verts)
            face_colors_list.append(self.colors.get(face_id, (0.5, 0.5, 0.5)))

        # Create collection
        collection = Poly3DCollection(polygons, alpha=0.8)
        collection.set_facecolors(face_colors_list)

        if show_edges:
            collection.set_edgecolor('black')
            collection.set_linewidth(0.5)

        ax.add_collection3d(collection)

        # Highlight seams (cut edges)
        if highlight_seams and len(self.clusters) > 1:
            dual = DualGraph(self.mesh)
            face_to_cluster = {}
            for i, cluster in enumerate(self.clusters):
                for f in cluster:
                    face_to_cluster[f] = i

            for edge in dual.edges.values():
                c1 = face_to_cluster.get(edge.face1_id, -1)
                c2 = face_to_cluster.get(edge.face2_id, -2)

                if c1 != c2:
                    v1 = self.mesh.vertices[edge.mesh_edge[0]].position
                    v2 = self.mesh.vertices[edge.mesh_edge[1]].position
                    ax.plot3D([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                              color='red', linewidth=2)

        # Set axis limits
        all_verts = [v.position for v in self.mesh.vertices.values()]
        min_pt = np.min(all_verts, axis=0)
        max_pt = np.max(all_verts, axis=0)
        center = (min_pt + max_pt) / 2
        max_range = np.max(max_pt - min_pt) * 0.6

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # Add legend
        colors = generate_cluster_colors(len(self.clusters))
        for i, (cluster, color) in enumerate(zip(self.clusters, colors)):
            ax.scatter([], [], [], c=[color], label=f'Cluster {i} ({len(cluster)} faces)')
        ax.legend(loc='upper left')

        plt.tight_layout()
        return fig, ax

    def visualize_2d_layout(self, title: str = "UV Layout Preview",
                             figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize clusters as a 2D UV layout preview.

        Projects each cluster to 2D and arranges them in a grid.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        n_clusters = len(self.clusters)
        cols = int(np.ceil(np.sqrt(n_clusters)))
        rows = int(np.ceil(n_clusters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_clusters == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        colors = generate_cluster_colors(n_clusters)

        for i, cluster in enumerate(self.clusters):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            # Project cluster faces to 2D (simple orthographic)
            polygons = []
            for face_id in cluster:
                face = self.mesh.faces[face_id]
                verts_3d = [self.mesh.vertices[v].position for v in face.vertex_ids]

                # Simple projection: use XZ plane for vertical objects
                verts_2d = [(v[0], v[2]) for v in verts_3d]
                polygons.append(MplPolygon(verts_2d))

            if polygons:
                collection = PatchCollection(polygons, alpha=0.7,
                                             facecolor=colors[i],
                                             edgecolor='black',
                                             linewidth=0.5)
                ax.add_collection(collection)

                # Set limits
                all_points = []
                for face_id in cluster:
                    for v_id in self.mesh.faces[face_id].vertex_ids:
                        pos = self.mesh.vertices[v_id].position
                        all_points.append((pos[0], pos[2]))

                if all_points:
                    xs, ys = zip(*all_points)
                    margin = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
                    ax.set_xlim(min(xs) - margin, max(xs) + margin)
                    ax.set_ylim(min(ys) - margin, max(ys) + margin)

            ax.set_aspect('equal')
            ax.set_title(f'Cluster {i}\n({len(cluster)} faces)')
            ax.axis('off')

        # Hide empty subplots
        for i in range(n_clusters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig, axes

    def visualize_edge_weights(self, title: str = "Edge Weights",
                                figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize edge weights as a heatmap on the mesh.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        dual = DualGraph(self.mesh)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Draw edges colored by weight
        weights = [e.weight for e in dual.edges.values()]
        min_w, max_w = min(weights), max(weights)
        range_w = max_w - min_w if max_w > min_w else 1.0

        for edge in dual.edges.values():
            v1 = self.mesh.vertices[edge.mesh_edge[0]].position
            v2 = self.mesh.vertices[edge.mesh_edge[1]].position

            # Normalize weight to color
            norm_w = (edge.weight - min_w) / range_w

            # Red (low/cut) to Green (high/keep)
            color = (1 - norm_w, norm_w, 0)

            ax.plot3D([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                      color=color, linewidth=2)

        # Set axis limits
        all_verts = [v.position for v in self.mesh.vertices.values()]
        min_pt = np.min(all_verts, axis=0)
        max_pt = np.max(all_verts, axis=0)
        center = (min_pt + max_pt) / 2
        max_range = np.max(max_pt - min_pt) * 0.6

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\nRed=Cut, Green=Keep')

        plt.tight_layout()
        return fig, ax

    def visualize_quality_dashboard(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Create a comprehensive quality dashboard.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        report = generate_quality_report(self.mesh, self.clusters)

        fig = plt.figure(figsize=figsize)

        # Layout: 2x2 grid
        # Top-left: 3D mesh view
        ax1 = fig.add_subplot(221, projection='3d')

        # Top-right: Quality metrics
        ax2 = fig.add_subplot(222)

        # Bottom-left: Cluster size distribution
        ax3 = fig.add_subplot(223)

        # Bottom-right: 2D cluster preview
        ax4 = fig.add_subplot(224)

        # === 3D Mesh View ===
        polygons = []
        face_colors_list = []

        for face_id, face in self.mesh.faces.items():
            verts = [self.mesh.vertices[v].position for v in face.vertex_ids]
            polygons.append(verts)
            face_colors_list.append(self.colors.get(face_id, (0.5, 0.5, 0.5)))

        collection = Poly3DCollection(polygons, alpha=0.8)
        collection.set_facecolors(face_colors_list)
        collection.set_edgecolor('black')
        collection.set_linewidth(0.3)
        ax1.add_collection3d(collection)

        all_verts = [v.position for v in self.mesh.vertices.values()]
        min_pt = np.min(all_verts, axis=0)
        max_pt = np.max(all_verts, axis=0)
        center = (min_pt + max_pt) / 2
        max_range = np.max(max_pt - min_pt) * 0.6

        ax1.set_xlim(center[0] - max_range, center[0] + max_range)
        ax1.set_ylim(center[1] - max_range, center[1] + max_range)
        ax1.set_zlim(center[2] - max_range, center[2] + max_range)
        ax1.set_title(f'3D Mesh ({len(self.mesh.faces)} faces)')

        # === Quality Metrics ===
        ax2.axis('off')
        metrics_text = [
            f"Quality Grade: {report.grade}",
            f"Overall Score: {report.overall_score:.2f}",
            "",
            f"Interior Edge Ratio: {report.interior_edge_ratio:.1%}",
            f"Normalized Seam Length: {report.normalized_seam_length:.1%}",
            "",
            f"Number of Clusters: {report.num_clusters}",
            f"Cluster Balance: {report.cluster_balance:.2f}",
            "",
            f"Max Distortion: {report.max_distortion:.3f}",
            f"Mean Distortion: {report.mean_distortion:.3f}",
        ]
        ax2.text(0.1, 0.9, '\n'.join(metrics_text),
                 transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title('Quality Metrics')

        # === Cluster Size Distribution ===
        colors = generate_cluster_colors(len(self.clusters))
        sizes = [len(c) for c in self.clusters]
        bars = ax3.bar(range(len(sizes)), sizes, color=colors)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Faces')
        ax3.set_title('Cluster Size Distribution')
        ax3.set_xticks(range(len(sizes)))

        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(size), ha='center', va='bottom', fontsize=9)

        # === 2D Preview (largest cluster) ===
        largest_cluster = max(self.clusters, key=len)
        largest_idx = self.clusters.index(largest_cluster)

        polygons_2d = []
        for face_id in largest_cluster:
            face = self.mesh.faces[face_id]
            verts_3d = [self.mesh.vertices[v].position for v in face.vertex_ids]
            verts_2d = [(v[0], v[2]) for v in verts_3d]
            polygons_2d.append(MplPolygon(verts_2d))

        if polygons_2d:
            collection = PatchCollection(polygons_2d, alpha=0.7,
                                         facecolor=colors[largest_idx],
                                         edgecolor='black',
                                         linewidth=0.5)
            ax4.add_collection(collection)

            all_points = []
            for face_id in largest_cluster:
                for v_id in self.mesh.faces[face_id].vertex_ids:
                    pos = self.mesh.vertices[v_id].position
                    all_points.append((pos[0], pos[2]))

            if all_points:
                xs, ys = zip(*all_points)
                margin = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
                ax4.set_xlim(min(xs) - margin, max(xs) + margin)
                ax4.set_ylim(min(ys) - margin, max(ys) + margin)

        ax4.set_aspect('equal')
        ax4.set_title(f'Largest Cluster (#{largest_idx}, {len(largest_cluster)} faces)')

        fig.suptitle('Polygon Unwrapping Quality Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def save_all_visualizations(self, output_prefix: str = "unwrap"):
        """Save all visualizations to files."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # 3D view
        fig, ax = self.visualize_3d_matplotlib()
        fig.savefig(f"{output_prefix}_3d.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_prefix}_3d.png")

        # 2D layout
        fig, ax = self.visualize_2d_layout()
        fig.savefig(f"{output_prefix}_2d_layout.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_prefix}_2d_layout.png")

        # Dashboard
        fig = self.visualize_quality_dashboard()
        fig.savefig(f"{output_prefix}_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_prefix}_dashboard.png")


def visualize_comparison(mesh: Mesh,
                          methods: Dict[str, List[Set[int]]],
                          figsize: Tuple[int, int] = (16, 8)):
    """
    Visualize multiple clustering methods side by side.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib not installed")
        return

    n_methods = len(methods)
    fig = plt.figure(figsize=figsize)

    for idx, (name, clusters) in enumerate(methods.items()):
        ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')

        colors = get_face_cluster_colors(mesh, clusters)

        polygons = []
        face_colors_list = []

        for face_id, face in mesh.faces.items():
            verts = [mesh.vertices[v].position for v in face.vertex_ids]
            polygons.append(verts)
            face_colors_list.append(colors.get(face_id, (0.5, 0.5, 0.5)))

        collection = Poly3DCollection(polygons, alpha=0.8)
        collection.set_facecolors(face_colors_list)
        collection.set_edgecolor('black')
        collection.set_linewidth(0.3)
        ax.add_collection3d(collection)

        all_verts = [v.position for v in mesh.vertices.values()]
        min_pt = np.min(all_verts, axis=0)
        max_pt = np.max(all_verts, axis=0)
        center = (min_pt + max_pt) / 2
        max_range = np.max(max_pt - min_pt) * 0.6

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        report = generate_quality_report(mesh, clusters)
        ax.set_title(f'{name}\n{len(clusters)} clusters, {report.interior_edge_ratio:.0%} edges, Grade: {report.grade}')

    fig.suptitle('Clustering Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# CLI entry point
if __name__ == "__main__":
    import sys

    print("Polygon Unwrapping Visualization Tool")
    print("=" * 40)

    # Demo with simple cube
    from utils.mesh import create_cube, create_simple_dog
    from algorithms import maximum_spanning_forest, greedy_region_growing, hierarchical_clustering

    # Create mesh
    print("\nCreating test mesh...")
    mesh = create_simple_dog()
    print(f"Mesh: {mesh}")

    # Run clustering
    print("\nRunning clustering algorithms...")
    msf_result = maximum_spanning_forest(mesh, max_distortion=0.4)
    greedy_result = greedy_region_growing(mesh, max_distortion=0.3)
    hier_result = hierarchical_clustering(mesh, max_distortion=0.3, target_clusters=5)

    # Visualize
    print("\nGenerating visualizations...")

    viz = MeshVisualizer(mesh, msf_result.clusters)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Save individual visualizations
        viz.save_all_visualizations("demo_msf")

        # Comparison
        fig = visualize_comparison(mesh, {
            'Max Spanning Forest': msf_result.clusters,
            'Greedy Growing': greedy_result.clusters,
            'Hierarchical': hier_result.clusters,
        })
        fig.savefig("demo_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: demo_comparison.png")

        print("\nVisualization complete!")

    except ImportError:
        print("\nmatplotlib not installed. Install with: pip install matplotlib")
        print("Printing text-based report instead:\n")
        report = generate_quality_report(mesh, msf_result.clusters)
        print(format_quality_report(report))
