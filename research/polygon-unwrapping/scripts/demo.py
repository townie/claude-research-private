#!/usr/bin/env python3
"""
Polygon Unwrapping Demo Script

Demonstrates all clustering algorithms and visualization tools
on various example meshes.

Usage:
    python demo.py [--mesh MESH_TYPE] [--algorithm ALGORITHM] [--visualize]
    python demo.py --file path/to/mesh.obj [--algorithm ALGORITHM]

Examples:
    python demo.py --mesh cube
    python demo.py --mesh dog --algorithm all --visualize
    python demo.py --mesh cylinder --algorithm msf
    python demo.py --file mymodel.obj --algorithm msf
"""

import sys
import argparse
from typing import List, Set

from scripts.utils.mesh import Mesh, create_cube, create_pyramid, create_cylinder, create_simple_dog
from scripts.utils.dual_graph import DualGraph
from scripts.utils.edge_weights import apply_dihedral_weights, apply_composite_weights, apply_pet_weights
from scripts.utils.quality_metrics import generate_quality_report, format_quality_report, compare_clustering_methods
from scripts.utils.optimization import full_optimization_pipeline

from scripts.algorithms.maximum_spanning_forest import maximum_spanning_forest
from scripts.algorithms.greedy_region_growing import greedy_region_growing
from scripts.algorithms.hierarchical_clustering import hierarchical_clustering


def create_mesh(mesh_type: str) -> Mesh:
    """Create a mesh based on type name."""
    creators = {
        'cube': lambda: create_cube(1.0),
        'pyramid': lambda: create_pyramid(1.0, 1.5),
        'cylinder': lambda: create_cylinder(0.5, 2.0, 8),
        'dog': lambda: create_simple_dog(detail=1),
        'dog_detailed': lambda: create_simple_dog(detail=2),
    }

    if mesh_type not in creators:
        print(f"Unknown mesh type: {mesh_type}")
        print(f"Available: {', '.join(creators.keys())}")
        sys.exit(1)

    return creators[mesh_type]()


def run_algorithm(mesh: Mesh, algorithm: str, max_distortion: float = 0.3) -> List[Set[int]]:
    """Run a clustering algorithm."""
    if algorithm == 'msf':
        result = maximum_spanning_forest(mesh, max_distortion)
        return result.clusters
    elif algorithm == 'greedy':
        result = greedy_region_growing(mesh, max_distortion)
        return result.clusters
    elif algorithm == 'hierarchical':
        result = hierarchical_clustering(mesh, max_distortion)
        return result.clusters
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available: msf, greedy, hierarchical, all")
        sys.exit(1)


def demo_single_algorithm(mesh: Mesh, algorithm: str, max_distortion: float = 0.3):
    """Demo a single algorithm on a mesh."""
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"{'='*60}")

    clusters = run_algorithm(mesh, algorithm, max_distortion)

    # Generate report
    report = generate_quality_report(mesh, clusters)
    print(format_quality_report(report))

    # Print cluster details
    print("\nCluster Details:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {len(cluster)} faces")

    return clusters


def demo_all_algorithms(mesh: Mesh, max_distortion: float = 0.3):
    """Compare all algorithms on a mesh."""
    print(f"\n{'='*60}")
    print("COMPARING ALL ALGORITHMS")
    print(f"{'='*60}")

    algorithms = ['msf', 'greedy', 'hierarchical']
    results = {}

    for alg in algorithms:
        clusters = run_algorithm(mesh, alg, max_distortion)
        results[alg] = clusters

    # Add optimized version
    print("\nRunning optimization on MSF result...")
    optimized = full_optimization_pipeline(
        mesh, results['msf'],
        max_distortion=max_distortion,
        use_annealing=True,
        verbose=True
    )
    results['msf_optimized'] = optimized

    # Print comparison
    print("\n" + compare_clustering_methods(mesh, results))

    return results


def demo_with_visualization(mesh: Mesh, clusters: List[Set[int]], output_prefix: str = "demo"):
    """Create and save visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        from visualize import MeshVisualizer

        print(f"\nGenerating visualizations...")

        viz = MeshVisualizer(mesh, clusters)
        viz.save_all_visualizations(output_prefix)

        print(f"Saved visualizations with prefix: {output_prefix}")

    except ImportError as e:
        print(f"\nVisualization requires matplotlib: {e}")
        print("Install with: pip install matplotlib")


def demo_optimization_pipeline(mesh: Mesh):
    """Demo the full optimization pipeline."""
    print(f"\n{'='*60}")
    print("OPTIMIZATION PIPELINE DEMO")
    print(f"{'='*60}")

    # Start with MSF
    print("\n1. Initial clustering (MSF)...")
    initial = maximum_spanning_forest(mesh, max_distortion=0.4)

    print(f"   Clusters: {len(initial.clusters)}")
    print(f"   Interior edges: {initial.interior_edge_ratio:.1%}")

    # Run optimization
    print("\n2. Running optimization pipeline...")
    optimized = full_optimization_pipeline(
        mesh, initial.clusters,
        min_cluster_size=3,
        max_distortion=0.3,
        use_annealing=True,
        verbose=True
    )

    # Compare
    print("\n3. Comparison:")
    comparison = compare_clustering_methods(mesh, {
        'Initial (MSF)': initial.clusters,
        'Optimized': optimized,
    })
    print(comparison)

    return optimized


def main():
    parser = argparse.ArgumentParser(
        description='Polygon Unwrapping Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --mesh cube
    python demo.py --mesh dog --algorithm all
    python demo.py --mesh dog --algorithm msf --visualize
    python demo.py --mesh dog --optimize
    python demo.py --file mymodel.obj --algorithm msf
        """
    )

    parser.add_argument('--mesh', type=str, default='dog',
                        choices=['cube', 'pyramid', 'cylinder', 'dog', 'dog_detailed'],
                        help='Built-in mesh type to use')

    parser.add_argument('--file', type=str, default=None,
                        help='Path to mesh file (.obj or .stl) to load (overrides --mesh)')

    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['msf', 'greedy', 'hierarchical', 'all'],
                        help='Algorithm to run')

    parser.add_argument('--distortion', type=float, default=0.3,
                        help='Maximum distortion threshold')

    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualizations')

    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization pipeline demo')

    parser.add_argument('--output', type=str, default='demo',
                        help='Output prefix for visualization files')

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("POLYGON UNWRAPPING DEMO")
    print("Edge-Maximized Clustering for UV Unwrapping")
    print("=" * 60)

    # Create mesh
    if args.file:
        import os
        print(f"\nLoading mesh from: {args.file}")
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        mesh = Mesh.load(args.file)
        mesh_name = os.path.basename(args.file)
    else:
        print(f"\nCreating mesh: {args.mesh}")
        mesh = create_mesh(args.mesh)
        mesh_name = args.mesh

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Apply edge weights
    dual = DualGraph(mesh)
    if not args.file and args.mesh in ['dog', 'dog_detailed']:
        apply_pet_weights(dual)
        print("Applied pet-optimized edge weights")
    else:
        apply_dihedral_weights(dual)
        print("Applied dihedral edge weights")

    # Run demo
    if args.optimize:
        clusters = demo_optimization_pipeline(mesh)
    elif args.algorithm == 'all':
        results = demo_all_algorithms(mesh, args.distortion)
        clusters = results.get('msf_optimized', results.get('msf'))
    else:
        clusters = demo_single_algorithm(mesh, args.algorithm, args.distortion)

    # Visualization
    if args.visualize:
        demo_with_visualization(mesh, clusters, args.output)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
