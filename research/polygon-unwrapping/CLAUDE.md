# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research and Python implementations for polygon unwrapping (UV unwrapping) with edge-maximized clustering. The goal is to segment 3D mesh polygons into UV islands that maximize interior edges (minimize seams) while keeping distortion bounded.

## Common Commands

```bash
# Install dependencies with uv
uv sync

# Run demo with visualization
uv run python scripts/demo.py --mesh dog --visualize

# Compare all algorithms on a mesh
uv run python scripts/demo.py --mesh dog --algorithm all

# Run a specific algorithm (msf, greedy, hierarchical)
uv run python scripts/demo.py --mesh cube --algorithm msf

# Run optimization pipeline
uv run python scripts/demo.py --mesh dog --optimize

# Load custom mesh file (OBJ or STL)
uv run python scripts/demo.py --file path/to/mesh.obj --algorithm msf
uv run python scripts/demo.py --file path/to/mesh.stl --algorithm msf

# Install with optional interactive 3D visualization (plotly)
uv sync --extra interactive

# Available built-in meshes: cube, pyramid, cylinder, dog, dog_detailed
```

## Architecture

### Core Data Flow
1. **Mesh** (`utils/mesh.py`) - Input 3D mesh with vertices, faces, and computed adjacency
2. **DualGraph** (`utils/dual_graph.py`) - Converts face adjacency to graph where faces are nodes
3. **Edge Weights** (`utils/edge_weights.py`) - Assigns weights to dual graph edges (higher = prefer not to cut)
4. **Clustering Algorithm** (`algorithms/`) - Partitions faces into clusters maximizing interior edges
5. **Quality Metrics** (`utils/quality_metrics.py`) - Evaluates results (interior edge ratio, distortion, cluster balance)
6. **Optimization** (`utils/optimization.py`) - Post-processing to refine clusters

### Algorithms (in `scripts/algorithms/`)
- **maximum_spanning_forest.py**: Kruskal's algorithm on dual graph - optimal for edge maximization
- **greedy_region_growing.py**: Grows clusters from seeds, prioritizing faces with most shared edges
- **hierarchical_clustering.py**: Bottom-up merging based on edge connectivity

### Key Concepts
- **Dual Graph**: Faces become nodes, edges connect adjacent faces. Maximum spanning forest = minimum cuts.
- **Distortion Threshold**: Clusters are capped when estimated distortion exceeds threshold (default 0.3)
- **Edge Weights**: Dihedral angle (flat = high weight), edge length, curvature continuity

### Result Objects
All algorithms return a result object with:
- `clusters`: List[Set[int]] - face IDs per cluster
- `interior_edge_ratio`: float - ratio of preserved edges
- `cut_edges`: set of edges that become seams
