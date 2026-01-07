# Polygon Unwrapping Scripts

Python implementation of edge-maximized polygon unwrapping algorithms.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo with visualization
python demo.py --mesh dog --visualize

# Compare all algorithms
python demo.py --mesh dog --algorithm all

# Run optimization pipeline
python demo.py --mesh dog --optimize
```

## Package Structure

```
scripts/
├── demo.py                 # Main demo script
├── visualize.py            # Visualization tools
├── requirements.txt        # Dependencies
│
├── algorithms/             # Clustering algorithms
│   ├── maximum_spanning_forest.py
│   ├── greedy_region_growing.py
│   └── hierarchical_clustering.py
│
├── utils/                  # Utility modules
│   ├── mesh.py             # Mesh data structures
│   ├── dual_graph.py       # Dual graph for clustering
│   ├── edge_weights.py     # Edge weight functions
│   ├── quality_metrics.py  # Quality evaluation
│   └── optimization.py     # Post-processing optimization
│
└── examples/               # Example meshes and scripts
```

## Algorithms

### 1. Maximum Spanning Forest (MSF)

Finds the maximum spanning forest on the dual graph, maximizing interior edges.

```python
from algorithms import maximum_spanning_forest
from utils.mesh import create_simple_dog

mesh = create_simple_dog()
result = maximum_spanning_forest(mesh, max_distortion=0.3)

print(f"Clusters: {len(result.clusters)}")
print(f"Interior edge ratio: {result.interior_edge_ratio:.1%}")
```

### 2. Greedy Region Growing

Grows clusters from seed faces, always adding the face with most shared edges.

```python
from algorithms import greedy_region_growing

result = greedy_region_growing(mesh, max_distortion=0.3)
```

### 3. Hierarchical Clustering

Bottom-up merging of clusters based on edge connectivity.

```python
from algorithms import hierarchical_clustering

result = hierarchical_clustering(mesh, max_distortion=0.3, target_clusters=5)
```

## Edge Weight Functions

```python
from utils.edge_weights import (
    dihedral_weight,          # Based on face angle
    edge_length_weight,       # Longer edges = higher weight
    curvature_continuity_weight,
    feature_edge_weight,      # Detect sharp edges
    visibility_weight,        # Hidden edges = lower weight
    pet_weight,               # Optimized for pet meshes
)

from utils.dual_graph import DualGraph

dual = DualGraph(mesh)

# Apply dihedral weights
from utils.edge_weights import apply_dihedral_weights
apply_dihedral_weights(dual)

# Or composite weights
from utils.edge_weights import apply_composite_weights, WeightConfig
config = WeightConfig(dihedral_weight=0.4, length_weight=0.2)
apply_composite_weights(dual, config)
```

## Quality Metrics

```python
from utils.quality_metrics import generate_quality_report, format_quality_report

report = generate_quality_report(mesh, clusters)
print(format_quality_report(report))
```

Output:
```
==================================================
POLYGON UNWRAPPING QUALITY REPORT
==================================================

--- EDGE CONNECTIVITY ---
Interior edge ratio:    87.5%
Total seam length:      4.32
Normalized seam length: 12.5%

--- CLUSTERS ---
Number of clusters:     5
Cluster sizes:          8 - 24
Balance score:          0.78

--- DISTORTION ---
Max distortion:         0.182
Mean distortion:        0.094

--- OVERALL ---
Quality score:          0.84
Grade:                  B
==================================================
```

## Optimization

```python
from utils.optimization import full_optimization_pipeline

optimized = full_optimization_pipeline(
    mesh, initial_clusters,
    min_cluster_size=3,
    max_distortion=0.3,
    use_annealing=True,
    verbose=True
)
```

## Visualization

```python
from visualize import MeshVisualizer

viz = MeshVisualizer(mesh, clusters)

# 3D view with cluster colors
fig, ax = viz.visualize_3d_matplotlib()

# 2D UV layout preview
fig, axes = viz.visualize_2d_layout()

# Quality dashboard
fig = viz.visualize_quality_dashboard()

# Save all visualizations
viz.save_all_visualizations("output_prefix")
```

## Creating Custom Meshes

```python
from utils.mesh import Mesh

mesh = Mesh()

# Add vertices
mesh.add_vertex(0, [0, 0, 0])
mesh.add_vertex(1, [1, 0, 0])
mesh.add_vertex(2, [0.5, 1, 0])

# Add faces (vertex IDs)
mesh.add_face(0, [0, 1, 2])

# Or load from OBJ
mesh = Mesh.from_obj("model.obj")

# Save to OBJ
mesh.to_obj("output.obj")
```

## Available Demo Meshes

- `cube`: Simple 6-face cube
- `pyramid`: 5-face pyramid
- `cylinder`: 8-segment cylinder with caps
- `dog`: Simple ~48-face dog
- `dog_detailed`: More detailed ~60-face dog

## Command Line Options

```
python demo.py --help

options:
  --mesh {cube,pyramid,cylinder,dog,dog_detailed}
  --algorithm {msf,greedy,hierarchical,all}
  --distortion DISTORTION   Maximum distortion threshold (default: 0.3)
  --visualize               Generate and save visualizations
  --optimize                Run optimization pipeline demo
  --output OUTPUT           Output prefix for visualization files
```

## Example Output

Running `python demo.py --mesh dog --algorithm all --visualize` produces:

1. Console output comparing all algorithms
2. `demo_3d.png` - 3D mesh with cluster coloring
3. `demo_2d_layout.png` - 2D UV layout preview
4. `demo_dashboard.png` - Quality metrics dashboard
5. `demo_comparison.png` - Side-by-side algorithm comparison
