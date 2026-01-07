# Decision Flowchart: Choosing Your Unwrapping Strategy

**Author**: Claude (AI Assistant)
**Date**: January 7, 2026

---

## Master Decision Tree

```
START: What are you unwrapping?
│
├─► Is it a SIMPLE primitive? (cube, cylinder, sphere)
│   │
│   YES ──► Use PREDEFINED PATTERNS
│   │       • Cube: Cross/T-shape unwrap
│   │       • Cylinder: Rectangle + circles
│   │       • Sphere: 2-4 island projection
│   │
│   NO ▼
│
├─► Is it a LOW-POLY model? (< 500 faces)
│   │
│   │ YES ──► Go to LOW-POLY DECISION TREE ──────────────────┐
│   │                                                         │
│   NO ▼                                                      │
│                                                             │
├─► Is it a HIGH-POLY model? (> 5000 faces)                  │
│   │                                                         │
│   │ YES ──► Use HIERARCHICAL METHODS                       │
│   │         • Spectral clustering                          │
│   │         • Multi-resolution approach                    │
│   │                                                         │
│   NO ▼                                                      │
│                                                             │
└─► MEDIUM-POLY (500-5000 faces)                             │
    │                                                         │
    └─► Use ADAPTIVE METHODS                                  │
        • Maximum spanning forest                             │
        • Region growing with distortion limits               │
                                                              │
    ◄─────────────────────────────────────────────────────────┘
```

---

## Low-Poly Decision Tree (< 500 faces)

```
LOW-POLY MODEL
│
├─► What TYPE of model?
│   │
│   ├─► CHARACTER (humanoid, animal, pet)
│   │   │
│   │   └─► ANATOMICAL CLUSTERING
│   │       │
│   │       ├─► Poly count 50-100?
│   │       │   └─► 2-3 clusters (simplified)
│   │       │
│   │       ├─► Poly count 100-200?
│   │       │   └─► 4-6 clusters (body parts)
│   │       │
│   │       └─► Poly count 200-500?
│   │           └─► 5-8 clusters (detailed parts)
│   │
│   ├─► VEHICLE / MECHANICAL
│   │   │
│   │   └─► COMPONENT-BASED CLUSTERING
│   │       • Separate by logical parts
│   │       • Cut at joints/seams
│   │       • Respect hard edges
│   │
│   ├─► ENVIRONMENT (building, prop)
│   │   │
│   │   └─► PLANAR PROJECTION + CLEANUP
│   │       • Project by dominant face direction
│   │       • Merge co-planar faces
│   │       • Cut at material boundaries
│   │
│   └─► ORGANIC (plant, rock)
│       │
│       └─► GREEDY REGION GROWING
│           • Start from largest flat region
│           • Grow until distortion limit
│           • Allow higher distortion threshold
│
└─► What is the PRIMARY GOAL?
    │
    ├─► MAXIMUM EDGE CONNECTIVITY
    │   │
    │   └─► Use MAXIMUM SPANNING FOREST
    │       • Weight: dihedral angle only
    │       • Higher distortion tolerance (0.3-0.4)
    │       • Accept 1-2 large clusters
    │
    ├─► MINIMAL DISTORTION
    │   │
    │   └─► Use CURVATURE-BASED SEGMENTATION
    │       • Split at high curvature
    │       • Many small clusters OK
    │       • Low distortion tolerance (0.1-0.2)
    │
    ├─► HIDDEN SEAMS
    │   │
    │   └─► Use VISIBILITY-AWARE WEIGHTS
    │       • Strong penalty for visible edges
    │       • Accept distortion in hidden areas
    │
    └─► TEXTURE EFFICIENCY
        │
        └─► Use BALANCED CLUSTERING
            • Target similar-sized clusters
            • Optimize UV packing
            • Moderate on all metrics
```

---

## Algorithm Selection Matrix

### By Model Type

| Model Type | Recommended Algorithm | Cluster Target | Distortion Limit |
|------------|----------------------|----------------|------------------|
| Pet (100-200 poly) | Anatomical + MSF | 4-6 | 0.25 |
| Human (200-500 poly) | Anatomical + MSF | 6-10 | 0.20 |
| Vehicle | Component-based | By parts | 0.15 |
| Building | Planar projection | By face direction | 0.10 |
| Rock/Organic | Region growing | 2-4 | 0.40 |
| Weapon | Component-based | 3-6 | 0.20 |
| Tree | Feature-based | 3-5 | 0.30 |

### By Priority

| Priority | Algorithm | Weight Function | Settings |
|----------|-----------|-----------------|----------|
| Max edges | Maximum Spanning Forest | Dihedral only | High distortion tolerance |
| Min distortion | Curvature segmentation | Stretch potential | Low tolerance, many clusters |
| Hidden seams | Visibility-aware | Visibility + Dihedral | Camera-specific weights |
| Fast processing | Greedy region growing | Simple dihedral | Fixed cluster count |
| Best quality | Hierarchical + refinement | Composite | Iterative optimization |

---

## Step-by-Step Selection Process

### Step 1: Analyze Your Mesh

```python
def analyze_mesh(mesh):
    """First step: understand what you're working with."""

    analysis = {
        'face_count': len(mesh.faces),
        'edge_count': len(mesh.edges),
        'vertex_count': len(mesh.vertices),
        'genus': compute_genus(mesh),
        'is_manifold': mesh.is_manifold(),
        'has_boundaries': mesh.has_boundaries(),
        'bounding_box': mesh.bounding_box(),
        'aspect_ratio': compute_aspect_ratio(mesh),
    }

    # Curvature analysis
    curvatures = [mesh.get_vertex_curvature(v) for v in mesh.vertices]
    analysis['curvature_stats'] = {
        'min': min(curvatures),
        'max': max(curvatures),
        'mean': np.mean(curvatures),
        'std': np.std(curvatures),
    }

    # Connectivity
    analysis['avg_valence'] = np.mean([mesh.vertex_valence(v) for v in mesh.vertices])

    return analysis
```

### Step 2: Classify Model Type

```python
def classify_model(mesh, analysis):
    """Determine what type of model this is."""

    # Heuristics for classification
    aspect = analysis['aspect_ratio']
    curv_std = analysis['curvature_stats']['std']

    # Check for character-like proportions
    bbox = analysis['bounding_box']
    height = bbox[1][1] - bbox[0][1]
    width = bbox[1][0] - bbox[0][0]
    depth = bbox[1][2] - bbox[0][2]

    if height > 1.5 * max(width, depth):
        # Tall and narrow - likely character or tree
        if has_bilateral_symmetry(mesh):
            return 'character'
        else:
            return 'organic'

    if curv_std < 0.1:
        # Low curvature variance - likely architectural
        return 'environment'

    if count_disconnected_components(mesh) > 1:
        return 'mechanical'

    return 'organic'  # Default
```

### Step 3: Select Algorithm

```python
def select_algorithm(model_type, face_count, priority='balanced'):
    """Choose the best algorithm based on analysis."""

    # Algorithm lookup table
    algorithms = {
        ('character', 'max_edges'): 'anatomical_msf',
        ('character', 'min_distortion'): 'anatomical_curvature',
        ('character', 'balanced'): 'anatomical_composite',

        ('organic', 'max_edges'): 'region_growing_greedy',
        ('organic', 'min_distortion'): 'curvature_segmentation',
        ('organic', 'balanced'): 'hierarchical_clustering',

        ('environment', 'max_edges'): 'planar_projection',
        ('environment', 'min_distortion'): 'planar_projection',
        ('environment', 'balanced'): 'planar_projection',

        ('mechanical', 'max_edges'): 'component_based',
        ('mechanical', 'min_distortion'): 'component_based',
        ('mechanical', 'balanced'): 'component_based',
    }

    key = (model_type, priority)
    return algorithms.get(key, 'hierarchical_clustering')
```

### Step 4: Configure Parameters

```python
def get_algorithm_params(algorithm, face_count, model_type):
    """Get recommended parameters for the algorithm."""

    # Base configurations
    configs = {
        'anatomical_msf': {
            'max_distortion': 0.25,
            'min_cluster_size': max(3, face_count // 30),
            'edge_weights': ['dihedral', 'joint', 'visibility'],
        },
        'anatomical_curvature': {
            'max_distortion': 0.15,
            'min_cluster_size': max(3, face_count // 50),
            'edge_weights': ['curvature', 'dihedral'],
        },
        'region_growing_greedy': {
            'max_distortion': 0.35,
            'min_cluster_size': max(5, face_count // 20),
            'edge_weights': ['dihedral'],
        },
        'hierarchical_clustering': {
            'max_distortion': 0.25,
            'target_clusters': estimate_cluster_count(face_count, model_type),
            'edge_weights': ['composite'],
        },
        'planar_projection': {
            'angle_threshold': 30.0,  # degrees
            'merge_coplanar': True,
        },
    }

    return configs.get(algorithm, {})


def estimate_cluster_count(face_count, model_type):
    """Estimate good cluster count."""

    # Rules of thumb
    if model_type == 'character':
        return max(4, min(10, face_count // 30))
    elif model_type == 'organic':
        return max(2, min(6, face_count // 50))
    elif model_type == 'environment':
        return max(3, min(8, face_count // 40))
    else:
        return max(3, face_count // 35)
```

---

## Quick Reference Flowcharts

### For 100-200 Poly Pets

```
PET MODEL (100-200 poly)
│
├─► Is the pet SYMMETRIC?
│   │
│   YES ──► Consider MIRROR UNWRAPPING
│   │       • Unwrap half, mirror UVs
│   │       • Saves texture space
│   │
│   NO ▼
│
├─► Does it have DISTINCT PARTS?
│   (head, body, legs, tail clearly separated)
│   │
│   YES ──► ANATOMICAL CLUSTERING
│   │       ├─► 4 clusters: head, body, front legs, back legs
│   │       └─► Tail attached to body or separate
│   │
│   NO ▼
│
├─► Is it a SIMPLE SHAPE?
│   (blob-like, few details)
│   │
│   YES ──► GREEDY REGION GROWING
│   │       ├─► 2-3 clusters
│   │       └─► Higher distortion OK
│   │
│   NO ▼
│
└─► Use MAXIMUM SPANNING FOREST
    with pet-specific weights
    ├─► Cut at joints
    ├─► Hidden seams on belly
    └─► 4-6 clusters typical
```

### For Texture Quality vs. Seam Visibility

```
PRIORITY QUESTION
│
├─► Is TEXTURE QUALITY more important?
│   (close-up viewing, high-res textures)
│   │
│   YES ──► MINIMIZE DISTORTION
│   │       ├─► More clusters (6-10)
│   │       ├─► Tight distortion limit (0.15)
│   │       └─► Accept more seams
│   │
│   NO ▼
│
├─► Is SEAM HIDING more important?
│   (game asset, multiple view angles)
│   │
│   YES ──► MAXIMIZE EDGE CONNECTIVITY
│   │       ├─► Fewer clusters (2-4)
│   │       ├─► Relaxed distortion (0.35)
│   │       └─► Visibility-weighted cuts
│   │
│   NO ▼
│
└─► BALANCED APPROACH
    ├─► 4-6 clusters
    ├─► Moderate distortion (0.25)
    └─► Composite edge weights
```

---

## Decision Examples

### Example 1: 150-Poly Cat

```
Input: Low-poly cat model, 150 faces
Use case: Mobile game pet

Analysis:
├─► Face count: 150 (low-poly)
├─► Type: Character (pet)
├─► Symmetry: Yes (bilateral)
├─► Priority: Hidden seams (game asset)

Decision:
├─► Algorithm: Anatomical clustering + MSF
├─► Clusters: 5 (head, body, 4 legs grouped as 2)
├─► Distortion limit: 0.25
├─► Seams: Along belly, inner legs, behind ears

Expected result:
├─► Interior edges: ~87%
├─► Max stretch: 1.15
└─► Quality grade: B+
```

### Example 2: 200-Poly Dog

```
Input: Stylized dog, 200 faces
Use case: Art print (high texture quality needed)

Analysis:
├─► Face count: 200 (low-poly)
├─► Type: Character (pet)
├─► Priority: Texture quality

Decision:
├─► Algorithm: Curvature-based segmentation
├─► Clusters: 7-8 (more islands for less distortion)
├─► Distortion limit: 0.15
├─► Seams: At all natural boundaries

Expected result:
├─► Interior edges: ~82%
├─► Max stretch: 1.08
└─► Quality grade: A- (texture) / B (seams)
```

### Example 3: 100-Poly Bird

```
Input: Simple bird, 100 faces
Use case: Background element

Analysis:
├─► Face count: 100 (very low-poly)
├─► Type: Character (organic)
├─► Priority: Fast/simple

Decision:
├─► Algorithm: Greedy region growing
├─► Clusters: 3 (body+head, 2 wings)
├─► Distortion limit: 0.35
├─► Seams: Wing-body junction

Expected result:
├─► Interior edges: ~90%
├─► Max stretch: 1.25
└─► Quality grade: B
```

---

## Common Decision Mistakes

### Mistake 1: Over-segmenting Low-Poly Models

```
WRONG: 150-poly model → 15 clusters
       Too many seams, inefficient UV space

RIGHT: 150-poly model → 4-6 clusters
       Balanced seams and distortion
```

### Mistake 2: Using Same Settings for All Models

```
WRONG: Same distortion limit (0.2) for everything

RIGHT:
├─► Organic models: 0.30-0.40
├─► Characters: 0.20-0.30
├─► Mechanical: 0.10-0.20
└─► Architectural: 0.05-0.15
```

### Mistake 3: Ignoring Model Symmetry

```
WRONG: Unwrap asymmetrically, waste 2x texture space

RIGHT: Detect symmetry → Mirror UVs → 2x texture efficiency
```

### Mistake 4: Wrong Priority Selection

```
WRONG: Real-time game asset → Minimize distortion
       (Results in too many visible seams)

RIGHT: Real-time game asset → Hidden seams
       (Accept some distortion for cleaner appearance)
```
