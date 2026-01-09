# Building an Image Index

Research compiled by Claude - January 2026

## Overview

Building an image index involves creating searchable data structures that enable efficient retrieval of images based on their content, visual similarity, or metadata. This document covers the major approaches, tools, and best practices.

---

## Table of Contents

1. [Indexing Approaches](#indexing-approaches)
2. [Feature Extraction Methods](#feature-extraction-methods)
3. [Vector Databases & Similarity Search](#vector-databases--similarity-search)
4. [Perceptual Hashing for Deduplication](#perceptual-hashing-for-deduplication)
5. [Metadata Indexing](#metadata-indexing)
6. [Implementation Guide](#implementation-guide)
7. [Tools & Libraries](#tools--libraries)
8. [Performance Optimization](#performance-optimization)

---

## Indexing Approaches

### 1. Content-Based Image Retrieval (CBIR)

CBIR systems retrieve images based on visual properties:
- **Color**: Histograms, dominant colors, color moments
- **Texture**: Gabor filters, Local Binary Patterns (LBP)
- **Shape**: Edge detection, contour analysis
- **Deep features**: CNN embeddings capturing semantic content

### 2. Embedding-Based Search (Modern Approach)

The current state-of-the-art converts images into dense vector embeddings:
1. Pass image through a neural network (encoder)
2. Extract a fixed-dimensional vector (e.g., 512-d, 768-d)
3. Store vectors in an index structure
4. Query by computing distance/similarity between vectors

### 3. Perceptual Hashing

Fast fingerprinting for exact/near-duplicate detection:
- Generates compact binary hashes (64-256 bits)
- Similar images produce similar hashes
- Comparison via Hamming distance

### 4. Metadata Indexing

Index images by their EXIF, IPTC, or XMP metadata:
- Camera settings, timestamps, GPS coordinates
- Keywords, descriptions, copyright info
- Standard database indexing techniques apply

---

## Feature Extraction Methods

### Traditional Computer Vision

| Method | Description | Use Case |
|--------|-------------|----------|
| Color Histogram | Distribution of pixel intensities | Color-based search |
| HOG (Histogram of Oriented Gradients) | Edge orientation statistics | Object detection |
| SIFT/SURF | Scale-invariant keypoints | Feature matching |
| LBP (Local Binary Patterns) | Texture encoding | Texture classification |

### Deep Learning Approaches

#### Pre-trained CNN Backbones

```python
# Example: ResNet feature extraction
from torchvision import models, transforms
import torch

model = models.resnet50(pretrained=True)
# Remove classification layer, keep feature extractor
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    with torch.no_grad():
        input_tensor = preprocess(image).unsqueeze(0)
        features = model(input_tensor)
        return features.squeeze().numpy()  # 2048-d vector
```

#### CLIP (Contrastive Language-Image Pre-training)

CLIP is the current gold standard for image indexing because:
- Maps images and text to the **same embedding space**
- Enables text-to-image search ("find images of dogs playing")
- Pre-trained on 400M image-text pairs
- Produces 512-d or 768-d embeddings

```python
# Example: CLIP feature extraction
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().numpy()

def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.squeeze().numpy()
```

---

## Vector Databases & Similarity Search

### Index Structures

| Index Type | Characteristics | Best For |
|------------|-----------------|----------|
| **Flat (Brute Force)** | Exact search, O(n) | Small datasets (<100K) |
| **IVF (Inverted File)** | Clusters + search within | Medium datasets |
| **HNSW** | Graph-based, very fast | Real-time applications |
| **LSH** | Hash-based approximation | Very large datasets |
| **PQ (Product Quantization)** | Compressed vectors | Memory-constrained |

### Popular Vector Databases

1. **FAISS** (Facebook AI Similarity Search)
   - Open-source, highly optimized
   - CPU and GPU support
   - Best for local/self-hosted deployments

2. **Pinecone**
   - Fully managed cloud service
   - Real-time updates, filtering
   - Best for production without ops overhead

3. **Milvus**
   - Open-source, distributed
   - Supports multiple index types
   - Good for large-scale deployments

4. **Qdrant**
   - Open-source with cloud option
   - Rich filtering capabilities
   - Payload storage with vectors

5. **Weaviate**
   - GraphQL API
   - Built-in vectorization modules
   - Hybrid search (vector + keyword)

6. **Chroma**
   - Simple, lightweight
   - Great for prototyping
   - Python-native

### Similarity Metrics

- **Cosine Similarity**: Measures angle between vectors (most common)
- **Euclidean Distance (L2)**: Straight-line distance
- **Inner Product (Dot Product)**: For normalized vectors, equivalent to cosine
- **Hamming Distance**: For binary vectors/hashes

---

## Perceptual Hashing for Deduplication

### Hash Algorithms

| Algorithm | Description | Bits | Speed |
|-----------|-------------|------|-------|
| **aHash** (Average Hash) | Compare to mean pixel value | 64 | Fastest |
| **dHash** (Difference Hash) | Compare adjacent pixels | 64-128 | Fast |
| **pHash** (Perceptual Hash) | DCT-based frequency analysis | 64 | Medium |
| **wHash** (Wavelet Hash) | Wavelet transform | 64 | Medium |

### Implementation Example

```python
from imagededup.methods import PHash

# Initialize hasher
phasher = PHash()

# Generate hashes for a directory
encodings = phasher.encode_images(image_dir='path/to/images/')

# Find duplicates
duplicates = phasher.find_duplicates(
    encoding_map=encodings,
    max_distance_threshold=10  # Hamming distance threshold
)
```

### Indexing Strategies for Hashes

1. **BK-Tree**: Data structure optimized for Hamming distance queries
2. **Multi-Index Hashing**: Split hash into parts, index separately (scales to billions)
3. **LSH for Hamming**: Locality-sensitive hashing for approximate matching

---

## Metadata Indexing

### EXIF Schema (Key Fields)

```sql
CREATE TABLE image_metadata (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,

    -- Camera info
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,

    -- Capture settings
    aperture REAL,
    shutter_speed TEXT,
    iso INTEGER,
    focal_length REAL,

    -- Date/Time
    date_taken TIMESTAMP,

    -- Location
    gps_latitude REAL,
    gps_longitude REAL,
    gps_altitude REAL,

    -- Image properties
    width INTEGER,
    height INTEGER,
    orientation INTEGER,
    color_space TEXT,

    -- Descriptive (IPTC/XMP)
    title TEXT,
    description TEXT,
    keywords TEXT[],
    copyright TEXT,

    -- Indexes
    INDEX idx_date (date_taken),
    INDEX idx_location (gps_latitude, gps_longitude),
    INDEX idx_camera (camera_make, camera_model)
);
```

### Python Extraction

```python
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def extract_exif(image_path):
    image = Image.open(image_path)
    exif_data = {}

    if hasattr(image, '_getexif') and image._getexif():
        for tag_id, value in image._getexif().items():
            tag = TAGS.get(tag_id, tag_id)
            exif_data[tag] = value

    return exif_data
```

---

## Implementation Guide

### Complete Image Indexing Pipeline

```python
import os
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import json

class ImageIndex:
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
        self.image_paths = []

        # Load CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def _normalize(self, embeddings):
        """Normalize vectors for cosine similarity via inner product."""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _get_embedding(self, image):
        """Extract CLIP embedding from image."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.squeeze().numpy()

    def add_images(self, image_dir):
        """Index all images in a directory."""
        embeddings = []

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path = os.path.join(image_dir, filename)
                try:
                    image = Image.open(path).convert('RGB')
                    embedding = self._get_embedding(image)
                    embeddings.append(embedding)
                    self.image_paths.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

        if embeddings:
            embeddings = np.array(embeddings).astype('float32')
            embeddings = self._normalize(embeddings)
            self.index.add(embeddings)

        print(f"Indexed {len(self.image_paths)} images")

    def search_by_image(self, query_image, k=10):
        """Find similar images given a query image."""
        embedding = self._get_embedding(query_image)
        embedding = self._normalize(embedding.reshape(1, -1)).astype('float32')

        distances, indices = self.index.search(embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths):
                results.append({
                    'path': self.image_paths[idx],
                    'similarity': float(dist)
                })
        return results

    def search_by_text(self, query_text, k=10):
        """Find images matching a text description."""
        inputs = self.processor(text=query_text, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        embedding = embedding.squeeze().numpy()
        embedding = self._normalize(embedding.reshape(1, -1)).astype('float32')

        distances, indices = self.index.search(embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths):
                results.append({
                    'path': self.image_paths[idx],
                    'similarity': float(dist)
                })
        return results

    def save(self, path):
        """Save index to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.json", 'w') as f:
            json.dump(self.image_paths, f)

    def load(self, path):
        """Load index from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.json", 'r') as f:
            self.image_paths = json.load(f)

# Usage
if __name__ == "__main__":
    index = ImageIndex()
    index.add_images("/path/to/images")

    # Search by text
    results = index.search_by_text("a sunset over mountains")
    for r in results:
        print(f"{r['path']}: {r['similarity']:.3f}")

    # Search by image
    query = Image.open("/path/to/query.jpg")
    results = index.search_by_image(query)
```

---

## Tools & Libraries

### Embedding Generation

| Tool | Language | Notes |
|------|----------|-------|
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Python | CLIP, ViT, and more |
| [OpenAI CLIP](https://github.com/openai/CLIP) | Python | Original implementation |
| [sentence-transformers](https://www.sbert.net/) | Python | Easy CLIP access |
| [clip-retrieval](https://github.com/rom1504/clip-retrieval) | Python | End-to-end pipeline |

### Vector Search

| Tool | Type | Notes |
|------|------|-------|
| [FAISS](https://github.com/facebookresearch/faiss) | Library | Fast, GPU support |
| [Annoy](https://github.com/spotify/annoy) | Library | Memory-mapped, Spotify |
| [HNSWlib](https://github.com/nmslib/hnswlib) | Library | Pure C++ HNSW |
| [Pinecone](https://www.pinecone.io/) | Managed | Cloud vector DB |
| [Milvus](https://milvus.io/) | Self-hosted | Scalable vector DB |
| [Qdrant](https://qdrant.tech/) | Self-hosted/Cloud | Filtering support |

### Perceptual Hashing

| Tool | Language | Notes |
|------|----------|-------|
| [imagededup](https://github.com/idealo/imagededup) | Python | Multiple algorithms |
| [ImageHash](https://github.com/JohannesBuchner/imagehash) | Python | Simple API |
| [pHash](http://phash.org/) | C/C++ | Reference implementation |
| [imgdupes](https://github.com/knjcode/imgdupes) | CLI | Command-line tool |

---

## Performance Optimization

### Scaling Strategies

1. **Dimensionality Reduction**
   - Use PCA to reduce 768-d to 256-d
   - Minimal accuracy loss, significant speed gain

2. **Quantization**
   - Product Quantization (PQ) for memory efficiency
   - Scalar quantization for faster distance computation

3. **Hierarchical Indexing**
   - IVF with HNSW for large-scale
   - Coarse quantizer + fine search

4. **Batching**
   - Process images in batches for embedding generation
   - Use GPU batching where possible

5. **Sharding**
   - Distribute index across multiple machines
   - Query all shards in parallel, merge results

### Memory Estimates

| Images | Dimensions | Index Type | RAM Required |
|--------|------------|------------|--------------|
| 1M | 512 | Flat | ~2 GB |
| 1M | 512 | IVF4096 | ~2.5 GB |
| 1M | 512 | PQ64 | ~256 MB |
| 10M | 512 | IVF + PQ | ~3 GB |

### Hardware Recommendations

- **CPU**: Good for <1M images, IVF/HNSW indices
- **GPU**: Essential for >1M images or real-time requirements
- **RAM**: 2-4x the index size for efficient operation
- **SSD**: Required for memory-mapped indices

---

## Summary

| Use Case | Recommended Approach |
|----------|---------------------|
| **Semantic search** | CLIP embeddings + FAISS/Qdrant |
| **Exact duplicate detection** | dHash or pHash |
| **Near-duplicate detection** | pHash + BK-tree or imagededup |
| **Text-to-image search** | CLIP embeddings |
| **Large scale (>10M)** | Milvus or Pinecone |
| **Prototype/MVP** | Chroma or FAISS Flat |
| **Metadata search** | PostgreSQL with proper indexing |

---

## References

- [Content-based image retrieval - Wikipedia](https://en.wikipedia.org/wiki/Content-based_image_retrieval)
- [CLIP Image Search - Pinecone Tutorial](https://www.pinecone.io/learn/clip-image-search/)
- [Building an Image Similarity Search Engine with FAISS and CLIP - Towards Data Science](https://towardsdatascience.com/building-an-image-similarity-search-engine-with-faiss-and-clip-2211126d08fa/)
- [Hugging Face Cookbook: FAISS with CLIP](https://huggingface.co/learn/cookbook/faiss_with_hf_datasets_and_clip)
- [clip-retrieval - GitHub](https://github.com/rom1504/clip-retrieval)
- [imagededup - GitHub](https://github.com/idealo/imagededup)
- [FAISS Tutorial - Pinecone](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [Top Vector Databases 2025 - LakeFS](https://lakefs.io/blog/best-vector-databases/)
- [Canva Engineering: Perceptual Hashing at Scale](https://www.canva.dev/blog/engineering/simple-fast-and-scalable-reverse-image-search-using-perceptual-hashes-and-dynamodb/)
- [Image Embeddings for Search - Pinecone](https://www.pinecone.io/learn/series/image-search/)
