"""GeoSAE: geometry-guided sparse autoencoder for brain MRI foundation models."""

from geosae.manifold import build_knn_graph, KnnGraph
from geosae.models import TopKSparseAutoencoder

__all__ = ["TopKSparseAutoencoder", "build_knn_graph", "KnnGraph"]
__version__ = "0.1.0"
