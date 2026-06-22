"""Cost-aware artwork identification support (decision D).

Local CLIP + Qdrant catalog for a free similarity cross-check that corrects the
VLM's title/date hallucinations. See docs/build/B1.1_identification_spike.md.
"""
from .artwork_catalog import ArtworkCatalog, CatalogMatch, parse_metadata_from_filename

__all__ = ["ArtworkCatalog", "CatalogMatch", "parse_metadata_from_filename"]
