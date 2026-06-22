"""Local artwork catalog — CLIP embeddings + Qdrant similarity.

Provides a cost-free identification cross-check (decision D): the VLM proposes
artist/title/date, and this catalog confirms or corrects the title/date by visual
similarity to known works. See docs/build/B1.1_identification_spike.md.

Dependencies: sentence-transformers (clip-ViT-B-32), qdrant-client, Pillow.
A running Qdrant instance is expected at QDRANT_URL (default http://localhost:6333).
"""
from __future__ import annotations

import glob
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "neural_canvas_artworks"
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"
EMBED_DIM = 512
DEFAULT_QDRANT_URL = "http://localhost:6333"
MATCH_THRESHOLD = 0.90  # cosine ≥ this ⇒ adopt the catalog's verified metadata
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp", ".gif")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


@dataclass
class CatalogMatch:
    score: float
    artist: str = ""
    title: str = ""
    date: str = ""
    source: str = ""
    image_path: str = ""


def parse_metadata_from_filename(path: str) -> dict:
    """Best-effort artist/title/date from a descriptive filename. Handles e.g.:
      'Emile Clause - Summer morning 1891.jpeg'        -> artist - title year
      'The Field of Derout-Lollichon (1886) Paul Gauguin.jpg'  -> title (year) artist
      'Jesus Ministered to by Angels, James Tissot 1886.jpg'   -> title, artist year
    Heuristic only; the VLM + catalog reconcile the truth at query time.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    ym = _YEAR_RE.search(stem)
    date = ym.group(1) if ym else ""
    s = _YEAR_RE.sub("", stem)
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s).strip(" ,-")

    artist, title = "", s
    if " - " in s:
        artist, title = (p.strip() for p in s.split(" - ", 1))
    elif "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            title, artist = ", ".join(parts[:-1]), parts[-1]
    return {"artist": artist, "title": title, "date": date, "source": "filename"}


class ArtworkCatalog:
    """CLIP+Qdrant artwork catalog for similarity-based identification cross-check."""

    def __init__(self, qdrant_url: Optional[str] = None,
                 collection: str = DEFAULT_COLLECTION,
                 clip_model: str = DEFAULT_CLIP_MODEL):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
        self.collection = collection
        self.clip_model_name = clip_model
        self._model = None
        self._client = None

    # ── lazy backends ────────────────────────────────────────────────────
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading CLIP model %s", self.clip_model_name)
            self._model = SentenceTransformer(self.clip_model_name)
        return self._model

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.qdrant_url)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        names = [c.name for c in self._client.get_collections().collections]
        if self.collection not in names:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %s (dim=%d, cosine)", self.collection, EMBED_DIM)

    # ── core ops ─────────────────────────────────────────────────────────
    def embed(self, image) -> list:
        """Embed an image path or PIL.Image into a normalized CLIP vector."""
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        vec = self.model.encode(image, normalize_embeddings=True)
        return vec.tolist()

    def add(self, image_path: str, metadata: Optional[dict] = None) -> str:
        from qdrant_client.models import PointStruct
        meta = dict(metadata or {})
        meta.setdefault("image_path", image_path)
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(image_path)))
        self.client.upsert(self.collection,
                           [PointStruct(id=pid, vector=self.embed(image_path), payload=meta)])
        return pid

    def build_from_dir(self, directory: str) -> int:
        """Seed the catalog from a directory of artwork images (filename metadata)."""
        count = 0
        for f in sorted(glob.glob(os.path.join(directory, "*"))):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                self.add(f, parse_metadata_from_filename(f))
                count += 1
        logger.info("Catalog built: %d artworks from %s", count, directory)
        return count

    def query(self, image, top_k: int = 5) -> list:
        vec = self.embed(image)
        try:  # qdrant-client >= 1.10
            pts = self.client.query_points(self.collection, query=vec, limit=top_k).points
        except AttributeError:  # older clients
            pts = self.client.search(self.collection, query_vector=vec, limit=top_k)
        out = []
        for r in pts:
            p = r.payload or {}
            out.append(CatalogMatch(score=float(r.score), artist=p.get("artist", ""),
                                    title=p.get("title", ""), date=p.get("date", ""),
                                    source=p.get("source", ""), image_path=p.get("image_path", "")))
        return out

    def cross_check(self, image, vlm_hypothesis: Optional[dict] = None,
                    threshold: float = MATCH_THRESHOLD) -> dict:
        """Reconcile a VLM identification hypothesis against the catalog (agree-or-flag).

        High-confidence catalog match ⇒ adopt its verified artist/title/date.
        Otherwise keep the VLM hypothesis but flag title/date as low-confidence.
        """
        result = dict(vlm_hypothesis or {})
        matches = self.query(image, top_k=3)
        top = matches[0] if matches else None
        if top and top.score >= threshold:
            result["artist"] = top.artist or result.get("artist", "")
            result["title"] = top.title or result.get("title", "")
            result["date"] = top.date or result.get("date", "")
            result["identification_source"] = f"catalog_match(score={top.score:.3f})"
            result["title_date_confidence"] = "high"
            result["needs_verification"] = False
        else:
            best = f"{top.score:.3f}" if top else "none"
            result["identification_source"] = f"vlm_hypothesis(best_catalog={best})"
            result["title_date_confidence"] = "low"  # VLM hallucinates exact title/date
            result["needs_verification"] = True
        result["catalog_matches"] = [vars(m) for m in matches]
        return result


def _main() -> None:
    import argparse, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="neural-canvas artwork catalog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("directory")
    q = sub.add_parser("query"); q.add_argument("image"); q.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qdrant-url", default=None)
    args = ap.parse_args()
    cat = ArtworkCatalog(qdrant_url=args.qdrant_url)
    if args.cmd == "build":
        print(f"Indexed {cat.build_from_dir(args.directory)} artworks into '{cat.collection}'.")
    elif args.cmd == "query":
        print(json.dumps([vars(m) for m in cat.query(args.image, top_k=args.top_k)], indent=2))


if __name__ == "__main__":
    _main()
