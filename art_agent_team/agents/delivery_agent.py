"""DeliveryAgent — push finished artworks to a Google Photos album.

The terminal pipeline stage. Uses the **photoslibrary.appendonly** scope (the broad
`photoslibrary` scope returns 403 PERMISSION_DENIED since 2025-04-01), writes to an
**app-owned album**, and creates media in **≤50-item batches, serially per account**
(per the Google Photos API constraints; see docs/reconstruction/05_comparative_research.md §5).

OAuth client setup + first user consent is a one-time human step (joint step E):
set `google_photos.oauth_client_path` in config to an OAuth 2.0 *client* JSON. Until
that exists, `available()` is False and the pipeline skips delivery gracefully.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly"]
UPLOAD_URL = "https://photoslibrary.googleapis.com/v1/uploads"
BATCH_CREATE_URL = "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate"
ALBUMS_URL = "https://photoslibrary.googleapis.com/v1/albums"
MAX_BATCH = 50  # Google Photos batchCreate hard limit


class DeliveryAgent:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        gp = (self.config.get("google_photos") or {}) if isinstance(self.config, dict) else {}
        self.oauth_client_path = gp.get("oauth_client_path") or ""
        self.album_name = gp.get("album_name", "neural-canvas")
        self.token_path = gp.get("token_path",
                                 os.path.join("art_agent_team", "config", ".gphotos_token.json"))
        self._creds = None
        self._album_id = None

    # ── availability ─────────────────────────────────────────────────────
    def available(self) -> bool:
        """True only if an OAuth client JSON is configured (step E done)."""
        return bool(self.oauth_client_path and os.path.exists(self.oauth_client_path))

    # ── auth ─────────────────────────────────────────────────────────────
    def _credentials(self):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(self.oauth_client_path, SCOPES)
            creds = flow.run_local_server(port=0)  # first-run consent = joint step E
        os.makedirs(os.path.dirname(self.token_path) or ".", exist_ok=True)
        with open(self.token_path, "w") as f:
            f.write(creds.to_json())
        return creds

    def _session(self):
        import requests
        creds = self._credentials()
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {creds.token}"})
        return s

    # ── album ────────────────────────────────────────────────────────────
    def _ensure_album(self, session) -> str:
        if self._album_id:
            return self._album_id
        page = None
        while True:
            params = {"pageSize": 50, "excludeNonAppCreatedData": True}
            if page:
                params["pageToken"] = page
            r = session.get(ALBUMS_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for a in data.get("albums", []):
                if a.get("title") == self.album_name:
                    self._album_id = a["id"]
                    return self._album_id
            page = data.get("nextPageToken")
            if not page:
                break
        r = session.post(ALBUMS_URL, json={"album": {"title": self.album_name}}, timeout=30)
        r.raise_for_status()
        self._album_id = r.json()["id"]
        logger.info("Created Google Photos album '%s' (%s)", self.album_name, self._album_id)
        return self._album_id

    # ── upload ───────────────────────────────────────────────────────────
    def _upload_bytes(self, session, image_path: str) -> Optional[str]:
        with open(image_path, "rb") as f:
            data = f.read()
        headers = {
            "Content-type": "application/octet-stream",
            "X-Goog-Upload-Content-Type": "image/jpeg",
            "X-Goog-Upload-Protocol": "raw",
            "X-Goog-Upload-File-Name": os.path.basename(image_path),
        }
        r = session.post(UPLOAD_URL, data=data, headers=headers, timeout=120)
        if r.status_code != 200 or not r.text:
            logger.warning("Upload failed for %s: %s %s", image_path, r.status_code, r.text[:200])
            return None
        return r.text  # upload token (valid ~1 day)

    # ── public API ───────────────────────────────────────────────────────
    def deliver(self, image_paths, descriptions: Optional[dict] = None) -> dict:
        """Upload finished images into the app-owned album. Returns a summary dict.
        Idempotency note: Google Photos has no dedup; callers should track delivered
        files to avoid re-uploading (the watcher/DocentAgent owns that)."""
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        descriptions = descriptions or {}
        if not self.available():
            logger.info("DeliveryAgent: no OAuth client configured — skipping delivery (step E pending).")
            return {"delivered": [], "failed": [], "skipped": "no_oauth_client", "album_id": None}

        delivered, failed = [], []
        try:
            session = self._session()
            album_id = self._ensure_album(session)
        except Exception as e:
            logger.error("DeliveryAgent: auth/album setup failed: %s", e)
            return {"delivered": [], "failed": list(image_paths), "skipped": f"auth_error:{e}", "album_id": None}

        # Build (path, token) pairs, then batchCreate in <=50 chunks, serially.
        items = []
        for p in image_paths:
            tok = self._upload_bytes(session, p)
            (items.append((p, tok)) if tok else failed.append(p))

        for i in range(0, len(items), MAX_BATCH):
            chunk = items[i:i + MAX_BATCH]
            new_media = [{
                "description": descriptions.get(p, "")[:1000],
                "simpleMediaItem": {"uploadToken": tok, "fileName": os.path.basename(p)},
            } for p, tok in chunk]
            try:
                r = session.post(BATCH_CREATE_URL,
                                 json={"albumId": album_id, "newMediaItems": new_media}, timeout=120)
                r.raise_for_status()  # 200 OK or 207 MULTI-STATUS (partial)
                results = r.json().get("newMediaItemResults", [])
                for (p, _), res in zip(chunk, results):
                    (delivered.append(p) if res.get("mediaItem") else failed.append(p))
            except Exception as e:
                logger.error("batchCreate failed for chunk: %s", e)
                failed.extend(p for p, _ in chunk)

        logger.info("DeliveryAgent: delivered %d, failed %d to album '%s'",
                    len(delivered), len(failed), self.album_name)
        return {"delivered": delivered, "failed": failed, "skipped": None, "album_id": album_id}
