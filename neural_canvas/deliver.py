"""Stage 6 — deliver the finished frame to the Google Photos album the TV's ambient mode shows.

Ported forward from art_agent_team/agents/delivery_agent.py, with two fixes:
* scopes: ``appendonly`` uploads and creates albums, but *listing* albums needs
  ``readonly.appcreateddata`` — the old code would 403 on its first album lookup;
* the album id is persisted, so a run never depends on listing at all.

Since 2025-03-31 the Library API can only touch media and albums the app itself created, so the app
creates "Neural Canvas" and the TV is pointed at that album once (Google TV → Ambient mode →
Google Photos → select album).

Never interactive inside the service: when the one-time OAuth consent has not been given, delivery
FAILS with ``AuthRequired`` naming the command that performs the consent, and the run stays open.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly",
          "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata"]
API = "https://photoslibrary.googleapis.com/v1"


class AuthRequired(RuntimeError):
    pass


@dataclass
class DeliveryResult:
    media_item_id: str
    album_id: str
    product_url: str
    passed: bool
    failures: list

    def to_dict(self) -> dict:
        return asdict(self)


class GooglePhotos:
    def __init__(self, cfg: dict, state_dir: Path):
        self.client_path = Path(os.path.expanduser(cfg["oauth_client_path"]))
        self.token_path = Path(os.path.expanduser(cfg["token_path"]))
        self.album_name = cfg["album_name"]
        self.album_file = state_dir / "gphotos-album.json"

    # one-time, interactive: `neural-canvas auth-photos`
    def authorize_interactive(self) -> None:
        from google_auth_oauthlib.flow import InstalledAppFlow
        if not self.client_path.exists():
            raise AuthRequired(self._client_missing())
        creds = InstalledAppFlow.from_client_secrets_file(str(self.client_path), SCOPES).run_local_server(port=0)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(creds.to_json())
        os.chmod(self.token_path, 0o600)

    def _client_missing(self) -> str:
        return (f"Google Photos OAuth client JSON not found at {self.client_path}. Create a Desktop OAuth "
                f"client in Google Cloud Console (APIs & Services → Credentials) with the Photos Library API "
                f"enabled, save its JSON there, then run `neural-canvas auth-photos` once.")

    def _session(self) -> requests.Session:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        if not self.client_path.exists():
            raise AuthRequired(self._client_missing())
        if not self.token_path.exists():
            raise AuthRequired("Google Photos consent not yet given: run `neural-canvas auth-photos` once.")
        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json())
            else:
                raise AuthRequired("Google Photos token invalid: re-run `neural-canvas auth-photos`.")
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {creds.token}"
        return s

    def _album_id(self, s: requests.Session) -> str:
        if self.album_file.exists():
            return json.loads(self.album_file.read_text())["id"]
        page = None
        while True:   # an app-created album from an earlier install
            r = s.get(f"{API}/albums", params={"pageSize": 50, **({"pageToken": page} if page else {})}, timeout=30)
            r.raise_for_status()
            d = r.json()
            for a in d.get("albums", []):
                if a.get("title") == self.album_name:
                    self.album_file.write_text(json.dumps({"id": a["id"], "title": self.album_name}))
                    return a["id"]
            page = d.get("nextPageToken")
            if not page:
                break
        r = s.post(f"{API}/albums", json={"album": {"title": self.album_name}}, timeout=30)
        r.raise_for_status()
        aid = r.json()["id"]
        self.album_file.write_text(json.dumps({"id": aid, "title": self.album_name}))
        log.info("created Google Photos album %r", self.album_name)
        return aid

    def deliver(self, image_path: Path, description: str) -> DeliveryResult:
        s = self._session()
        album = self._album_id(s)
        r = s.post(f"{API}/uploads", data=image_path.read_bytes(), timeout=180, headers={
            "Content-type": "application/octet-stream", "X-Goog-Upload-Content-Type": "image/jpeg",
            "X-Goog-Upload-Protocol": "raw", "X-Goog-Upload-File-Name": image_path.name})
        r.raise_for_status()
        token = r.text
        r = s.post(f"{API}/mediaItems:batchCreate", timeout=120, json={"albumId": album, "newMediaItems": [
            {"description": description[:1000], "simpleMediaItem": {"uploadToken": token, "fileName": image_path.name}}]})
        r.raise_for_status()
        res = (r.json().get("newMediaItemResults") or [{}])[0]
        item = res.get("mediaItem") or {}
        failures = [] if item.get("id") else [f"batchCreate status {res.get('status')}"]
        return DeliveryResult(item.get("id", ""), album, item.get("productUrl", ""), not failures, failures)
