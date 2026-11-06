"""Stage 6 — deliver the finished frame to the Google Photos album the TV's ambient mode shows.

Ported forward from art_agent_team/agents/delivery_agent.py, with two fixes:
* scopes: ``appendonly`` uploads and creates albums, but *listing* albums needs
  ``readonly.appcreateddata`` — the old code would 403 on its first album lookup;
* the album id is persisted, so a run never depends on listing at all.

Since 2025-03-31 the Library API can only touch media and albums the app itself created, so the app
creates its own album and the TV is pointed at that album once (Google TV → Ambient mode → Google
Photos → select album). The album keeps the title in config (``google_photos.album_name``, e.g. "."
so the screensaver overlay shows no name): a title change renames it, which needs
``edit.appcreateddata``.

Never interactive inside the service: when the one-time OAuth consent has not been given, delivery
FAILS with ``AuthRequired`` naming the command that performs the consent, and the run stays open.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly",
          "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata",
          "https://www.googleapis.com/auth/photoslibrary.edit.appcreateddata"]
REAUTH = "re-run `neural-canvas auth-photos`"
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

    _TOKEN = threading.Lock()          # parallel workers share one token file; refresh it one at a time

    def _session(self) -> requests.Session:
        with self._TOKEN:
            return self._session_locked()

    def _session_locked(self) -> requests.Session:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        if not self.client_path.exists():
            raise AuthRequired(self._client_missing())
        if not self.token_path.exists():
            raise AuthRequired("Google Photos consent not yet given: run `neural-canvas auth-photos` once.")
        from google.auth.exceptions import RefreshError
        granted = set(json.loads(self.token_path.read_text()).get("scopes") or [])
        missing = [s.rsplit("/", 1)[-1] for s in SCOPES if s not in granted]
        if missing:
            raise AuthRequired(f"Google Photos permission not granted ({', '.join(missing)}): {REAUTH}.")
        creds = Credentials.from_authorized_user_file(str(self.token_path), sorted(granted))
        if not creds.valid:
            if not (creds.expired and creds.refresh_token):
                raise AuthRequired(f"Google Photos token invalid: {REAUTH}.")
            try:
                creds.refresh(Request())
            except RefreshError as e:   # an app in Testing loses its refresh token after 7 days
                raise AuthRequired(f"Google Photos sign-in expired or revoked ({str(e)[:80]}): {REAUTH}.") from None
            self.token_path.write_text(creds.to_json())
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {creds.token}"
        return s

    def check(self) -> str:
        """'' when delivery can authenticate right now, else the reason (the runner pauses on it)."""
        try:
            self._session()
            return ""
        except AuthRequired as e:
            return str(e)

    def _album_id(self, s: requests.Session) -> str:
        if self.album_file.exists():
            cached = json.loads(self.album_file.read_text())
            if cached.get("title") != self.album_name:      # config renamed the album: follow it
                r = s.patch(f"{API}/albums/{cached['id']}", params={"updateMask": "title"},
                            json={"title": self.album_name}, timeout=30)
                if r.status_code == 403:
                    raise AuthRequired(f"renaming the album to {self.album_name!r} needs edit permission: {REAUTH}.")
                r.raise_for_status()
                log.info("renamed Google Photos album %r -> %r", cached.get("title"), self.album_name)
                self.album_file.write_text(json.dumps({"id": cached["id"], "title": self.album_name}))
            return cached["id"]
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

    def remove_from_album(self, media_item_id: str) -> None:
        """Take an app-created item out of the album (it stays in the library: the API cannot delete)."""
        s = self._session()
        r = s.post(f"{API}/albums/{self._album_id(s)}:batchRemoveMediaItems",
                   json={"mediaItemIds": [media_item_id]}, timeout=30)
        r.raise_for_status()

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
