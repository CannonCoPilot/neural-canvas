"""The Lens slot: one Google Lens-type reverse-image search per work, from a rotating pool of free credits.

Sir's rules (2026-11-06):

* nothing ever spills into paid. Every provider stops at its free allowance, and a provider under 10% of
  that allowance is skipped (``FLOOR``), so a pool is never run to its last credit;
* each work goes to the provider with the largest share of its allowance left, which spreads the load.
  Pools that renew monthly go first, since what they leave unused is lost at the reset; one-time pools
  (Serper, SearchAPI, Oxylabs) are the reserve;
* balances are counted locally between checks. Once only one provider is left above the floor, every
  provider's real balance is re-read before each pick, so a monthly reset is caught the moment it lands.
  A provider with no balance endpoint (Cloud Vision) keeps a local monthly count;
* an error, a redirect or an empty reply is a failure of that provider, never "no match": it is recorded and
  the next provider is tried. When every provider is below the floor the slot is UNAVAILABLE and alerts;
  it never quietly drops Lens and it never pays.

Providers register a search function and a balance function. Adding one (BrightData, Apify, ...) is one
``Provider`` entry in ``LensPool.default_providers``.
"""
from __future__ import annotations

import contextlib
import fcntl
import io
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests
from PIL import Image

from .websearch import WebReference

log = logging.getLogger(__name__)

FLOOR = 0.10          # a provider below this share of its allowance is skipped (Sir, 2026-11-06)
STALE_S = 24 * 3600   # a live balance older than this is re-read before use


class LensSlotUnavailable(RuntimeError):
    """Every provider is below the floor or failed: the slot is unavailable, which alerts."""


@dataclass
class Provider:
    name: str
    search: Callable[["Work"], list[WebReference]]
    allowance: int                       # free credits per period
    renews: str                          # "monthly" | "one-time"
    cost: int = 1                        # credits one search spends
    balance: Callable[[], int] | None = None   # live remaining credits; None = counted locally per month
    enabled: Callable[[], bool] = lambda: True
    charges_itself: bool = False         # the provider's client keeps the ledger (Vision: cached replies are free)

    @property
    def floor(self) -> int:
        return math.ceil(FLOOR * self.allowance)


@dataclass
class Work:
    """One picture as the providers need it: bytes for uploads, a public URL made at most once."""
    img: Image.Image
    url_fn: Callable[[Image.Image], str]
    _url: str = ""
    _t: float = 0.0
    tried: list[str] = field(default_factory=list)

    def url(self) -> str:
        if not self._url or time.time() - self._t > 240:     # tmpfiles' signed link lives about 5 min
            self._url, self._t = self.url_fn(self.img), time.time()
        return self._url

    def jpeg(self, max_bytes: int = 0, side: int = 1600) -> bytes:
        im, q = self.img.convert("RGB"), 88
        while True:
            im2 = im.copy()
            im2.thumbnail((side, side))
            b = io.BytesIO()
            im2.save(b, "JPEG", quality=q)
            if not max_bytes or b.tell() < max_bytes or side <= 500:
                return b.getvalue()
            side, q = int(side * 0.8), max(70, q - 4)


class Ledger:
    """Credits per provider, on disk (var/state/lens_pool.json). Every read and write goes through the file
    under an exclusive lock: the service's workers, the benchmark and the indexer may share one pool."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self.path.with_suffix(".lock"), "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, d: dict):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
        tmp.replace(self.path)

    def get(self, name: str) -> dict:
        with self._locked():
            return dict(self._load().get(name, {}))

    def update(self, name: str, **kv):
        with self._locked():
            d = self._load()
            d.setdefault(name, {}).update(kv)
            self._save(d)

    @staticmethod
    def _left(p: "Provider", e: dict) -> int:
        if p.balance is None:                               # counted locally, reset each month
            return max(0, p.allowance - (e.get("used_month", 0) if e.get("month") == time.strftime("%Y-%m") else 0))
        return int(e.get("remaining", 0))

    def reserve(self, p: "Provider") -> bool:
        """Charge one search if it keeps the provider at or above its floor; the check and the charge are one
        locked step, so two workers can never both take the last credit above the floor."""
        with self._locked():
            d = self._load()
            e = d.setdefault(p.name, {})
            if self._left(p, e) - p.cost < p.floor:
                return False
            month = time.strftime("%Y-%m")
            if e.get("month") != month:                     # a new month: local counts restart
                e["month"], e["used_month"] = month, 0
            e["used_month"] = e.get("used_month", 0) + p.cost
            if "remaining" in e:
                e["remaining"] = max(0, e["remaining"] - p.cost)
            e["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save(d)
            return True


class LensPool:
    def __init__(self, providers: list[Provider], ledger: Ledger, url_fn: Callable[[Image.Image], str]):
        self.providers = providers
        self.ledger = ledger
        self.url_fn = url_fn
        self.alert: Callable[[str, dict], None] | None = None     # set by the service: a Pulse alert
        self.extras: dict[str, tuple[str, Callable[[Work], list[WebReference]]]] = {}

    def extra(self, name: str, img: Image.Image) -> list[WebReference]:
        """A rescue search that spends one pool's credits (Yandex through SearchAPI, Lens exact matches through
        SerpAPI): the same ledger, the same 10% floor. An empty reply is a failure, never "no match"."""
        prov_name, fn = self.extras[name]
        p = next(p for p in self.providers if p.name == prov_name)
        if not p.enabled():
            raise RuntimeError(f"{name}: no {prov_name} key")
        self.refresh(p)
        if not self.ledger.reserve(p):
            raise RuntimeError(f"{name}: {prov_name} is at its 10% floor")
        refs = fn(Work(img, self.url_fn))
        if not refs:
            raise RuntimeError(f"{name}: empty reply")
        return refs

    # ── balances ──────────────────────────────────────────────────────────
    def remaining(self, p: Provider) -> int:
        return Ledger._left(p, self.ledger.get(p.name))

    def refresh(self, p: Provider, force: bool = False):
        """Re-read a provider's real balance (its account endpoint costs no credit)."""
        if p.balance is None:
            return
        e = self.ledger.get(p.name)
        if not force and "remaining" in e and time.time() - e.get("checked", 0) < STALE_S:
            return
        try:
            self.ledger.update(p.name, remaining=int(p.balance()), checked=time.time(), balance_error="")
        except Exception as ex:          # an unreadable balance is not a full one: the provider sits out
            log.warning("lens pool: %s balance unreadable: %s", p.name, _scrub(str(ex)))
            self.ledger.update(p.name, remaining=0, checked=time.time(), balance_error=_scrub(str(ex))[:200])

    def above_floor(self, p: Provider) -> bool:
        return self.remaining(p) - p.cost >= p.floor

    def status(self) -> list[dict]:
        return [{"provider": p.name, "remaining": self.remaining(p), "allowance": p.allowance, "floor": p.floor,
                 "renews": p.renews, "eligible": p.enabled() and self.above_floor(p)} for p in self.providers]

    # ── picking ───────────────────────────────────────────────────────────
    def pick(self, exclude: set[str] | frozenset[str] = frozenset()) -> Provider | None:
        live = [p for p in self.providers if p.enabled() and p.name not in exclude]
        for p in live:
            self.refresh(p)
        ok = [p for p in live if self.above_floor(p)]
        if len(ok) <= 1:      # only one left above the floor: real balances before every pick (Sir)
            for p in live:
                self.refresh(p, force=True)
            ok = [p for p in live if self.above_floor(p)]
        if not ok:
            return None
        # monthly pools first: their unused credits are lost at the reset; one-time pools are the reserve
        return max(ok, key=lambda p: (p.renews == "monthly", self.remaining(p) / p.allowance, p.name))

    def search(self, img: Image.Image, exclude: set[str] | frozenset[str] = frozenset(), work: Work | None = None
               ) -> tuple[str, list[WebReference], list[str]]:
        """(provider, references, errors). Tries providers in pick order until one answers with results;
        raises LensSlotUnavailable when none can."""
        work = work or Work(img, self.url_fn)
        errors: list[str] = []
        tried = set(exclude)
        while True:
            p = self.pick(tried)
            if p is None:
                why = "; ".join(errors) or "every provider is below its 10% floor"
                msg = f"Lens slot unavailable: {why}"
                log.error(msg)
                if self.alert:
                    try:
                        self.alert(msg, {"pool": self.status()})
                    except Exception as e:
                        log.warning("lens pool alert failed: %s", e)
                raise LensSlotUnavailable(msg)
            tried.add(p.name)
            work.tried.append(p.name)
            t = time.time()
            try:
                # counted before the call (a timeout may still have been billed), in one step with the floor check
                if not p.charges_itself and not self.ledger.reserve(p):
                    errors.append(f"{p.name}: reached its 10% floor")
                    continue
                refs = p.search(work)
            except Exception as e:
                errors.append(f"{p.name}: {_scrub(str(e))[:160]}")
                log.warning("lens pool: %s failed: %s", p.name, _scrub(str(e))[:200])
                continue
            if not refs:                     # an empty reply is a failure, never "no match"
                errors.append(f"{p.name}: empty reply")
                log.warning("lens pool: %s returned nothing", p.name)
                continue
            log.info("lens pool: %s answered %d references in %.1f s (%d left)", p.name, len(refs),
                     time.time() - t, self.remaining(p))
            return p.name, refs, errors


# ── providers ──────────────────────────────────────────────────────────────────

_KEYS: set[str] = set()       # every key value handed to a provider, scrubbed from any error text


def _scrub(text: str) -> str:
    for k in _KEYS:
        text = text.replace(k, "***")
    return re.sub(r"(api_key|token|key)=[^&\s'\"]+", r"\1=***", text)


def _req(method: str, url: str, **kw) -> requests.Response:
    """requests.request with every failure re-raised scrubbed: a connection error quotes the full URL, and
    SerpAPI takes its key only in the query string."""
    try:
        return requests.request(method, url, **kw)
    except requests.RequestException as e:
        raise RuntimeError(_scrub(f"{type(e).__name__}: {e}")) from None


def _usable(u) -> bool:
    return isinstance(u, str) and (u.startswith("http") or u.startswith("data:image"))


def _ref(title: str, link: str, image: str, thumb: str, source: str, site: str = "") -> WebReference | None:
    img = image if _usable(image) else thumb
    if not _usable(img):
        return None
    return WebReference(title or "", hash(img[:4096]) & 0x7FFFFFFFFFFF, thumb if _usable(thumb) else img,
                        link or "", img, {"source": source, "site": site})


def _http_json(r: requests.Response, key: str, name: str) -> dict:
    try:
        d = r.json()
    except ValueError:
        d = {}
    if r.status_code != 200 or (isinstance(d, dict) and d.get("error") and "hasn't returned any results" not in str(d["error"])):
        body = (json.dumps(d) if d else r.text)[:240]
        raise RuntimeError(_scrub(f"{name} HTTP {r.status_code}: {body.replace(key, '***') if key else body}"))
    return d


def serpapi_exact(key: Callable[[], str]) -> Callable[[Work], list[WebReference]]:
    """Lens exact matches: thumbnails only (verification is weaker), but the pages' titles are the best
    provenance leads (Sir: rescue only)."""
    def fn(w: Work) -> list[WebReference]:
        k = key()
        up = _http_json(_req("POST", "https://serpapi.com/image", data={"api_key": k},
                             files={"image": ("work.jpg", w.jpeg(490_000), "image/jpeg")}, timeout=90), k, "serpapi upload")
        d = _http_json(_req("GET", "https://serpapi.com/search.json", timeout=90, params={
            "engine": "google_lens", "type": "exact_matches", "hl": "en", "country": "us", "image_id": up.get("image_id", ""),
            "api_key": k}), k, "serpapi lens exact")
        return [r for x in d.get("exact_matches") or []
                if (r := _ref(x.get("title", ""), x.get("link", ""), "", x.get("thumbnail", ""), "lens_exact:serpapi",
                              x.get("source", "")))]
    return fn


def searchapi_yandex(key: Callable[[], str]) -> Callable[[Work], list[WebReference]]:
    """Yandex reverse image through SearchAPI: the same engine as the free direct arm, a paid route (rescue)."""
    def fn(w: Work) -> list[WebReference]:
        k = key()
        d = _http_json(_req("GET", "https://www.searchapi.io/api/v1/search", timeout=120,
                            params={"engine": "yandex_reverse_image", "url": w.url()},
                            headers={"Authorization": "Bearer " + k}), k, "searchapi yandex")
        out = _visual_rows(d.get("visual_matches"), "yandex:searchapi")
        for size_list in (d.get("image_sizes") or {}).values():     # other copies of the same picture
            out += _visual_rows([{"image": x.get("link", ""), "link": ""} for x in size_list or []], "yandex:searchapi")
        return out
    return fn


def serpapi_provider(key: Callable[[], str]) -> Provider:
    def search(w: Work) -> list[WebReference]:
        k = key()
        # the JPEG goes to SerpAPI's own image store (free, 500 KB max, the id lives 10 min): no public host
        up = _http_json(_req("POST", "https://serpapi.com/image", data={"api_key": k},
                                      files={"image": ("work.jpg", w.jpeg(490_000), "image/jpeg")}, timeout=90), k,
                        "serpapi upload")
        if not up.get("image_id"):
            raise RuntimeError("serpapi upload returned no image_id")
        d = _http_json(_req("GET", "https://serpapi.com/search.json", timeout=90, params={
            "engine": "google_lens", "type": "all", "hl": "en", "country": "us", "image_id": up["image_id"], "api_key": k}),
            k, "serpapi lens")
        return [r for x in d.get("visual_matches") or []
                if (r := _ref(x.get("title", ""), x.get("link", ""), x.get("image", ""), x.get("thumbnail", ""),
                              "google_lens:serpapi", x.get("source", "")))]

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://serpapi.com/account.json", params={"api_key": k}, timeout=20), k, "serpapi account")
        return int(d.get("total_searches_left") or 0)
    return Provider("serpapi", search, allowance=250, renews="monthly", balance=balance, enabled=lambda: bool(key()))


def searchapi_provider(key: Callable[[], str]) -> Provider:
    def search(w: Work) -> list[WebReference]:
        k = key()
        d = _http_json(_req("GET", "https://www.searchapi.io/api/v1/search", timeout=120,
                                    params={"engine": "google_lens", "url": w.url(), "search_type": "all"},
                                    headers={"Authorization": "Bearer " + k}), k, "searchapi lens")
        out = []
        for x in d.get("visual_matches") or []:
            im = x.get("image")
            im = im.get("link", "") if isinstance(im, dict) else (im or "")
            if r := _ref(x.get("title", ""), x.get("link", ""), im, x.get("thumbnail", ""), "google_lens:searchapi",
                         x.get("source", "")):
                out.append(r)
        return out

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://www.searchapi.io/api/v1/me", headers={"Authorization": "Bearer " + k},
                                    timeout=20), k, "searchapi account")
        return int((d.get("account") or {}).get("remaining_credits") or 0)
    # a 100-search grant with no monthly allowance on the account (2026-11-05): one-time until a reset is seen
    return Provider("searchapi", search, allowance=100, renews="one-time", balance=balance, enabled=lambda: bool(key()))


def serper_provider(key: Callable[[], str]) -> Provider:
    def search(w: Work) -> list[WebReference]:
        k = key()
        out = []
        for _ in range(2):        # an empty 200 came back on 2 of 10 trials; a retry answered (not charged)
            d = _http_json(_req("POST", "https://google.serper.dev/lens", json={"url": w.url(), "gl": "us", "hl": "en"},
                                         headers={"X-API-KEY": k}, timeout=90), k, "serper lens")
            out = [r for x in d.get("organic") or []
                   if (r := _ref(x.get("title", ""), x.get("link", ""), x.get("imageUrl", ""), x.get("thumbnailUrl", ""),
                                 "google_lens:serper", x.get("source", "")))]
            if out:
                break
            time.sleep(2)
        return out

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://google.serper.dev/account", headers={"X-API-KEY": k}, timeout=20), k,
                       "serper account")
        return int(d.get("balance") or 0)
    # 2,500 one-time credits, 3 per Lens search; the same credits pay confirm's text image searches
    return Provider("serper", search, allowance=2500, renews="one-time", cost=3, balance=balance,
                    enabled=lambda: bool(key()))


_BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/130 Safari/537.36"}


def _resolve_goto(rel: str) -> str:
    """Google '/goto?url=<opaque>' -> the real page, from one no-follow GET to google.com (free)."""
    absu = "https://www.google.com" + rel
    try:
        r = _req("GET", absu, headers=_BROWSER, allow_redirects=False, timeout=10)
        loc = r.headers.get("location", "")
        if r.status_code in (301, 302, 303, 307, 308) and loc.startswith("http"):
            return loc
    except requests.RequestException:
        pass
    return absu


OXY_PRICE = 0.00135        # $ per Google request with JS (Oxylabs price list, 2026-11-06); failed calls are free


def oxylabs_provider(key: Callable[[], str]) -> Provider:
    """Oxylabs Web API, google/lens: a one-time $5 welcome credit that pauses at $0 (no card on file).
    Found 9 of 10 trial works (2026-11-06). Its account endpoint counts requests, not money."""
    def search(w: Work) -> list[WebReference]:
        k = key()
        d = _http_json(_req("POST", "https://webapi.oxylabs.io/v1/scrape/google/lens", timeout=180,
                                     json={"query": w.url(), "output": ["json"]},
                                     headers={"Authorization": "Bearer " + k}), k, "oxylabs lens")
        out = []
        for res in d.get("results") or []:
            js = ((res.get("json") or {}).get("results") or {})
            for x in (js.get("exact_match") or []) + (js.get("organic") or []):
                link = x.get("url") or ""
                if link.startswith("/goto?") or link.startswith("/url?"):
                    link = _resolve_goto(link) if len(out) < 30 else "https://www.google.com" + link
                if r := _ref(x.get("title", ""), link, "", x.get("url_thumbnail", ""), "google_lens:oxylabs",
                             x.get("domain", "")):
                    out.append(r)
        return out

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://data.oxylabs.io/v2/stats", params={"group_by": "month"}, timeout=30,
                                    headers={"Authorization": "Bearer " + k}), k, "oxylabs stats")
        used = sum(p.get("all_count", 0) for row in d.get("data") or [] for p in row.get("products") or [])
        return max(0, int(5.0 / OXY_PRICE) - used)
    return Provider("oxylabs", search, allowance=int(5.0 / OXY_PRICE), renews="one-time", balance=balance,
                    enabled=lambda: bool(key()))


APIFY_PER_IMAGE = 0.0035  # $ measured on 10 works (2026-11-06): $0.00005 start + $0.0025 item + platform


def apify_provider(key: Callable[[], str]) -> Provider:
    """Apify actor omkar-cloud/google-lens-scraper on the free plan: $5 of credit a month, a hard stop (no
    billing). Found 10 of 10 trial works, 9 naming the artist, ~5 s (2026-11-06). Sir's first pick, actor
    MaNVYRogwHemtywEz, charged its cap and delivered nothing; the $5 serves any actor."""
    def search(w: Work) -> list[WebReference]:
        k = key()
        r = _req("POST", "https://api.apify.com/v2/acts/omkar-cloud~google-lens-scraper/run-sync-get-dataset-items",
                          params={"maxTotalChargeUsd": 0.05, "timeout": 240}, timeout=300,
                          headers={"Authorization": "Bearer " + k},
                          json={"images": [w.url()], "mode": "visual_matches", "country": "US", "language": "en"})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"apify HTTP {r.status_code}: {r.text[:240].replace(k, '***')}")
        items = r.json() or []
        if not items:      # the actor pushes nothing when Google refuses the picture's host (not billed)
            raise RuntimeError("apify: no result pushed (Google Lens refused the image URL"
                               + (", a tmpfiles link)" if "tmpfiles" in w.url() else ")"))
        out = []
        for it in items:
            for x in it.get("visual_matches") or []:
                im = x.get("image")
                im = im.get("link", "") if isinstance(im, dict) else (im or "")
                if ref := _ref(x.get("title", ""), x.get("link", ""), im, x.get("thumbnail", ""), "google_lens:apify",
                               x.get("source", "")):
                    out.append(ref)
        return out

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://api.apify.com/v2/users/me/limits", timeout=30,
                                    headers={"Authorization": "Bearer " + k}), k, "apify limits")["data"]
        left = float(d["limits"]["maxMonthlyUsageUsd"]) - float(d["current"]["monthlyUsageUsd"])
        return max(0, int(left / APIFY_PER_IMAGE))
    return Provider("apify", search, allowance=int(5.0 / APIFY_PER_IMAGE), renews="monthly", balance=balance,
                    enabled=lambda: bool(key()))


def _visual_rows(items, source: str) -> list[WebReference]:
    out = []
    for x in items or []:
        im = x.get("image") or x.get("image_url")
        im = (im.get("link") or im.get("url") or "") if isinstance(im, dict) else (im or "")
        if r := _ref(x.get("title", ""), x.get("link") or x.get("url") or "", im,
                     x.get("thumbnail") or x.get("thumbnail_url") or "", source, x.get("source") or x.get("domain") or ""):
            out.append(r)
    return out


def rapidapi_provider(name: str, host: str, path: str, params: Callable[[str], dict], rows: Callable[[dict], list],
                      free_limit: int, quota_path: Path, key: Callable[[], str]) -> Provider:
    """A RapidAPI API on its free BASIC plan: a monthly request quota with a HARD limit (no overage, checked
    from each API's plan list, 2026-11-06). Each vendor has its own quota. The balance comes back in every
    reply's x-ratelimit headers, kept in ``quota_path``. If the reply's limit is not the free plan's, a paid
    plan was subscribed and overage could bill: the provider refuses (Sir: nothing may ever spill into paid)."""
    def _quota() -> dict:
        try:
            return json.loads(quota_path.read_text())
        except (OSError, ValueError):
            return {}

    def search(w: Work) -> list[WebReference]:
        q = _quota().get(host) or {}
        if q.get("limit") and int(q["limit"]) != free_limit:
            raise RuntimeError(f"{name}: plan limit {q['limit']} is not the free quota {free_limit}; refusing (overage risk)")
        k = key()
        r = _req("GET", f"https://{host}{path}", params=params(w.url()), timeout=150,
                         headers={"x-rapidapi-key": k, "x-rapidapi-host": host})
        h = {k_.lower(): v for k_, v in r.headers.items()}
        if "x-ratelimit-requests-remaining" in h:
            allq = _quota()
            allq[host] = {"limit": h.get("x-ratelimit-requests-limit"), "remaining": h["x-ratelimit-requests-remaining"],
                          "reset_s": h.get("x-ratelimit-requests-reset"), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            quota_path.write_text(json.dumps(allq, indent=1))
            if int(h.get("x-ratelimit-requests-limit") or free_limit) != free_limit:
                raise RuntimeError(f"{name}: plan limit is not the free quota; refusing further calls")
        return rows(_http_json(r, k, name))

    def balance() -> int:
        q = _quota().get(host)
        if not q:
            return free_limit                 # never called: the whole free quota
        t = time.mktime(time.strptime(q["t"], "%Y-%m-%dT%H:%M:%S"))
        if q.get("reset_s") and time.time() > t + int(q["reset_s"]):
            return int(q.get("limit") or free_limit)      # the quota has reset since the last reply
        return int(q["remaining"])
    return Provider(name, search, allowance=free_limit, renews="monthly", balance=balance, enabled=lambda: bool(key()))


def scrapebadger_provider(key: Callable[[], str]) -> Provider:
    """ScrapeBadger Google Lens: 20 credits a search from 1,000 one-time sign-up credits (a prepaid wallet,
    hard stop), shared with the scrapebadger-reddit MCP. Found 10 of 10 trial works (2026-11-06). Reserve."""
    def search(w: Work) -> list[WebReference]:
        k = key()
        for _ in range(2):
            r = _req("GET", "https://scrapebadger.com/v1/google/lens/search", timeout=120,
                             params={"url": w.url(), "gl": "us", "hl": "en"}, headers={"x-api-key": k})
            if r.status_code == 429:          # the free tier allows 5 calls a minute per account
                time.sleep(min(60, int(r.headers.get("Retry-After") or 15)))
                continue
            break
        d = _http_json(r, k, "scrapebadger lens")
        return _visual_rows([{**x, "image": x.get("original_thumbnail")} for x in d.get("lens_results") or []],
                            "google_lens:scrapebadger")

    def balance() -> int:
        k = key()
        d = _http_json(_req("GET", "https://scrapebadger.com/v1/account/me", headers={"x-api-key": k}, timeout=20), k,
                       "scrapebadger account")
        return int(d.get("total_credits_balance") or 0)
    return Provider("scrapebadger", search, allowance=1000, renews="one-time", cost=20, balance=balance,
                    enabled=lambda: bool(key()))


def vision_provider(vision) -> Provider:
    """Cloud Vision web detection: 1,000 free calls a month, billed after that, so it is counted locally and
    stops at the floor (VisionWebDetection.detect enforces the same limit for every other caller)."""
    def search(w: Work) -> list[WebReference]:
        return vision.references(vision.detect(w.img))
    return Provider("vision", search, allowance=vision.free_per_month, renews="monthly", balance=None,
                    enabled=lambda: bool(vision.enabled), charges_itself=True)


def read_keys(path: str) -> dict[str, str]:
    """The "name: key" secrets file; never logged."""
    out = {}
    try:
        for line in open(path, encoding="utf-8"):
            name, sep, val = line.strip().partition(":")
            if sep and val.strip() and not name.startswith("#"):
                out[name.strip().lower()] = val.strip()
    except OSError:
        pass
    return out


def build_pool(cfg, web, vision) -> LensPool:
    """The Lens slot as configured: providers whose key is present, one ledger in the state directory."""
    keys = read_keys(cfg["sources"].get("api_keys_path", ""))

    def key(name: str, store: str = "") -> Callable[[], str]:
        def get() -> str:
            k = keys.get(name) or (web._key(store) if store and web is not None else "")
            if k:
                _KEYS.add(k)
            return k
        return get
    providers = [serpapi_provider(key("serpapi", "serpapi")), searchapi_provider(key("searchapi")),
                 serper_provider(key("serper", "serper")), oxylabs_provider(key("oxylabs")),
                 apify_provider(key("apify")), scrapebadger_provider(key("scrapebadger"))]
    quota = cfg.path("state_dir") / "rapidapi-quota.json"
    # RapidAPI (Sir, 2026-11-06): one key, each API its own vendor and free monthly quota
    providers.append(rapidapi_provider(
        "rapidapi_omkar", "best-google-lens-scraper-free-200-calls.p.rapidapi.com", "/search",
        lambda u: {"image": u, "country": "US", "language": "en"},
        lambda d: _visual_rows(d.get("visual_matches") or (d.get("data") or {}).get("visual_matches"),
                               "google_lens:rapidapi_omkar"), 200, quota, key("rapidapi")))
    providers.append(rapidapi_provider(
        "rapidapi_letscrape", "real-time-lens-data.p.rapidapi.com", "/search",
        lambda u: {"url": u, "language": "en", "country": "us", "safe_search": "off"},
        lambda d: _visual_rows((d.get("data") or {}).get("visual_matches"), "google_lens:rapidapi_letscrape"),
        50, quota, key("rapidapi")))
    if vision is not None:
        providers.append(vision_provider(vision))
    ledger = vision.ledger if vision is not None and vision.ledger is not None else Ledger(cfg.path("state_dir") / "lens_pool.json")
    pool = LensPool(providers, ledger, web.temp_public_url)
    pool.extras = {"serpapi_exact": ("serpapi", serpapi_exact(key("serpapi", "serpapi"))),
                   "searchapi_yandex": ("searchapi", searchapi_yandex(key("searchapi")))}
    return pool
