"""Score finished runs against a hand-written gold file. Lives outside the pipeline on purpose:
the pipeline never sees gold, file names or this module.

A run PASSES only when its status is ``complete`` (every stage gate passed, delivered) AND the
identification agrees with gold: artist surname, ≥60% of the gold title's key words, year ±2.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import yaml

from .pipeline import content_id

STOP = {"the", "of", "a", "an", "and", "in", "to", "by", "at", "on", "le", "la", "de", "du"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def _words(s: str) -> set[str]:
    return {w for w in _fold(s).split() if w not in STOP and len(w) > 1}


def score(gold: dict, ident: dict) -> dict:
    sur = _fold(gold["artist"]).split()[-1]
    artist_ok = sur in _fold(ident.get("artist", "")).split()
    gw = _words(gold["title"])
    title_cov = len(gw & _words(ident.get("title", ""))) / max(1, len(gw))
    years = [int(y) for y in re.findall(r"\d{4}", str(ident.get("date", "")))]
    gy = gold.get("year")
    lo, hi = (gy if isinstance(gy, list) else [gy, gy]) if gy else (None, None)
    year_ok = lo is None or any(lo - 2 <= y <= hi + 2 for y in years)
    return {"artist_ok": artist_ok, "title_coverage": round(title_cov, 2), "title_ok": title_cov >= 0.6,
            "year_ok": year_ok}


def evaluate(cfg, gold_path: Path, out_json: str | None = None) -> int:
    gold = yaml.safe_load(gold_path.read_text())
    base = gold_path.parent / gold.get("images_dir", ".")
    rows, passed = [], 0
    for g in gold["artworks"]:
        img = base / g["file"]
        rid = content_id(img)
        rec_p = cfg.path("work_dir") / rid / "record.json"
        rec = json.loads(rec_p.read_text()) if rec_p.exists() else {}
        ident = rec.get("stages", {}).get("identify", {})
        sc = score(g, ident)
        ok = rec.get("status") == "complete" and sc["artist_ok"] and sc["title_ok"] and sc["year_ok"]
        passed += ok
        rows.append({"file": g["file"], "run_id": rid, "status": rec.get("status", "not run"),
                     "blocked_stage": rec.get("blocked_stage", ""), "identified_artist": ident.get("artist", ""),
                     "identified_title": ident.get("title", ""), "identified_date": ident.get("date", ""),
                     "id_tier": (ident.get("evidence") or {}).get("tier", ""), **sc, "PASS": ok,
                     "frame": rec.get("frame", "")})
    for r in rows:
        print(f"{'PASS' if r['PASS'] else 'FAIL'}  {r['status']:9} {r['blocked_stage']:9} "
              f"{r['identified_artist'][:24]:24} {r['identified_title'][:40]:40} {r['identified_date']:6}  "
              f"[a={int(r['artist_ok'])} t={r['title_coverage']:.2f} y={int(r['year_ok'])}]  {r['file'][:40]}")
    print(f"\n{passed}/{len(rows)} artworks passed end to end")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=2))
    return 0 if passed == len(rows) else 1
