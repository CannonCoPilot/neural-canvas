"""neural-canvas command line.

    neural-canvas serve                 watch folder → Pulse tasks → runs (the launchd service)
    neural-canvas submit IMG...         copy images into the watch folder
    neural-canvas run IMG... [--no-deliver]   run directly, without Pulse (development)
    neural-canvas retry [--stage S]     put blocked Pulse tasks back to staging (optionally only stage S)
    neural-canvas auth-photos           one-time Google Photos consent (opens a browser)
    neural-canvas eval GOLD.yaml        score finished runs against a gold file
    neural-canvas status                board summary of neural-canvas tasks
    neural-canvas report GOLD OUT.html  self-contained gallery of finished runs with their evidence
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neural-canvas", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config.yaml (default: repo root or $NEURAL_CANVAS_CONFIG)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--once", action="store_true", help="one scan + run pass, then exit")
    p = sub.add_parser("submit")
    p.add_argument("images", nargs="+")
    p = sub.add_parser("run")
    p.add_argument("images", nargs="+")
    p.add_argument("--no-deliver", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore stage results from earlier runs")
    p.add_argument("--redeliver", action="store_true",
                   help="deliver again even if delivered before; the new frame replaces the old one in the album")
    p = sub.add_parser("retry")
    p.add_argument("--stage", help="only tasks blocked at this stage (e.g. deliver)")
    sub.add_parser("auth-photos")
    p = sub.add_parser("eval")
    p.add_argument("gold")
    p.add_argument("--json", help="write the scored table here")
    sub.add_parser("status")
    p = sub.add_parser("report")
    p.add_argument("gold")
    p.add_argument("out")
    p.add_argument("--notes", help="HTML fragment appended to the page (open items)")
    p = sub.add_parser("build-index", help="embed the catalogues of the N painters with the most catalogued "
                                           "paintings (Wikidata, NGA, Artvee), resumable; runs search them from cache")
    p.add_argument("--top", type=int, default=500)
    sub.add_parser("build-global-index", help="rebuild first look's cached picture index from every embedded catalogue")
    p = sub.add_parser("nga-index", help="index the National Gallery of Art open-data CSVs (objects, constituents, "
                                         "objects_constituents, published_images) into the configured SQLite file")
    p.add_argument("csv_dir")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        # the service runs works in parallel; each run's lines carry its worker thread's name
                        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s" if a.cmd == "serve"
                        else "%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "sentence_transformers", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(a.config)

    if a.cmd == "build-index":
        from .indexer import build_top_painters
        return build_top_painters(cfg, a.top)
    if a.cmd == "build-global-index":
        from .global_index import GlobalIndex
        from .references import Wikimedia
        s = cfg["sources"]
        print(GlobalIndex(cfg.path("state_dir")).rebuild(Wikimedia(s["commons_api"], s["wikidata_api"], s["user_agent"])))
        return 0
    if a.cmd == "nga-index":
        from .nga import build_index
        print(build_index(Path(a.csv_dir).expanduser(), Path(cfg["sources"]["nga_db"]).expanduser()))
        return 0
    if a.cmd == "serve":
        from .service import Service
        Service(cfg).serve(once=a.once)
        return 0
    if a.cmd == "submit":
        dst = cfg.path("watch_dir")
        for i in a.images:
            shutil.copy2(i, dst / Path(i).name)
            print(f"submitted {Path(i).name} → {dst}")
        return 0
    if a.cmd == "run":
        from .pipeline import Docent, ingest
        d = Docent(cfg)
        bad = 0
        for i in a.images:
            rid, src = ingest(cfg, Path(i))
            r = d.run(rid, src, resume=not a.fresh, deliver=not a.no_deliver, redeliver=a.redeliver)
            print(json.dumps({"input": Path(i).name, **r.__dict__}), flush=True)
            bad += r.status not in ("complete", "rendered")
        return 1 if bad else 0
    if a.cmd == "retry":
        from .service import Pulse
        pl = Pulse(cfg["pulse"]["url"])
        for t in pl.list(cfg["pulse"]["runner_label"], status="open"):
            labels = t.get("labels", [])
            if "blocked:yes" not in labels:
                continue
            if a.stage and f"reason:docent-{a.stage}" not in labels:
                continue
            pl.labels(t["id"], add=["staging:wait", "blocked:no", "active:no"],
                      remove=[l for l in labels if l.startswith(("reason:", "staging:", "active:"))])
            pl.note(t["id"], f"re-queued by `neural-canvas retry`{' --stage ' + a.stage if a.stage else ''}")
            print(f"re-queued {t['id']}")
        return 0
    if a.cmd == "auth-photos":
        from .deliver import AuthRequired, GooglePhotos
        try:
            GooglePhotos(cfg["google_photos"], cfg.path("state_dir")).authorize_interactive()
        except AuthRequired as e:
            print(e, file=sys.stderr)
            return 2
        print("Google Photos authorised.")
        return 0
    if a.cmd == "eval":
        from .evaluate import evaluate
        return evaluate(cfg, Path(a.gold), a.json)
    if a.cmd == "report":
        from .report import build
        known = cfg.path("state_dir") / "known.json"
        tasks = json.loads(known.read_text()) if known.exists() else {}
        notes = Path(a.notes).read_text() if a.notes else ""
        print(json.dumps(build(cfg, Path(a.gold), Path(a.out), tasks, notes)))
        return 0
    if a.cmd == "status":
        from .service import Pulse
        pl = Pulse(cfg["pulse"]["url"])
        for t in sorted(pl.list(cfg["pulse"]["runner_label"]), key=lambda t: t.get("created_at", "")):
            res = (t.get("metadata") or {}).get("result") or {}
            lab = [l for l in t.get("labels", []) if l.startswith(("docent:", "reason:", "blocked:yes", "active:"))]
            print(f"{t['id']:15} {t['status']:11} {' '.join(lab):45} {res.get('artist', '')} — {res.get('title', '')}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
