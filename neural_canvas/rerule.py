"""``neural-canvas rerule``: every finished work's catalogue ruled again by the current rules over its stored
evidence, with no search and no model call (Sir, 2026-11-13). A dry run works on a copy of the state stores (the
one-name, one-nationality and life stores remember the first form they see), so it never decides a real card."""
import json
import shutil
import tempfile
import time
from pathlib import Path

from .catalogue import rerule

CARD = ("artist", "title", "date", "medium", "collection", "movement", "artist_nationality", "artist_born",
        "artist_died", "inscribed_title")
STORES = ("artist_names.json", "artist_nationality.json", "artist_lives.json")


class _State:
    """The config with its state directory moved to a scratch copy of the stores."""

    def __init__(self, cfg, state_dir: Path):
        self._cfg, self._dir = cfg, state_dir

    def path(self, name: str):
        return self._dir if name == "state_dir" else self._cfg.path(name)

    def __getitem__(self, k):
        return self._cfg[k]


def diff(old: dict, new: dict) -> dict:
    return {k: (old.get(k, ""), new.get(k, "")) for k in CARD if (old.get(k) or "") != (new.get(k) or "")}


def rerule_all(cfg, apply: bool = False, ids_file: str | None = None, out: str | None = None) -> int:
    work = cfg.path("work_dir")
    want = {l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()} if ids_file else None
    tmp = None
    use = cfg
    if not apply:
        tmp = Path(tempfile.mkdtemp(prefix="rerule-", dir=cfg.path("state_dir").parent))
        for f in STORES:
            if (cfg.path("state_dir") / f).exists():
                shutil.copy2(cfg.path("state_dir") / f, tmp / f)
        use = _State(cfg, tmp)
    changed, same, skipped, unblocked = [], 0, 0, []
    try:
        for rp in sorted(work.glob("*/record.json")):
            rec = json.loads(rp.read_text())
            rid = rec.get("run_id", rp.parent.name)
            if want is not None and rid not in want and rid[:8] not in want:
                continue
            c = (rec.get("stages") or {}).get("catalogue") or {}
            if not c or rec.get("status") not in ("complete", "rendered", "blocked"):
                continue
            new = rerule(rec, use)
            if new is None:
                skipped += 1
                continue
            d = diff(c.get("facts") or {}, new["facts"])
            was_blocked = bool(c.get("failures"))
            if was_blocked and not new["failures"]:
                unblocked.append(rid)
            if not d and was_blocked == bool(new["failures"]):
                same += 1
                continue
            changed.append(rid)
            print(f"{rid[:8]} {rec.get('status')}" + (" UNBLOCKED" if was_blocked and not new["failures"] else ""))
            for k, (o, n) in d.items():
                print(f"    {k}: {o!r} -> {n!r}")
            if apply:
                kept = {k: c[k] for k in ("seconds",) if k in c}
                rec["stages"]["catalogue"] = {"passed": not new["failures"], **kept, **new,
                                              "rerule": {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                         "changed": sorted(d)}}
                tmpf = rp.with_suffix(".tmp")
                tmpf.write_text(json.dumps(rec, indent=2, default=str))     # as the pipeline saves it
                tmpf.replace(rp)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(changed)} changed ({len(unblocked)} unblocked), {same} unchanged, {skipped} with nothing ruled to "
          f"rule again{'' if apply else ' (dry run: nothing stored)'}")
    if out:
        Path(out).write_text("".join(f"{r}\n" for r in changed))
    return 0
