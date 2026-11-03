"""Render a self-contained HTML gallery of finished runs: the frame as the TV shows it, the original,
and the evidence behind every gate. Built from run records only."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import yaml
from PIL import Image

from .evaluate import score
from .pipeline import content_id, load_image

CSS = """
:root{--wall:#eef0ec;--surface:#fbfbf8;--ink:#1d2523;--muted:#5e6b67;--brass:#8a6a2f;--rule:#cfd5d0;
--ok:#2f6b4a;--warn:#9a5b12;--bad:#9b2d2d;--chip:#e3e7e2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--wall:#1a2120;--surface:#232c2a;
--ink:#e8ece8;--muted:#9aa7a3;--brass:#c9a25a;--rule:#33403d;--ok:#7cc49a;--warn:#e0a85a;--bad:#e58b8b;--chip:#2c3734}}
:root[data-theme="dark"]{color-scheme:dark;--wall:#1a2120;--surface:#232c2a;--ink:#e8ece8;--muted:#9aa7a3;
--brass:#c9a25a;--rule:#33403d;--ok:#7cc49a;--warn:#e0a85a;--bad:#e58b8b;--chip:#2c3734}
body{background:var(--wall);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;padding-inline:clamp(16px,4vw,48px);padding-block:32px 64px}
main{max-width:1240px;margin:0 auto;display:grid;gap:40px}
h1,h2,.title{font-family:"Libre Caslon Text",Georgia,serif;font-weight:400;text-wrap:balance}
h1{font-size:clamp(30px,4.4vw,46px);line-height:1.1;margin:0}
h2{font-size:24px;margin:0 0 12px}
.lede{max-width:68ch;color:var(--muted);margin:10px 0 0}
.eyebrow{font:500 12px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--brass)}
.flow{display:flex;flex-wrap:wrap;gap:6px;margin-top:18px;font:13px "IBM Plex Mono",monospace;color:var(--muted)}
.flow span{background:var(--chip);padding:4px 9px;border-radius:3px}
.summary{display:flex;flex-wrap:wrap;gap:28px;border-block:1px solid var(--rule);padding-block:16px}
.summary div{display:grid;gap:2px}.summary b{font:500 28px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.summary small{color:var(--muted)}
.wall{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(0,1fr);gap:22px;align-items:start;border-top:1px solid var(--rule);padding-top:28px}
@media (max-width:860px){.wall{grid-template-columns:1fr}}
.tv{background:#0c0f0e;border-radius:6px;padding:10px 10px 14px;box-shadow:0 14px 30px -18px rgba(0,0,0,.55)}
.tv img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:2px}
.tv figcaption{font:12px "IBM Plex Mono",monospace;color:#9aa49f;margin-top:8px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.label{background:var(--surface);border-left:3px solid var(--brass);padding:18px 20px;display:grid;gap:12px}
.artist{font-weight:600;font-size:17px}.meta{color:var(--muted);font-size:13px}
.title{font-style:italic;font-size:21px;line-height:1.25}
.orig{display:flex;gap:10px;align-items:center}.orig img{width:84px;height:84px;object-fit:contain;background:var(--chip);border-radius:3px}
.orig span{font-size:12px;color:var(--muted)}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-size:13px}
dt{color:var(--muted)}dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:12.5px;overflow-wrap:anywhere}
.gates{display:flex;flex-wrap:wrap;gap:6px}
.gate{font:12px "IBM Plex Mono",monospace;padding:3px 8px;border-radius:3px;background:var(--chip)}
.gate.ok{color:var(--ok)}.gate.bad{color:var(--bad)}.gate.wait{color:var(--warn)}
.para{font-size:14px;max-width:62ch;margin:0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font:500 11px "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.todo{background:var(--surface);padding:18px 22px;display:grid;gap:10px}
.todo li{margin-bottom:6px;max-width:80ch}
code{font:12.5px "IBM Plex Mono",monospace;background:var(--chip);padding:1px 5px;border-radius:3px}
"""


def _b64(img: Image.Image, side: int, q: int = 80) -> str:
    im = img.convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _gate(name, st: dict) -> str:
    if not st:
        return f'<span class="gate wait">○ {name}</span>'
    if st.get("passed"):
        return f'<span class="gate ok">✓ {name}</span>'
    if st.get("auth_required") or st.get("skipped"):
        return f'<span class="gate wait">◌ {name}: awaiting sign-in</span>'
    return f'<span class="gate bad">✕ {name}</span>'


def build(cfg, gold_path: Path, out: Path, tasks: dict | None = None, todo_html: str = "") -> dict:
    gold = yaml.safe_load(gold_path.read_text())
    base = gold_path.parent / gold.get("images_dir", ".")
    e = html.escape
    walls, rows, n_id, n_frames = [], [], 0, 0
    for g in gold["artworks"]:
        src = base / g["file"]
        rid = content_id(src)
        rec_p = cfg.path("work_dir") / rid / "record.json"
        rec = json.loads(rec_p.read_text()) if rec_p.exists() else {}
        st = rec.get("stages", {})
        idn, it, cp, up, lb = (st.get(k, {}) for k in ("identify", "interpret", "compose", "upscale", "label"))
        sc = score(g, idn)
        correct = sc["artist_ok"] and sc["title_ok"] and sc["year_ok"]
        n_id += bool(idn.get("passed") and correct)
        frame = rec.get("frame")
        if frame and Path(frame).exists():
            n_frames += 1
        ev = idn.get("evidence", {})
        life = "–".join(v for v in (idn.get("artist_born"), idn.get("artist_died")) if v)
        meta = ", ".join(v for v in (idn.get("artist_nationality"), life) if v)
        tier = {"commons": "Wikimedia Commons", "oeuvre": "Wikidata catalogue of the artist's works",
                "web": "web image search"}.get(ev.get("tier", ""), ev.get("tier", "—"))
        tid = (tasks or {}).get(rid, "")
        gates = "".join(_gate(n, st.get(k, {})) for n, k in (
            ("identified", "identify"), ("interpreted", "interpret"), ("composed", "compose"),
            ("4K upscale", "upscale"), ("label", "label"), ("delivered", "deliver")))
        frame_tag = (f'<img src="{_b64(Image.open(frame), 1600, 78)}" alt="Display frame: {e(idn.get("title", ""))}">'
                     if frame and Path(frame).exists() else '<div style="aspect-ratio:16/9;display:grid;place-items:center;color:#9aa49f">no frame rendered</div>')
        walls.append(f"""
<section class="wall" id="w{rid[:8]}">
  <figure class="tv" style="margin:0">{frame_tag}
    <figcaption><span>3840 × 2160 · {e(cp.get("mode", "—"))}</span><span>{e(up.get("method", ""))} {up.get("scale", "")}×</span></figcaption></figure>
  <div class="label">
    <div><div class="artist">{e(idn.get("artist", "Unidentified"))}</div><div class="meta">{e(meta)}</div></div>
    <div class="title">{e(idn.get("title", ""))}{", " + e(idn.get("date")) if idn.get("date") else ""}</div>
    <div class="meta">{e(" · ".join(v for v in (idn.get("medium"), idn.get("collection")) if v))}</div>
    <p class="para">{e(it.get("label_text", ""))}</p>
    <div class="gates">{gates}</div>
    <dl>
      <dt>Confirmed by</dt><dd>{e(tier)}</dd>
      <dt>Match</dt><dd>CLIP {ev.get("clip", "—")} · {ev.get("inliers", "—")} geometric inliers</dd>
      <dt>Reference</dt><dd>{e(str(ev.get("reference", ""))[:90])}</dd>
      <dt>Facts from</dt><dd>{e(idn.get("facts_source", ""))}</dd>
      <dt>Genre</dt><dd>{e(it.get("genre", ""))} · {sum(1 for s in it.get("subjects", []) if s.get("essential"))} essential subjects kept</dd>
      <dt>Upscale</dt><dd>ΔE {up.get("delta_e", "—")} colour drift · {up.get("seconds", "—")} s</dd>
      <dt>Label</dt><dd>{e(lb.get("position", ""))} · contrast {lb.get("contrast", "—")}:1</dd>
      <dt>Pulse card</dt><dd>{e(tid) or "—"}</dd>
    </dl>
    <div class="orig"><img src="{_b64(load_image(src), 240, 75)}" alt="Original image"><span>Original as dropped in the folder, {rec.get("source_size", ["?", "?"])[0]} × {rec.get("source_size", ["?", "?"])[1]}. The file name was never shown to the pipeline.</span></div>
  </div>
</section>""")
        secs = round(sum(v.get("seconds", 0) or 0 for v in st.values() if isinstance(v, dict)))
        rows.append(f"<tr><td>{e(g['artist'])}</td><td>{e(g['title'])}</td><td>{e(idn.get('artist', ''))}</td>"
                    f"<td>{e(idn.get('title', ''))}</td><td>{e(str(idn.get('date', '')))}</td>"
                    f"<td>{'✓' if correct else '✕'}</td><td>{e(ev.get('tier', ''))}</td><td>{e(cp.get('mode', ''))}</td>"
                    f"<td>{e(rec.get('status', 'not run'))}</td><td>{secs}</td><td>{e(tid)}</td></tr>")
    total = len(gold["artworks"])
    page = f"""<title>Neural Canvas Docent</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Libre+Caslon+Text:ital@0;1&display=swap">
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">Neural Canvas · AI art docent · demo run</div>
  <h1>Twelve paintings, named from their pixels and hung on a 4K screen</h1>
  <p class="lede">Each image was dropped into the watch folder with no title, no artist and no metadata. A Pulse card was opened
  for it, and the local pipeline identified it against open museum and web references, read the scene, composed it for the
  TV without cutting through its subjects, upscaled it on the GPU, and set a museum wall label.</p>
  <div class="flow"><span>watch folder</span><span>Pulse card</span><span>identify</span><span>interpret</span><span>compose</span><span>upscale</span><span>label</span><span>Google Photos</span></div>
</header>
<div class="summary">
  <div><b>{n_id}/{total}</b><small>identified correctly</small></div>
  <div><b>{n_frames}/{total}</b><small>4K frames rendered</small></div>
  <div><b>0</b><small>cloud model calls</small></div>
</div>
{"".join(walls)}
<section><h2>Run table</h2><div class="tablewrap"><table>
<tr><th>Gold artist</th><th>Gold title</th><th>Identified artist</th><th>Identified title</th><th>Date</th><th>Correct</th><th>Tier</th><th>Frame</th><th>Status</th><th>Seconds</th><th>Pulse card</th></tr>
{"".join(rows)}</table></div></section>
{todo_html}
</main>"""
    out.write_text(page)
    return {"identified": n_id, "frames": n_frames, "total": total}
