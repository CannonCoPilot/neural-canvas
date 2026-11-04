"""The placard's movement line: the art-historical movement or style a work belongs to.

Catalogue first, model second, and the record says which:

1. the work's own Wikidata movement (P135), printed as catalogued;
2. the artist's Wikidata movements: one is printed as is; several (Gauguin has seven, from
   Impressionism to Primitivism) are narrowed by the vision model to the one that fits THIS work and
   its date, and its answer must be one of them;
3. neither: the vision model names the style from the image and the verified facts.

A model answer is an interpretation, like the label paragraph, never a catalogue fact, and the run
record keeps ``source`` so the two are never confused. No usable answer raises: the run blocks
rather than print a placard without the line.
"""
from __future__ import annotations

import re

from PIL import Image

from .references import Wikimedia
from .vlm import VLM

CHOOSE = """This artwork is "{title}"{date} by {artist}. The artist is catalogued under these movements:
{options}.
Which ONE of them does THIS work belong to? Judge from the image and the date: an artist's early and late
work often belong to different movements. Reply ONLY with JSON:
{{"movement": "exactly one name from the list", "reason": "one sentence"}}"""

NAME = """This artwork is "{title}"{date} by {artist}. Name the art-historical movement or style it belongs
to, as a museum label would print it: 1-4 words, e.g. "Impressionism", "Hudson River School",
"Golden Age of Danish Painting", "American Illustration", "Contemporary Realism". Reply ONLY with JSON:
{{"movement": "the name", "reason": "one sentence"}}"""


def display_name(s: str) -> str:
    """Wikidata labels are lower-case ('synthetism', 'Post-impressionism'): print them as names."""
    small = {"of", "the", "and", "de", "du", "des", "la", "le", "in"}
    words = re.split(r"(\s+|-)", s.strip())
    out = []
    for i, w in enumerate(words):
        out.append(w if (w.lower() in small and i) or not w.strip(" -") else w[:1].upper() + w[1:])
    return "".join(out)


def movements(wiki: Wikimedia, qid: str) -> list[str]:
    if not qid:
        return []
    ent = wiki.entities([qid]).get(qid, {})
    ids = wiki.claim_ids(ent, "P135")
    rel = wiki.entities(ids) if ids else {}
    return [display_name(wiki.label(rel[i])) for i in ids if i in rel and wiki.label(rel[i])]


def classify(wiki: Wikimedia, vlm: VLM, img: Image.Image, facts: dict) -> dict:
    work = facts.get("wikidata") or ""
    if own := movements(wiki, work):
        return {"movement": own[0], "source": "wikidata: the work", "candidates": own}
    artist_q = ""
    if work:
        creators = wiki.claim_ids(wiki.entities([work]).get(work, {}), "P170")
        artist_q = creators[0] if creators else ""
    if not artist_q and facts.get("artist"):
        artist_q = wiki.find_artist(facts["artist"]) or ""
    cands = movements(wiki, artist_q)
    if len(cands) == 1:
        return {"movement": cands[0], "source": "wikidata: the artist", "candidates": cands}
    ctx = {"title": facts.get("title", ""), "artist": facts.get("artist", ""),
           "date": f" ({facts['date']})" if facts.get("date") else ""}
    if cands:
        d = vlm.ask_json(CHOOSE.format(options="; ".join(cands), **ctx), [img], required=("movement",))
        pick = {c.lower(): c for c in cands}.get(str(d.get("movement", "")).strip().lower())
        if not pick:
            raise ValueError(f"movement answer {d.get('movement')!r} is not one of the artist's {cands}")
        return {"movement": pick, "source": "model, choosing among the artist's Wikidata movements",
                "candidates": cands, "reason": d.get("reason", "")}
    d = vlm.ask_json(NAME.format(**ctx), [img], required=("movement",))
    name = display_name(str(d.get("movement", "")))
    if not (2 <= len(name) <= 40) or name.count(" ") > 4:
        raise ValueError(f"unusable movement answer {d.get('movement')!r}")
    return {"movement": name, "source": "model: no Wikidata movement for the work or the artist",
            "candidates": [], "reason": d.get("reason", "")}
