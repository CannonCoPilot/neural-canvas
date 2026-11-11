"""Two local model agents rule on a work's facts from its evidence ledger; mechanical rules decide what prints.

* The ADJUDICATOR (the vision model, shown the photograph) rules each fact and cites the ledger items that
  state it. It may ask for up to three web searches when the evidence is thin or contradictory.
* The CHECKER (a separate text model: a ruling is never checked by the model that made it) reads each cited
  item and says whether it really states the value, and which items state something else. Where words can be
  compared, the deterministic check ``states`` decides instead: measured on the withheld bench, the 8b checker
  missed true support and vetoed verified copies that named the artist outright. It rules items in another
  script, and its contradictions are recorded (a curated record's contradiction vetoes).
* The RULES are code, not models (Sir, 2026-11-07): a printed fact needs at least two supporting items from
  two independent origins, at least one of them a verified copy or a catalogue record (an engine's answer
  supports but never anchors: engines name look-alikes); the card's date, medium and collection need a
  catalogue record among them. Every cited item must carry the value in its own words (``states``). Leads
  (model guesses, the signature reading) never count.

Rounds are capped (``rounds_max``). The cap is a circuit-breaker, never an acceptance: a fact that fails
stays unruled, the work stays OPEN and blocks, and the caller alerts that the approach needs redesign
(No Silent Degradation).
"""
from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
import unicodedata
from typing import Callable

from PIL import Image

from .evidence import EVIDENCE, Ledger
from .vlm import VLM

log = logging.getLogger(__name__)

FACTS = ("artist", "title", "date", "medium", "collection")
# what can anchor a fact to THIS picture: a verified copy or a catalogue record. Engine answers and captions support
# but never anchor: engines name look-alikes (withheld bench pass 4: a James Ward colour study was ruled
# "Josef Albers" on engine answers about similar squares)
ANCHORS = ("copy", "record")
# a caption is evidence about the picture beside it: only a near-copy (the identification bar is 0.86 with geometry)
# speaks for this one. An article on Eric Wert's own paintings, found by a search for his name, ruled a Suzan Visser
# "Eric Wert" (bench pass 7); Visser's Instagram caption sits beside a 0.86 near-copy
CAPTION_CLIP = 0.85

ADJUDICATE_PROMPT = """You are a museum registrar ruling on what an artwork is. The photograph is attached. Searches of
the web, search engines and catalogues have gathered the evidence below, one item per line, each with an id.
EVIDENCE:
{ledger}
{context}
How to weigh it:
- A VERIFIED COPY is this same artwork, proven by two image instruments; its text is what that page calls it.
  An unverified copy or a page caption may show a different picture.
- A search engine's answer is what Google, Bing or Yandex say about this exact photograph.
- A catalogue record marked curated outranks web pages when it states a fact.
- Shops, music pages, social media and Pinterest boards reuse pictures under other names (an album cover names
  the band, a board names its collector). Prefer what most independent sites agree on; one site repeating
  itself is one witness.
- Model guesses and signature readings are leads only: never cite them.
- A name may be spelled differently or written in another script (Suzan/Susan, Cyrillic): the same person when
  the rest agrees.
- Years in brackets beside an artist's name are the artist's life, not the work's date.
Rule each fact: the artist (full name only, no dates or nationality), the work's own title (no artist, size,
price, site or "print" wording; in English when an item gives it), its date, medium and holding collection.
For each give the value and the ids of up to 8 items that state it, verified copies and records first, from as
many different sites as possible. Use "" and no ids when no item states it:
never fill a fact from your own knowledge. If the evidence is thin or contradictory, give up to 3 web searches
(artist and title wording) that would settle it.
Reply ONLY with JSON:
{{"artist": {{"value": "", "cites": []}}, "title": {{"value": "", "cites": []}}, "date": {{"value": "", "cites": []}},
  "medium": {{"value": "", "cites": []}}, "collection": {{"value": "", "cites": []}},
  "searches": [], "reasoning": "one or two sentences"}}"""

CHECK_PROMPT = """You check a registrar's ruling on an artwork. Each fact cites evidence items by id. Be strict.
RULING:
{ruling}
EVIDENCE (every item gathered for this artwork):
{ledger}
For each fact, list:
- "supports": the cited ids whose text really states that value (the same name in another spelling, language or
  script counts; a different work, a different person, or a vaguer description does not);
- "unsupported": the cited ids whose text does not state it;
- "contradicts": ids of any items, cited or not, that state a DIFFERENT value for this artwork (verified copies
  and catalogue records matter most; ignore model guesses).
Then "verdict": "accept" when every ruled fact is supported, else "revise", with a one-sentence "reason", and
up to 3 web "searches" that would settle a doubtful fact.
Reply ONLY with JSON:
{{"artist": {{"supports": [], "unsupported": [], "contradicts": []}}, "title": {{"supports": [], "unsupported": [], "contradicts": []}},
  "date": {{"supports": [], "unsupported": [], "contradicts": []}}, "medium": {{"supports": [], "unsupported": [], "contradicts": []}},
  "collection": {{"supports": [], "unsupported": [], "contradicts": []}}, "verdict": "accept", "reason": "", "searches": []}}"""


def _ids(xs) -> list[str]:
    """Item ids as the models write them ('E12', 'e12', 12) -> 'E12'."""
    out = []
    for x in xs if isinstance(xs, list) else []:
        s = str(x).strip().upper()
        out.append(s if s.startswith("E") else f"E{s}" if s.isdigit() else s)
    return out


def _fact(d: dict, name: str) -> tuple[str, list[str]]:
    v = d.get(name)
    if isinstance(v, dict):
        return str(v.get("value") or "").strip(), [str(x) for x in v.get("cites") or []][:12]
    return str(v or "").strip(), []


class Agents:
    def __init__(self, adjudicator: VLM, checker: VLM):
        self.adjudicator, self.checker = adjudicator, checker

    def adjudicate(self, img: Image.Image | None, ledger: Ledger, context: str = "") -> dict:
        prompt = ADJUDICATE_PROMPT.format(ledger=ledger.to_prompt(), context=context)
        # 3000 tokens: a ruling citing 40-odd verified copies was cut off mid-JSON three times (af Klint, pass 6)
        return self.adjudicator.ask_json(prompt, [img] if img is not None else [], required=("artist", "title"),
                                         max_side=768, num_predict=3000)

    def check(self, ledger: Ledger, ruling: dict) -> dict:
        """The checker sees the cited items and every verified copy and record (where a contradiction would
        matter), not the whole ledger."""
        lines = "\n".join(f"{f}: {v!r} citing {', '.join(c) or 'nothing'}" for f in FACTS for v, c in [_fact(ruling, f)] if v)
        cited = {x for f in FACTS for x in _fact(ruling, f)[1]}
        keep = [i for i in ledger.items if i.id in cited or i.kind == "record" or (i.kind == "copy" and i.verified)]
        return self.checker.ask_json(CHECK_PROMPT.format(ruling=lines, ledger="\n".join(i.line() for i in keep[:80])),
                                     required=("verdict",), num_predict=3000)

    def settle(self, img: Image.Image | None, ledger: Ledger, need: tuple[str, ...] = ("artist", "title"),
               record_facts: tuple[str, ...] = (), rounds_max: int = 2, context: str = "",
               search: Callable[[dict, list[str]], int] | None = None) -> dict:
        """Adjudicate, check and apply the rules, up to ``rounds_max`` rounds. Between rounds ``search(ruling,
        queries)`` gathers more evidence (it returns how many items it added), and the adjudicator is told which
        of its rulings failed the rules and why: a model that ruled from its own sense of the picture gets the
        evidence put back in front of it (withheld bench pass 5: a James Ward study ruled "Josef Albers" while a
        verified Artvee copy and Artvee's page named Ward). Returns the last verdict with every round's record."""
        rounds = []
        verdict: dict = {}
        feedback = ""
        for rnd in range(1, max(1, rounds_max) + 1):
            # a feedback round rules from the text alone: the picture is already proven by the instruments, and its
            # style is what pulled the first ruling to a look-alike (a James Ward study kept coming back "Josef
            # Albers" with the photograph attached, even with the anchors listed; offline test, 2026-11-07)
            ruling = self.adjudicate(None if feedback else img, ledger, context + feedback)
            check = self.check(ledger, ruling) if any(_fact(ruling, f)[0] for f in FACTS) else {"verdict": "revise"}
            verdict = rule_facts(ledger, ruling, check, need, record_facts)
            rounds.append({"round": rnd, "ruling": ruling, "check": check, "failures": verdict["failures"]})
            log.info("adjudication round %d: %s", rnd, {f: verdict["facts"].get(f) for f in need} if not verdict["failures"]
                     else verdict["failures"])
            if not verdict["failures"] or rnd >= rounds_max or search is None:
                break
            asked = [q for d in (ruling, check) for q in (d.get("searches") if isinstance(d.get("searches"), list) else [])]
            queries = list(dict.fromkeys(q.strip() for q in asked if isinstance(q, str) and len(q.strip()) >= 4))[:3]
            rounds[-1]["new_items"] = search(ruling, queries)
            anchors = [i.line() for i in ledger.items if i.kind == "record" or (i.kind == "copy" and i.verified)][:30]
            feedback = ("\nYOUR PREVIOUS RULING FAILED THE EVIDENCE RULES: " + "; ".join(verdict["failures"])
                        + ". A fact needs two independent sites that state it, at least one of them a VERIFIED COPY "
                        "or a catalogue record. These are the ONLY items that can anchor a ruling; each verified copy "
                        "is proven to be this very picture:\n" + "\n".join(anchors)
                        + "\nThe photograph's style can suggest a famous look-alike artist; the anchors decide. Which "
                        "artist and title do these anchors themselves name? Rule that, citing them and every other item "
                        "that agrees. Do not repeat a ruling the anchors do not carry.")
        verdict["rounds"] = rounds
        return verdict


def rule_facts(ledger: Ledger, ruling: dict, check: dict, need: tuple[str, ...] = ("artist", "title"),
               record_facts: tuple[str, ...] = ()) -> dict:
    """The mechanical rules over one adjudication and its check (see the module docstring)."""
    out: dict = {"facts": {}, "support": {}, "unruled": {}, "failures": [], "contradicted": {}}
    if not ledger.verified():
        out["failures"].append("no copy of the picture verified on both instruments")
    for f in FACTS:
        value, cites = _fact(ruling, f)
        if not value:
            if f in need:
                out["unruled"][f] = "no item states it"
            continue
        c = check.get(f) if isinstance(check.get(f), dict) else {}
        confirmed, rejected = _ids(c.get("supports")), _ids(c.get("unsupported"))
        # an item supports when its own words carry the value; an item the words cannot judge (another script)
        # needs the checker's confirmation and no rejection. The 8b checker is unreliable both ways where words
        # can be compared: it missed Instagram's "Hummingbird and Fish. Painted by Suzan Visser" (bench pass 5)
        # and rejected verified copies reading "... Different Colours by James Ward" (smoke test, 2026-11-07)
        items = [i for x in _ids(cites) if (i := ledger.get(x)) and i.kind in EVIDENCE
                 and (i.kind != "caption" or i.clip >= CAPTION_CLIP)
                 and ((lex := states(i, f, value)) is True or (lex is None and x in confirmed and x not in rejected))]
        if f in record_facts:
            # a catalogue record whose OWN field for this fact states the value supports it, cited or not: the
            # adjudicator often cites pages and leaves the records out (Wyeth's date, 2026-11-10: Wikidata,
            # Wikipedia and WikiArt all read 1948, none was cited, and the date stayed off the card)
            cited = {i.id for i in items}
            items += [i for i in ledger.items if i.kind == "record" and i.id not in cited and i.claims.get(f)
                      and states(SimpleNamespace(text=str(i.claims[f]), claims={}), f, value) is True]
        origins = {i.origin for i in items}
        anchors = [i for i in items if i.kind in ANCHORS]
        why = ""
        if f == "title" and re.search(r"(\.\.\.|\u2026)\s*$", value):
            why = "cut off mid-title (a page caption's ellipsis)"
        elif len(origins) < 2:
            why = f"{len(origins)} independent item(s) state it (2 needed)"
        elif not anchors:
            why = "no verified copy or catalogue record states it"
        elif f in record_facts and not any(i.kind == "record" for i in items):
            why = "no catalogue record states it"
        against = [x for x in c.get("contradicts") or [] if (i := ledger.get(str(x))) and i.kind == "record" and i.overrule
                   and contradicts(i, f, value)]
        if not why and against:
            why = f"curated record(s) {', '.join(map(str, against))} state otherwise"
        if why:
            out["unruled"][f] = f"{value!r}: {why}"
            continue
        out["facts"][f] = value
        out["support"][f] = [i.id for i in items]
        if c.get("contradicts"):
            out["contradicted"][f] = [str(x) for x in c["contradicts"]]
    for f in need:
        if f not in out["facts"]:
            out["failures"].append(f"{f} not ruled: {out['unruled'].get(f, 'no item states it')}")
    return out


_STOP = {"the", "and", "with", "of", "on", "in", "a", "an", "de", "la", "le", "het", "en", "van", "der", "des", "du", "und", "von"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


_ENGLISH = {"the", "of", "and", "in", "on", "at", "with", "from", "a", "an", "by", "to", "for"}


def contradicts(item, fact: str, value: str) -> bool:
    """A curated record's contradiction stands only when its own words can contradict: the checker flagged the
    Ghent Altarpiece's co-author ("Hubert van Eyck") and its German and Dutch titles as contradicting "Jan van
    Eyck" and the English title, and the work blocked (2026-11-10). An artist is contradicted only by another
    surname; a title only by a record that states one in the same language (English titles carry English words)."""
    claim = str(item.claims.get(fact) or "").strip()
    if not claim:
        return False
    if fact == "artist":
        a, b = _fold(claim).split(), _fold(value).split()
        return bool(a and b) and a[-1] != b[-1] and b[-1] not in a
    if fact == "title":
        eng = lambda t: bool(set(_fold(t).split()) & _ENGLISH)   # noqa: E731
        return eng(claim) == eng(value) and states(SimpleNamespace(text=claim, claims={}), fact, value) is False
    from .catalogue import medium_clash, same_collection
    if fact == "date":          # "before 1426-1432" does not contradict 1432: only a record sharing no year does
        a, b = set(re.findall(r"\d{4}", claim)), set(re.findall(r"\d{4}", value))
        return bool(a and b) and not a & b
    if fact == "medium":        # only another paint contradicts (oil against tempera), not other words for the same
        return medium_clash(claim, value)
    if fact == "collection":
        return not same_collection(claim, value) and states(SimpleNamespace(text=claim, claims={}), fact, value) is False
    return True


def states(item, fact: str, value: str) -> bool | None:
    """A citation check the models cannot talk past: the item's own words must carry the value. Both models
    accepted a page about another Visser work, "Fruit and Ice Cream with Hummingbirds", as stating "Hummingbird
    and Fish" (smoke test, 2026-11-07). None when the item is mostly in another script (Yandex's Cyrillic): words
    cannot be compared, and only the checker can judge it."""
    raw = " ".join([item.text, *map(str, item.claims.values())])
    letters = [c for c in raw if c.isalpha()]
    foreign = sum(1 for c in letters if re.match(r"[\u0400-\u04ff\u0590-\u06ff\u3040-\u9fff]", c))
    if letters and foreign / len(letters) >= 0.3:      # mostly another script: one stray character is not enough
        return None
    text = set(_fold(raw).split())
    joined = " " + _fold(raw) + " "
    if fact == "artist":
        # the last name, or two of the name's parts: pages write "José María Velasco" where Wikidata has
        # "José María Velasco Gómez Obregón" (withheld bench pass 4)
        name = _fold(value).split()
        parts = [w for w in name if len(w) >= 3 and w not in _STOP]
        return bool(name) and (name[-1] in text or sum(w in text for w in parts) >= 2)
    if fact == "date":
        years = re.findall(r"\d{4}", value)
        return bool(years) and any(y in joined for y in years)
    words = [w for w in _fold(value).split() if w not in _STOP and len(w) > 2] or _fold(value).split()
    stem = lambda w: w[:-1] if w.endswith("s") and len(w) > 3 else w      # noqa: E731  (hummingbirds ~ hummingbird)
    have = {stem(w) for w in text}
    need = 0.6 if fact == "title" else 0.5
    return bool(words) and sum(stem(w) in have for w in words) / len(words) >= need


def summary(verdict: dict, ledger: Ledger) -> str:
    """A short, human-readable account of a verdict for the run record and alerts."""
    lines = [f"{f}: {v} ({', '.join(verdict['support'].get(f, []))})" for f, v in verdict.get("facts", {}).items()]
    lines += [f"UNRULED {f}: {why}" for f, why in verdict.get("unruled", {}).items()]
    return "; ".join(lines)[:600] or json.dumps(verdict)[:600]
