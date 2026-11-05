"""National Gallery of Art open data as a source of truth (Sir, 2026-11-05).

nga.gov answers automated requests with a Cloudflare challenge, and NGA offers no query API. It does
publish its whole collection (130,000+ works, CC0, refreshed daily) as CSV on GitHub
(github.com/NationalGalleryOfArt/opendata), with Wikidata ids reconciled, and its IIIF image server
(api.nga.gov) answers plain requests. So the pipeline keeps a local index:

* the four CSVs it needs (objects, constituents, objects_constituents, published_images; ~218 MB) are
  fetched one at a time into ~/Claude/Staging (bulk-download policy), verified against GitHub's blob
  hashes, and indexed here into SQLite: ``neural-canvas nga-index <csv dir>``;
* a lookup gives an artist's works (by surname) with title, date, medium, attribution, inscription,
  accession number, Wikidata id and the primary IIIF image.

The catalogue stage verifies the picture like any other museum's, and counts NGA in the overrule tier.
"""
from __future__ import annotations

import csv
import logging
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)
ARTIST_ROLES = ("artist", "painter", "draftsman", "printmaker", "sculptor", "photographer", "author")


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def build_index(src: Path, db: Path) -> dict:
    """SQLite index of the NGA open-data CSVs in ``src``; replaces ``db`` atomically."""
    src, db = Path(src), Path(db)
    tmp = db.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        CREATE TABLE works (objectid INTEGER PRIMARY KEY, title TEXT, displaydate TEXT, medium TEXT, attribution TEXT,
                            inscription TEXT, accessionnum TEXT, classification TEXT, wikidataid TEXT);
        CREATE TABLE artists (constituentid INTEGER PRIMARY KEY, name TEXT, lastname TEXT, displaydate TEXT,
                              nationality TEXT, wikidataid TEXT, ulanid TEXT);
        CREATE TABLE work_artists (objectid INTEGER, constituentid INTEGER, role TEXT, prefix TEXT);
        CREATE TABLE images (objectid INTEGER, iiifurl TEXT, viewtype TEXT, sequence INTEGER);""")
    def rows(name):
        with open(src / f"{name}.csv", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    counts = {}
    counts["works"] = con.executemany("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?)", (
        (r["objectid"], r["title"], r["displaydate"], r["medium"], r["attribution"], r["inscription"],
         r["accessionnum"], r["classification"], r["wikidataid"]) for r in rows("objects"))).rowcount
    counts["artists"] = con.executemany("INSERT INTO artists VALUES (?,?,?,?,?,?,?)", (
        (r["constituentid"], r["forwarddisplayname"] or r["preferreddisplayname"], fold(r["lastname"]), r["displaydate"],
         r["nationality"], r["wikidataid"], r["ulanid"]) for r in rows("constituents"))).rowcount
    counts["work_artists"] = con.executemany("INSERT INTO work_artists VALUES (?,?,?,?)", (
        (r["objectid"], r["constituentid"], r["role"], r["prefix"]) for r in rows("objects_constituents"))).rowcount
    counts["images"] = con.executemany("INSERT INTO images VALUES (?,?,?,?)", (
        (r["depictstmsobjectid"], r["iiifurl"], r["viewtype"], int(r["sequence"] or 0)) for r in rows("published_images")
        if r["depictstmsobjectid"])).rowcount
    con.executescript("""CREATE INDEX a_last ON artists(lastname); CREATE INDEX wa_c ON work_artists(constituentid);
                         CREATE INDEX wa_o ON work_artists(objectid); CREATE INDEX i_o ON images(objectid);""")
    con.commit()
    con.close()
    tmp.replace(db)
    log.info("nga index %s: %s", db, counts)
    return counts


class NGA:
    def __init__(self, db: Path):
        self.db = Path(db)

    @property
    def available(self) -> bool:
        return self.db.exists()

    def works_by(self, artist: str) -> list[dict]:
        """Every NGA work whose artist has this surname (artist roles only), with its primary image."""
        sur = fold(artist).split()[-1:] or [""]
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        q = f"""SELECT w.*, a.name AS artist, a.displaydate AS artist_dates, a.nationality, a.wikidataid AS artist_qid,
                       wa.prefix, (SELECT iiifurl FROM images i WHERE i.objectid = w.objectid
                                   ORDER BY viewtype != 'primary', sequence LIMIT 1) AS iiif
                FROM artists a JOIN work_artists wa ON wa.constituentid = a.constituentid
                JOIN works w ON w.objectid = wa.objectid
                WHERE a.lastname = ? AND lower(wa.role) IN ({",".join("?" * len(ARTIST_ROLES))})"""
        out = [dict(r) for r in con.execute(q, (sur[0], *ARTIST_ROLES))]
        con.close()
        first = fold(artist).split()[:1]
        if first:          # several artists share a surname: keep the one whose first name matches, when any does
            same = [r for r in out if fold(r["artist"]).split()[:1] == first]
            out = same or out
        return out

    @staticmethod
    def image(row: dict, size: int = 500) -> str:
        return f"{row['iiif']}/full/!{size},{size}/0/default.jpg" if row.get("iiif") else ""
