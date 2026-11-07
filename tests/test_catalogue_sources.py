"""The catalogue's museum adapters: a source that cannot answer raises, never returns [] (Sir, 2026-11-06).
No network: each test stands a fake reply in for the API."""
import numpy as np
import pytest
from PIL import Image

from neural_canvas.catalogue import Catalogue, SourceError, _copy_after, _one, _title_words, source


def _cat(keys=None, cfg=None):
    cat = Catalogue.__new__(Catalogue)
    cat.consulted, cat.cfg, cat._apikeys, cat._unchecked = [], cfg, dict(keys or {}), None
    cat.t = {"id_clip_min": 0.86, "id_inliers_min": 25}
    return cat


class _Reply:
    def __init__(self, status=200, body=None, ctype="application/json", text=""):
        self.status_code, self._body, self.headers, self.text = status, body, {"content-type": ctype}, text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class _Session:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append((method, url, params, json))
        return self.replies.pop(0)


def test_an_api_reply_that_is_refused_or_malformed_is_an_error_and_the_key_never_shows(monkeypatch):
    monkeypatch.setattr("neural_canvas.catalogue.time.sleep", lambda s: None)
    cat = _cat({"harvard": "SECRETKEY123"})
    cat.s = _Session(_Reply(403, text="forbidden"))
    with pytest.raises(SourceError) as e:
        cat._api("https://api.example/object", "records", apikey="SECRETKEY123")
    assert "403" in str(e.value) and "SECRETKEY123" not in str(e.value)
    cat.s = _Session(_Reply(200, {"error": "bad"}))
    with pytest.raises(SourceError, match="without 'records'"):
        cat._api("https://api.example/object", "records")
    cat.s = _Session(_Reply(200, None, "text/html"))
    with pytest.raises(SourceError, match="not JSON"):
        cat._api("https://api.example/object", "records")
    cat.s = _Session(_Reply(429), _Reply(503), _Reply(200, {"records": []}))     # rate limits are retried, paced
    assert cat._api("https://api.example/object", "records") == {"records": []}
    cat.s = _Session(_Reply(404, text='{"message":"Not a valid object"}'))         # a withdrawn Met object
    assert cat._api("https://api.example/objects/828564", "objectID", gone=(404,)) == {}


def test_a_missing_key_is_an_error_naming_the_secrets_file():
    cat = _cat({}, cfg={"sources": {"api_keys_path": "/x/keys.txt"}})
    with pytest.raises(SourceError, match="no 'europeana' key in /x/keys.txt"):
        cat._europeana(None, None, ["Klimt Litzlberg"], "klimt")


def test_an_adapter_whose_candidates_could_not_be_compared_raises_instead_of_returning_nothing():
    class Cat:
        _unchecked = None

        @source("Test Museum")
        def search(self, unchecked):
            self._unchecked += unchecked
            return []
    with pytest.raises(SourceError, match="Test Museum: no record matched the picture, and 1 candidate"):
        Cat().search(["Met 1 'X' (no image served)"])
    assert Cat().search([]) == []                 # every candidate compared, none matched: a real no-match


def test_an_image_that_will_not_load_is_unchecked_not_a_no_match(monkeypatch):
    monkeypatch.setattr("neural_canvas.catalogue.fetch_image", lambda url, ua: None)    # a rate challenge
    cat = _cat()
    cat._unchecked = []
    assert cat._verify(Image.new("RGB", (8, 8)), None, "https://www.artic.edu/iiif/2/x/full/843,/0/default.jpg",
                       "Art Institute 59798") == (False, 0.0, 0)
    assert cat._unchecked == ["Art Institute 59798 (image would not load)"]


def test_rijksmuseum_reads_a_printmaker_named_only_in_a_statement_and_a_listed_timespan():
    en = [{"id": "http://vocab.getty.edu/aat/300388277"}]
    part = lambda pid, text: {"assigned_by": [{"assigned": [{"id": pid}], "assigned_property": "carried_out_by"}],
                              "referred_to_by": [{"content": text, "language": en}]}
    x = {"id": "https://id.rijksmuseum.nl/200136156",
         "identified_by": [{"type": "Name", "content": "Sneeuw te Ochanomizu"}],
         "produced_by": {"part": [part("p1", "publisher: Fukuda Kumajirô (mentioned on object), Tokyo"),
                                  part("p2", "printmaker: Kobayashi Kiyochika (mentioned on object), Japan")],
                         "timespan": [{"identified_by": [{"content": "1877 - 1882", "language": en}]}]},
         "subject_of": [{"digitally_carried_by": [{"access_point": [{"id": "https://iiif.micr.io/a/full/max/0/default.jpg"}]}]}]}
    r = Catalogue._rijks_rec(x, "kiyochika")
    assert r.artist == "Kobayashi Kiyochika" and r.date == "1877 - 1882"
    assert Catalogue._rijks_rec(x).artist == "Kobayashi Kiyochika"          # the publisher is never the artist
    assert _one([{"a": 1}]) == {"a": 1} and _one(None) == {} and _one({"b": 2}) == {"b": 2}


def test_the_met_search_pages_past_its_first_hundred_ids():
    cat = _cat()
    pages = {0: {"total": 569, "objectIDs": list(range(500))}, 500: {"total": 569, "objectIDs": list(range(500, 569))}}
    seen = []
    cat._api = lambda url, expect, **p: seen.append(p) or pages[p["offset"]]
    assert cat._met_ids("James McNeill Whistler", artistOrCulture="true") == list(range(569))
    assert [p["limit"] for p in seen] == [500, 500] and seen[0]["artistOrCulture"] == "true"
    cat._api = lambda url, expect, **p: {"total": 0, "objectIDs": None}
    assert cat._met_ids("zzqq") == []


def test_smithsonian_asks_for_the_surname_and_any_title_word_inside_its_unit_filter(monkeypatch):
    cat = _cat({"smithsonian": "K" * 10})
    asked = []
    cat._api = lambda url, expect, **p: asked.append(p["q"]) or {"response": {"rows": []}}
    assert cat._smithsonian(None, None, ["James McNeill Whistler The Riva—Sunset: Red and Gold"], "whistler",
                            ["The Riva—Sunset: Red and Gold"]) == []
    assert asked[0].endswith("AND whistler AND (riva OR sunset OR red OR gold)") and asked[1].endswith("AND whistler")
    assert _title_words("The Riva—Sunset: Red and Gold", "whistler") == ["riva", "sunset", "red", "gold"]


def test_europeana_judges_a_record_that_names_its_creator_only_by_uri_on_its_picture(monkeypatch):
    cat = _cat({"europeana": "K" * 10})
    items = [{"id": "/a", "dcCreator": ["http://viaf.org/viaf/54156267"], "edmIsShownBy": ["https://x/ghent.jpg"],
              "title": ["The Ghent Altarpiece"], "dataProvider": ["Saint Bavo Cathedral"]},
             {"id": "/b", "dcCreator": ["Someone Else"], "edmIsShownBy": ["https://x/other.jpg"], "title": ["Other"]}]
    queries = []
    cat._api = lambda url, expect, **p: queries.append(p["query"]) or {"success": True, "items": items}
    monkeypatch.setattr("neural_canvas.catalogue.fetch_image", lambda url, ua: Image.new("RGB", (8, 8)))
    tried = []
    cat._verify = lambda img, vec, url, what="": tried.append(url) or (url.endswith("ghent.jpg"), 0.9, 300)
    [r] = cat._europeana(None, None, ["Jan van Eyck Adoration of the Lamb"], "eyck", ["Adoration of the Lamb"])
    assert r.title == "The Ghent Altarpiece" and tried == ["https://x/ghent.jpg"]
    assert queries[0] == "who:eyck AND (adoration OR lamb)"


def test_nga_without_its_index_is_an_error_not_an_empty_list(tmp_path):
    cat = _cat(cfg={"sources": {"nga_db": str(tmp_path / "missing.sqlite")}})
    with pytest.raises(SourceError, match="no open-data index"):
        cat._nga(None, np.zeros(4), ["Five Butterflies"], "Odilon Redon")


def test_europeana_skips_a_copy_after_the_artist_and_falls_back_to_its_own_thumbnail(monkeypatch):
    assert _copy_after("The adoration of the lamb. Chromolithograph after Jan van Eyck.", "eyck")
    assert _copy_after("Am Attersee nach Gustav Klimt, Tafel 29", "klimt")
    assert not _copy_after("Litzlberg on the Attersee", "klimt") and not _copy_after("After the Bath", "degas")
    cat = _cat({"europeana": "K" * 10})
    items = [{"id": "/copy", "dcCreator": ["Jan van Eyck"], "edmIsShownBy": ["https://w/copy.jpg"],
              "title": ["The adoration of the lamb. Chromolithograph after Jan van Ey"],     # Europeana cuts titles
              "dcDescription": ['"The adoration of the lamb" after Jan van Eyck. The lamb ...']},
             {"id": "/belvedere", "dcCreator": ["Jan van Eyck"], "edmIsShownBy": ["https://belvedere/403"],
              "edmPreview": ["https://api.europeana.eu/thumbnail/v2/x"], "title": ["Genter Altar"]}]
    cat._api = lambda url, expect, **p: {"success": True, "items": items}
    monkeypatch.setattr("neural_canvas.catalogue.fetch_image",
                        lambda url, ua: None if "403" in url else Image.new("RGB", (8, 8)))
    tried = []
    cat._verify = lambda img, vec, url, what="": tried.append(url) or (True, 0.9, 300)
    [r] = cat._europeana(None, None, ["Jan van Eyck Ghent"], "eyck", ["Ghent"])
    assert r.title == "Genter Altar" and tried == ["https://api.europeana.eu/thumbnail/v2/x"]


def test_art_institute_images_are_fetched_with_the_header_its_image_server_asks_for():
    from neural_canvas.matching import _host_headers
    assert _host_headers("www.artic.edu") == ("AIC-User-Agent",) and _host_headers("artic.edu") == ("AIC-User-Agent",)
    assert _host_headers("upload.wikimedia.org") == ()


def test_an_art_institute_image_is_asked_for_at_an_allowed_width_no_wider_than_the_original():
    rec = lambda w: Catalogue._artic_rec({"id": 4, "title": "Priest and Boy", "image_id": "abc",
                                          "thumbnail": {"width": w} if w else None}, "image").image_url
    assert rec(1596).endswith("/abc/full/843,/0/default.jpg") and rec(None).endswith("/full/843,/0/default.jpg")
    assert rec(557).endswith("/abc/full/400,/0/default.jpg") and rec(300).endswith("/full/200,/0/default.jpg")
