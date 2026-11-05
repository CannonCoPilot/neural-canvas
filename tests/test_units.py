"""Unit tests for the deterministic parts of the pipeline (no models, no network)."""
import numpy as np
import pytest
from PIL import Image

from neural_canvas.crop import Composer, _retained
from neural_canvas.evaluate import score
from neural_canvas.interpret import Interpretation, Subject, _box
from neural_canvas.placard import contrast_ratio
from neural_canvas.vlm import _extract_json


def _img(w, h):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _interp(subjects, genre="genre-scene", faces=(), focal=(0.5, 0.5)):
    return Interpretation(genre, "", "", focal, None, subjects, "", list(faces))


def _is_16x9_full_bleed(plan):
    x0, y0, x1, y1 = plan.window_px
    return plan.mode == "full-bleed" and plan.art_size == (3840, 2160) and abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.01


def test_the_window_keeps_the_short_side_whole_and_is_16x9():
    for w, h in ((2000, 1500), (1000, 1500), (4000, 1000)):
        plan = Composer(3840, 2160, 0.97).plan(_img(w, h), _interp([Subject("thing", (0.4, 0.4, 0.6, 0.6), 3, True)]))
        assert plan.passed and _is_16x9_full_bleed(plan)
        x0, y0, x1, y1 = plan.window_px
        assert (x1 - x0 == w) if w / h < 16 / 9 else (y1 - y0 == h)


def test_the_window_centres_on_the_importance_weighted_centroid():
    subs = [Subject("big", (0.60, 0.4, 0.66, 0.6), 4, True), Subject("small", (0.70, 0.4, 0.74, 0.6), 1, False)]
    plan = Composer(3840, 2160, 0.97).plan(_img(4000, 1000), _interp(subs))
    cx = (4 * 1.5 * 0.63 + 1 * 0.72) / 7                       # essential subjects count 1.5x
    assert abs(plan.centroid[0] - cx) < 1e-3 and abs((plan.window[0] + plan.window[2]) / 2 - cx) < 0.01
    assert not plan.detail and plan.costs["cut"] == 0


def test_a_tall_figure_keeps_head_and_body_and_gives_up_the_sky():
    # Summer Morning: a woman standing in a meadow under trees, the canvas far taller than the screen
    img = _img(1035, 1535)
    subs = [Subject("young woman", (0.445, 0.24, 0.855, 0.84), 5, True, "figure"),
            Subject("sack of greenery", (0.38, 0.5, 0.6, 0.685), 4, False, "object"),
            Subject("meadow", (0.0, 0.18, 1.0, 1.0), 3, False, "region"),
            Subject("trees", (0.0, 0.0, 1.0, 0.44), 3, False, "region"),
            Subject("sunlight", (0.0, 0.0, 1.0, 0.3), 2, False, "region")]
    face = Subject("young woman", (0.605, 0.25, 0.695, 0.335), 5, True, "head")
    plan = Composer(3840, 2160, 0.97).plan(img, _interp(subs, faces=[face]))
    y0, y1 = plan.window[1], plan.window[3]
    assert y0 <= 0.25 - 0.04 * (y1 - y0) and y1 > 0.6         # her head with headroom, cut below the hands
    assert plan.faces_kept == "1/1"
    assert plan.detail and "young woman" in plan.detail_reasons[0]   # her feet cannot fit: a detail, said so


def test_principal_faces_that_cannot_share_a_window_make_a_detail_not_a_block():
    faces = [Subject("a", (0.4, 0.05, 0.5, 0.12), 5, True, "head"), Subject("b", (0.4, 0.85, 0.5, 0.92), 5, True, "head")]
    plan = Composer(3840, 2160, 0.97).plan(_img(1000, 2000), _interp([Subject("a", (0.3, 0.0, 0.7, 1.0), 5, True, "figure")],
                                                                      faces=faces))
    assert plan.passed and plan.detail and _is_16x9_full_bleed(plan)
    assert any("principal head" in w for w in plan.warnings)
    assert plan.faces_kept == "1/2"                             # one head whole, not two halves


def test_a_crowd_marked_essential_still_keeps_the_principal_face():
    subs = [Subject("Jesus", (0.07, 0.54, 0.83, 0.99), 5, True, "figure"),
            Subject("angels", (0.0, 0.0, 1.0, 0.85), 4, True, "figure")]
    faces = [Subject("Jesus", (0.3, 0.62, 0.38, 0.72), 5, True, "head"), Subject("angels", (0.5, 0.02, 0.56, 0.08), 5, True, "head")]
    plan = Composer(3840, 2160, 0.97).plan(_img(2326, 1586), _interp(subs, faces=faces))
    assert plan.window[1] <= 0.62 and plan.window[3] >= 0.72   # Jesus's face kept


def test_a_wide_panorama_keeps_the_more_important_subject():
    subs = [Subject("minor", (0.02, 0.3, 0.1, 0.7), 2, False), Subject("main", (0.8, 0.3, 0.95, 0.7), 5, True)]
    plan = Composer(3840, 2160, 0.97).plan(_img(4000, 1000), _interp(subs))
    assert _retained(subs[1].box, plan.window) == 1.0 and not plan.detail


def test_a_landscape_keeps_its_horizon_off_the_centre_line():
    subs = [Subject("sky", (0.0, 0.0, 1.0, 0.5), 2, False, "region"), Subject("fields", (0.0, 0.5, 1.0, 1.0), 2, False, "region")]
    it = Interpretation("landscape", "", "", (0.5, 0.5), 0.5, subs, "", [])
    plan = Composer(3840, 2160, 0.97).plan(_img(1600, 1600), it)
    hz = (0.5 - plan.window[1]) / (plan.window[3] - plan.window[1])
    assert abs(hz - 0.5) >= 1 / 6 - 0.01 and plan.window[1] < 1 - plan.window[3]   # off centre; sky kept


def test_the_crop_keeps_faces_out_of_the_label_corner():
    img = _img(1600, 1200)                  # 4:3: the window slides vertically over 25% of the height
    face = Subject("man", (0.88, 0.70, 0.95, 0.78), 5, True, "head")
    plan = Composer(3840, 2160, 0.97).plan(img, _interp([Subject("man", (0.8, 0.6, 1.0, 1.0), 5, True, "figure")], faces=[face]))
    win = plan.window
    fy1 = (face.box[3] - win[1]) / (win[3] - win[1])
    assert fy1 < 0.894                                         # the face ends above the card


def _noise_art(w, h, seed=0):
    return np.random.default_rng(seed).integers(40, 220, (h, w, 3), dtype=np.uint8)


def test_a_caption_band_is_trimmed_with_its_lettering():
    from neural_canvas.borders import trim_borders
    a = np.zeros((1750, 1000, 3), dtype=np.uint8)
    a[:1518] = _noise_art(1000, 1518)
    for x in range(300, 740, 40):                              # white "letters" inside the black band
        a[1600:1640, x:x + 22] = 255
    img, rec = trim_borders(Image.fromarray(a))
    assert rec["cut"] == {"top": 0, "bottom": 232, "left": 0, "right": 0} and img.size == (1000, 1518)


def test_a_painted_ground_is_not_a_border():
    from neural_canvas.borders import trim_borders
    a = np.full((1200, 900, 3), 250, dtype=np.uint8)           # a vignette: a figure drawn on white paper
    rng = np.random.default_rng(2)
    for y in range(80, 1150):
        x0 = int(200 + 150 * np.sin(y / 60) + rng.integers(0, 40)); x1 = int(700 - 120 * np.cos(y / 45))
        a[y, x0:x1] = rng.integers(30, 200, (x1 - x0, 3))
    a[30:80, 420:470] = 90                                     # a hat above the figure
    _, rec = trim_borders(Image.fromarray(a))
    assert rec["cut"] == {"top": 0, "bottom": 0, "left": 0, "right": 0}


def test_a_coloured_edge_is_part_of_the_art():
    from neural_canvas.borders import trim_borders
    a = _noise_art(800, 600)
    a[:60] = (200, 40, 40)                                     # a red band painted across the top
    _, rec = trim_borders(Image.fromarray(a))
    assert rec["cut"]["top"] == 0


def test_a_detail_is_named_on_the_card():
    from neural_canvas.placard import _layout
    lines, fails = _layout({"artist": "Emile Claus", "title": "Summer Morning", "date": "c. 1891", "detail": True}, 0.3, 221)
    assert "Summer Morning, c. 1891 (detail)" in [t for t, _, _ in lines] and not fails


def test_box_normalisation_and_rejects():
    assert _box([0, 0, 500, 1000]) == (0.0, 0.0, 0.5, 1.0)
    assert _box([600, 100, 200, 300]) == (0.2, 0.1, 0.6, 0.3)     # swapped corners
    assert _box([1, 1, 2, 2]) is None                              # degenerate
    assert _box("nonsense") is None


def test_json_extraction():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('noise {"a": {"b": 2}} tail') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        _extract_json("no object here")


def test_contrast_ratio_extremes():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0, rel=1e-3)
    assert contrast_ratio((29, 28, 26), (244, 241, 234)) > 7


@pytest.mark.parametrize("ident,ok", [
    ({"artist": "Émile Claus", "title": "Summer Morning (Matin d'été)", "date": "1891"}, True),
    ({"artist": "Emile Claus", "title": "Winter Evening", "date": "1891"}, False),
    ({"artist": "Paul Gauguin", "title": "The Field of Derout-Lollichon", "date": "1888"}, True),
    ({"artist": "Paul Gauguin", "title": "The Field of Derout-Lollichon", "date": "1880"}, False),
])
def test_gold_scoring(ident, ok):
    gold = {"artist": "Emile Claus", "title": "Summer Morning", "year": 1891} if "Claus" in ident["artist"] \
        else {"artist": "Paul Gauguin", "title": "The Field of Derout-Lollichon", "year": 1886}
    s = score(gold, ident)
    assert (s["artist_ok"] and s["title_ok"] and s["year_ok"]) is ok


def test_gold_year_range():
    s = score({"artist": "James Tissot", "title": "Jesus Ministered to by Angels", "year": [1886, 1894]},
              {"artist": "James Tissot", "title": "Jesus Ministered to by Angels", "date": "1890"})
    assert s["year_ok"]


def test_web_evidence_line_names_the_image_file():
    from neural_canvas.identify import _web_line
    from neural_canvas.websearch import WebReference
    r = WebReference(file_title="San Diego Sunset — Jim Musil Painter", page_id=1, thumb_url="",
                     page_url="https://jimmusil.com/prints/p/san-diego-sunset",
                     image_url="https://images.example-cdn.com/content/v1/abc/Towering+Storm.jpeg?format=500w")
    line = _web_line(r)
    assert "file: Towering Storm" in line and "page: San Diego Sunset" in line and "jimmusil.com" in line


def test_the_label_never_covers_a_face():
    from neural_canvas.crop import CropPlan
    from neural_canvas.placard import compose
    plan = CropPlan("full-bleed", (0, 0, 1, 1), (0, 0, 3840, 2160), (3840, 2160), (0, 0), passed=True)
    ident = {"artist": "A. Painter", "title": "Crowd", "date": "1900"}
    _, ok = compose(_img(3840, 2160), plan, (3840, 2160), ident, [(0.0, 0.0, 1.0, 1.0)], 7.0, faces=[(0.1, 0.1, 0.2, 0.2)])
    assert ok.passed and ok.position == "bottom-right" and ok.essential_overlap > 0       # a body under it is recorded
    _, bad = compose(_img(3840, 2160), plan, (3840, 2160), ident, [], 7.0, faces=[(0.9, 0.9, 0.95, 0.95)])
    assert not bad.passed and any("face" in f for f in bad.failures)


def test_placard_is_facts_only_fixed_half_size_and_bottom_right():
    from neural_canvas.crop import CropPlan
    from neural_canvas.placard import card_size, compose
    art = _img(3840, 2160)
    plan = CropPlan("full-bleed", (0, 0, 1, 1), (0, 0, 3840, 2160), (3840, 2160), (0, 0), passed=True)
    ident = {"artist": "Paul Gauguin", "artist_nationality": "France", "artist_born": "1848", "artist_died": "1903",
             "title": "The Field of Derout-Lollichon", "date": "1886", "medium": "Oil paint on canvas",
             "movement": "Impressionism", "collection": "Los Angeles County Museum of Art"}
    _, pl = compose(art, plan, (3840, 2160), ident, [(0.0, 0.0, 0.3, 0.3)], 7.0, scale=0.5)
    assert pl.position == "bottom-right" and pl.passed, pl.failures
    text = [s for s in pl.lines if s]
    assert text[-2:] == ["Impressionism", "Los Angeles County Museum of Art"]   # movement right above it
    assert len(text) == 6 and "The Field of Derout-Lollichon, 1886" in text   # no paragraph; title one line
    x0, y0, x1, y1 = pl.box
    assert (x1, y1) == (3840 - 40, 2160 - 40) and (x1 - x0, y1 - y0) == card_size(0.5) == (411, 190)   # fixed, half of 822


def test_wikidata_movement_labels_print_as_names():
    from neural_canvas.movement import display_name
    assert display_name("Post-impressionism") == "Post-Impressionism"
    assert display_name("synthetism") == "Synthetism"
    assert display_name("Hudson River school") == "Hudson River School"
    assert display_name("Golden Age of Danish painting") == "Golden Age of Danish Painting"


def test_imify_names_an_untitled_hit_by_its_page_url():
    from neural_canvas.imify import _slug_title
    assert _slug_title("https://barbarapicci.com/2021/11/19/fotogallery-54/tomioka-soichiro/") == \
        "fotogallery 54 / tomioka soichiro"


def test_vision_web_pages_come_first_and_similar_images_never_count():
    from neural_canvas.vision_web import VisionWebDetection
    web = {"pagesWithMatchingImages": [{"url": "https://a.org/p", "pageTitle": "Trees (1961) by Tomioka",
                                        "fullMatchingImages": [{"url": "https://a.org/t.jpg"}]}],
           "fullMatchingImages": [{"url": "https://a.org/t.jpg"}, {"url": "https://b.org/t2.jpg"}],
           "visuallySimilarImages": [{"url": "https://c.org/other.jpg"}],
           "bestGuessLabels": [{"label": "tomioka soichiro trees 1961"}],
           "webEntities": [{"description": "Painting", "score": 0.4}, {"description": "Sōichirō Tomioka", "score": 0.9}]}
    refs = VisionWebDetection.references(web)
    assert [r.image_url for r in refs] == ["https://a.org/t.jpg", "https://b.org/t2.jpg"]
    assert refs[0].file_title == "Trees (1961) by Tomioka" and refs[0].page_url == "https://a.org/p"
    h = VisionWebDetection.hints(web)
    assert h["best_guess"] == ["tomioka soichiro trees 1961"] and h["entities"][0] == "Sōichirō Tomioka"


def test_vision_key_is_found_in_any_document_of_a_multi_doc_yaml(tmp_path):
    from neural_canvas.vision_web import _read_key
    f = tmp_path / "creds.yaml"
    f.write_text("github:\n  token: x\n---\nimage:\n  google_vision: ' k123 '\n")
    assert _read_key(str(f), "image.google_vision") == "k123"
    with pytest.raises(KeyError):
        _read_key(str(f), "image.missing")


def test_web_search_interleaves_engines_and_survives_one_failing(monkeypatch):
    from neural_canvas.websearch import WebImageSearch, WebReference
    ref = lambda u: WebReference(u, hash(u), u, u, u)
    w = WebImageSearch(searxng_url="http://searx.invalid")
    monkeypatch.setattr(w, "_ddg", lambda q, n: [ref("d1"), ref("same"), ref("d2")])
    monkeypatch.setattr(w, "_searx", lambda q, n: [ref("s1"), ref("same")])
    assert [r.image_url for r in w.search("q", 3)] == ["d1", "s1", "same", "d2"]
    monkeypatch.setattr(w, "_searx", lambda q, n: (_ for _ in ()).throw(RuntimeError("down")))
    assert [r.image_url for r in w.search("q", 3)] == ["d1", "same", "d2"]
    monkeypatch.setattr(w, "_ddg", lambda q, n: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        w.search("q", 3)


def _painting(seed):
    """Smooth random structure, like a painting (pure noise has no low frequencies to hash)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((1600, 1200), Image.Resampling.BICUBIC)


def test_phash_matches_resized_recompressed_copies_and_separates_different_works():
    import io
    from neural_canvas.dedup import PHASH_NEAR, hamming, phash
    a = _painting(1)
    buf = io.BytesIO()
    a.resize((640, 480)).save(buf, "JPEG", quality=60)
    copy = Image.open(io.BytesIO(buf.getvalue()))
    assert hamming(phash(a), phash(copy)) <= 4
    assert min(hamming(phash(a), phash(_painting(s))) for s in range(2, 12)) > PHASH_NEAR


def test_work_key_folds_case_accents_articles_and_initials():
    from neural_canvas.dedup import work_key
    assert work_key("Andrew Wyeth", "The Carry") == work_key("A. WYETH", "carry") == "wyeth|carry"
    assert work_key("Paul Cézanne", "Abandoned House") == work_key("Paul Cezanne", "abandoned house")
    assert work_key("", "Untitled") == ""


def test_registry_flags_only_delivered_twins_and_never_undelivers(tmp_path):
    from neural_canvas.dedup import ArtworkRegistry
    r = ArtworkRegistry(tmp_path)
    r.record("run1", phash=123, artist="Carl Bloch", title="Transfiguration of Jesus", status="identified")
    assert r.delivered_twin("C. Bloch", "transfiguration of jesus", exclude="run2") is None
    r.record("run1", status="delivered")
    assert r.delivered_twin("C. Bloch", "transfiguration of jesus", exclude="run2")[0] == "run1"
    assert r.delivered_twin("C. Bloch", "transfiguration of jesus", exclude="run1") is None   # itself
    r.record("run1", status="identified")                       # a resumed run
    assert ArtworkRegistry(tmp_path).works["run1"]["status"] == "delivered"
    assert ArtworkRegistry(tmp_path).near(123 ^ 0b111)[0] == "run1"


def test_generic_titles_never_make_two_works_the_same():
    from neural_canvas.dedup import work_key
    for t in ("Untitled", "Composition No. 7", "Self-Portrait", "Still Life", "Improvisation 31", "Study"):
        assert work_key("Mark Rothko", t) == "", t
    assert work_key("Vincent van Gogh", "Irises") == "gogh|irises"


def test_vision_key_never_reaches_an_error_message(tmp_path, monkeypatch):
    import requests
    from neural_canvas import vision_web
    creds = tmp_path / "c.yaml"
    creds.write_text("image:\n  google_vision: SECRET12345\n")
    v = vision_web.VisionWebDetection({"credentials_path": str(creds), "key_path": "image.google_vision"}, tmp_path / "cache")

    def boom(url, **kw):
        assert "SECRET12345" not in url and kw["headers"]["X-goog-api-key"] == "SECRET12345"
        raise requests.ConnectionError(f"Max retries exceeded with url: /v1/images:annotate?key=SECRET12345 (refused)")
    monkeypatch.setattr(vision_web.requests, "post", boom)
    with pytest.raises(RuntimeError) as e:
        v.detect(_img(64, 64))
    assert "SECRET12345" not in str(e.value) and "<redacted>" in str(e.value)


def test_delivery_check_names_a_missing_permission(tmp_path):
    import json
    from neural_canvas.deliver import GooglePhotos, SCOPES
    (tmp_path / "client.json").write_text("{}")
    tok = tmp_path / "token.json"
    tok.write_text(json.dumps({"scopes": SCOPES[:2], "refresh_token": "r", "token": "t"}))
    g = GooglePhotos({"oauth_client_path": str(tmp_path / "client.json"), "token_path": str(tok),
                      "album_name": "."}, tmp_path)
    assert "edit.appcreateddata" in g.check() and "auth-photos" in g.check()


def test_runner_pauses_without_claiming_when_delivery_cannot_sign_in(tmp_path, monkeypatch):
    from neural_canvas import service as svc
    from neural_canvas.config import DEFAULTS, Config, _merge
    cfg = Config(_merge(DEFAULTS, {"paths": {k: str(tmp_path / k) for k in ("watch_dir", "work_dir", "output_dir", "state_dir")}}))
    s = svc.Service(cfg)
    alerts, claims = [], []
    monkeypatch.setattr(s.pulse, "list", lambda label: [{"id": t, "labels": ["staging:wait"], "metadata": {"run_id": t}}
                                                       for t in ("t1", "t2")])
    monkeypatch.setattr(s.pulse, "claim", lambda tid, label: claims.append(tid) or False)
    monkeypatch.setattr(s.pulse, "alert", lambda sev, text, data: alerts.append((sev, data)))
    monkeypatch.setattr(s.photos, "check", lambda: "Google Photos sign-in expired: re-run auth-photos.")
    assert s.run_ready() == 0 and claims == []
    assert alerts == [("critical", {"waiting": 2})]
    s.run_ready()
    assert len(alerts) == 1                     # one alert an hour, not one per poll


def test_wikidata_dates_print_at_their_own_precision():
    from neural_canvas.references import Wikimedia

    def ent(time, precision, **quals):
        q = {k: [{"datavalue": {"value": v}}] for k, v in quals.items()}
        return {"claims": {"P571": [{"mainsnak": {"datavalue": {"value": {"time": time, "precision": precision}}},
                                     "qualifiers": q}]}}
    y = Wikimedia.claim_year
    assert y(ent("+1886-00-00T00:00:00Z", 9), "P571") == "1886"
    assert y(ent("+1862-01-01T00:00:00Z", 8), "P571") == "1860s"                    # Moran, not "1862"
    assert y(ent("+1800-00-00T00:00:00Z", 8), "P571") == "1800s"                    # Bloch, not "1800"
    assert y(ent("+1890-00-00T00:00:00Z", 7), "P571") == "19th century"
    assert y(ent("+1890-00-00T00:00:00Z", 7, P1319={"time": "+1886-00-00T00:00:00Z"},
                 P1326={"time": "+1894-00-00T00:00:00Z"}), "P571") == "1886–1894"     # Tissot, not "1890"
    assert y(ent("+1862-00-00T00:00:00Z", 9, P1480={"id": "Q5727902"}), "P571") == "c. 1862"


def _card(ident, **kw):
    from neural_canvas.crop import CropPlan
    from neural_canvas.placard import compose
    plan = CropPlan("full-bleed", (0, 0, 1, 1), (0, 0, 3840, 2160), (3840, 2160), (0, 0), passed=True)
    return compose(_img(3840, 2160), plan, (3840, 2160), ident, [(0.0, 0.0, 0.3, 0.3)], 7.0, scale=0.5, **kw)[1]


def test_every_card_is_the_same_size_and_keeps_title_and_date_on_one_line():
    short = _card({"artist": "Jim Musil", "title": "Towering Storm", "date": "2025"})
    full = _card({"artist": "James Tissot", "artist_nationality": "France", "artist_born": "1836",
                  "artist_died": "1902", "title": "The Miracle of the Loaves and Fishes", "date": "1886–1894",
                  "medium": "Opaque watercolor over graphite", "movement": "Academic Art",
                  "collection": "Brooklyn Museum"})
    for pl in (short, full):
        x0, y0, x1, y1 = pl.box
        assert (x1 - x0, y1 - y0) == (411, 190) and pl.passed, pl.failures
    assert "The Miracle of the Loaves and Fishes, 1886–1894" in full.lines     # one line


def test_a_long_title_shrinks_then_wraps_and_an_impossible_one_fails():
    mid = _card({"artist": "Alphonse Mucha", "title": "Slav Epic: The Apotheosis of the Slavs", "date": "1926"})
    assert mid.passed and "Slav Epic: The Apotheosis of the Slavs, 1926" in mid.lines   # set smaller, one line
    long = _card({"artist": "Alfons Mucha", "title": "Slavs in their Original Homeland: Between the Turanian Whip "
                  "and the Sword of the Goths", "date": "1912"})
    assert long.passed and sum(1 for s in long.lines if "Slavs" in s or "Goths" in s) == 2 and long.box[3] - long.box[1] == 190
    huge = _card({"artist": "A", "title": "word " * 60, "date": "1900"})
    assert not huge.passed and any("too long" in f for f in huge.failures)


def test_a_missing_card_texture_is_an_error_not_a_flat_card(tmp_path):
    with pytest.raises(FileNotFoundError):
        _card({"artist": "A", "title": "B"}, texture=str(tmp_path / "nope.jpg"))


# Pulse's pipeline dimensions (alfred/pulse/app.py PIPELINE_DIMENSIONS): any other value is a 400.
PULSE_DIMENSIONS = {"staging": {"wait", "processing", "done", "blocked"}, "evaluated": {"no", "processing", "done", "blocked"},
                    "queued": {"no", "done", "blocked"}, "active": {"no", "claiming", "running", "done", "blocked"},
                    "completed": {"no", "reviewing", "done"}, "blocked": {"no", "yes", "diagnosing"}}


def _service(tmp_path):
    from neural_canvas import service as svc
    from neural_canvas.config import DEFAULTS, Config, _merge
    paths = {k: tmp_path / k for k in ("watch_dir", "work_dir", "output_dir", "state_dir")}
    for p in paths.values():
        p.mkdir()
    cfg = Config(_merge(DEFAULTS, {"paths": {k: str(v) for k, v in paths.items()}}))
    return svc, svc.Service(cfg), paths["watch_dir"]


def _pulse_labels_valid(labels):
    for lbl in labels:
        dim, _, val = lbl.partition(":")
        assert dim not in PULSE_DIMENSIONS or val in PULSE_DIMENSIONS[dim], lbl


def test_a_failed_task_create_leaves_the_file_in_the_inbox(tmp_path, monkeypatch):
    import requests
    svc, s, watch = _service(tmp_path)
    _img(64, 48).save(watch / "work.png")
    alerts = []

    def refuse(**kw):
        raise requests.HTTPError("400 POST /tasks: Invalid value")
    monkeypatch.setattr(s.pulse, "create", refuse)
    monkeypatch.setattr(s.pulse, "alert", lambda sev, text, data: alerts.append(text))
    s.scan(); s.scan(); s.scan()                    # first look sees the size; then fails; then waits
    assert (watch / "work.png").exists() and not list((watch / "ingested").iterdir())
    assert len(alerts) == 1 and s.known == {}
    made = []
    monkeypatch.setattr(s.pulse, "create", lambda **kw: made.append(kw) or {"id": "AION-1"})
    monkeypatch.setattr(svc.time, "time", lambda: 1e12)     # past the retry wait
    assert s.scan() == ["AION-1"]
    assert not (watch / "work.png").exists() and (watch / "ingested" / "work.png").exists()
    _pulse_labels_valid(made[0]["labels"])


def test_a_duplicate_notice_is_valid_on_pulse_and_never_runnable(tmp_path, monkeypatch):
    from neural_canvas.dedup import phash
    svc, s, watch = _service(tmp_path)
    img = _img(64, 48)
    img.save(watch / "again.png")
    s.registry.record("r-delivered", phash=phash(img), status="delivered", original_name="first.png")
    made, closed = [], []
    monkeypatch.setattr(s.pulse, "create", lambda **kw: made.append(kw) or {"id": "AION-d"})
    monkeypatch.setattr(s.pulse, "close", lambda tid, reason: closed.append(tid))
    s.scan(); s.scan()
    labels = made[0]["labels"]
    _pulse_labels_valid(labels)
    assert "outcome:duplicate" in labels and "staging:done" in labels and "staging:wait" not in labels
    assert closed == ["AION-d"] and (watch / "duplicates" / "again.png").exists()
    # a task without a run is never claimed, whatever its labels say
    claims = []
    monkeypatch.setattr(s.pulse, "list", lambda label: [{"id": "AION-d", "labels": ["staging:wait"], "metadata": {}}])
    monkeypatch.setattr(s.pulse, "claim", lambda tid, label: claims.append(tid) or False)
    monkeypatch.setattr(s.photos, "check", lambda: "")
    assert s.run_ready() == 0 and claims == []


def _paper(tmp_path, w=2000, h=2000):
    rng = np.random.default_rng(1)
    p = tmp_path / "paper.png"
    Image.fromarray(rng.integers(200, 255, (h, w, 3), dtype=np.uint8)).save(p)
    return str(p)


IDENT = {"artist": "Paul Gauguin", "artist_nationality": "France", "artist_born": "1848", "artist_died": "1903",
         "title": "The Field of Derout-Lollichon", "date": "1886", "medium": "Oil paint on canvas",
         "movement": "Impressionism", "collection": "Los Angeles County Museum of Art"}


def test_a_card_drawn_for_a_1080_viewer_halves_back_exactly(tmp_path):
    from neural_canvas.placard import compose, viewer_grid
    grid = viewer_grid(2160, 1080)
    assert grid == 2
    art = Image.new("RGB", (3840, 2160), (90, 110, 80))
    frame, pl = compose(art, None, (3840, 2160), IDENT, [], 4.5, 0.5, _paper(tmp_path), grain=0.47, grid=grid)
    assert pl.passed, pl.failures
    x0, y0, x1, y1 = pl.box
    assert all(v % 2 == 0 for v in pl.box)              # the card sits on the viewer's pixel grid
    card = np.asarray(frame.crop(pl.box), dtype=int)
    small = card[::2, ::2]
    for dy in (0, 1):
        for dx in (0, 1):
            assert (card[dy::2, dx::2] == small).all()   # every viewer pixel is a solid 2x2 block
    # so a 2:1 box reduction of the frame returns the card exactly as drawn
    halved = np.asarray(frame.resize((1920, 1080), Image.BOX), dtype=int)[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    assert np.abs(halved - small).max() <= 1


def test_a_paper_photo_too_small_for_the_card_is_refused(tmp_path):
    from neural_canvas.placard import compose
    art = Image.new("RGB", (3840, 2160), (90, 110, 80))
    with pytest.raises(ValueError, match="smaller than the card"):
        compose(art, None, (3840, 2160), IDENT, [], 4.5, 0.5, _paper(tmp_path, 600, 600), grain=0.47, grid=2)


def test_caption_facts_that_cannot_be_right_are_blanked_with_a_reason():
    from types import SimpleNamespace
    from neural_canvas.identify import check_facts
    tomioka = SimpleNamespace(artist_born="1922", artist_died="1994", date="1922-1994", medium="")
    assert "artist's life" in check_facts(tomioka)["date"] and tomioka.date == ""
    lovell = SimpleNamespace(artist_born="1909", artist_died="1997", date="1943", medium="Oil on")
    r = check_facts(lovell)
    assert list(r) == ["medium"] and lovell.medium == "" and lovell.date == "1943"
    tissot = SimpleNamespace(artist_born="1836", artist_died="1902", date="1886–1894", medium="Oil paint on canvas")
    assert check_facts(tissot) == {} and tissot.date == "1886–1894"
    later = SimpleNamespace(artist_born="1836", artist_died="1902", date="1921", medium="Oil on canvas")
    assert "outside" in check_facts(later)["date"]
    living = SimpleNamespace(artist_born="", artist_died="", date="2025", medium="Acrylic")
    assert check_facts(living) == {}


def test_a_small_cut_to_a_secondary_subject_is_a_warning_not_a_detail():
    subs = [Subject("main", (0.3, 0.2, 0.6, 0.7), 5, True, "figure"), Subject("side", (0.05, 0.1, 0.25, 0.9), 3, True, "figure")]
    plan = Composer(3840, 2160, 0.97, detail_min=0.75).plan(_img(1000, 600), _interp(subs))
    assert not plan.detail and plan.detail_reasons == [] and _retained(subs[0].box, plan.window) == 1.0


def test_artist_names_lose_bracketed_and_trailing_facts():
    from neural_canvas.identify import clean_artist
    assert clean_artist("Tom Lovell (American, 1909-1997)") == "Tom Lovell"
    assert clean_artist("Eyvind Earle (1916-2000)") == "Eyvind Earle"
    assert clean_artist("Thomas Moran, 1837–1926") == "Thomas Moran"
    assert clean_artist("Jim Musil") == "Jim Musil" and clean_artist("Carl Heinrich Bloch") == "Carl Heinrich Bloch"
