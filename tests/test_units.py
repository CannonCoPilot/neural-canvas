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


def test_faces_that_cannot_share_a_window_keep_the_main_head_and_make_a_detail_not_a_block():
    faces = [Subject("a", (0.4, 0.05, 0.5, 0.12), 5, True, "head"), Subject("b", (0.4, 0.85, 0.5, 0.92), 5, True, "head")]
    plan = Composer(3840, 2160, 0.97).plan(_img(1000, 2000), _interp([Subject("a", (0.3, 0.0, 0.7, 1.0), 5, True, "figure")],
                                                                      faces=faces))
    assert plan.passed and plan.detail and _is_16x9_full_bleed(plan)
    assert _retained(faces[0].box, plan.window) == 1.0          # the main subject's head, whole
    assert any("secondary face of 'b'" in w for w in plan.warnings)
    assert plan.faces_kept == "1/2"                             # one head whole, not two halves


def test_a_crowd_of_secondary_faces_never_outvotes_the_main_head():
    # Bloch, Christus Consolator (1881), as the VLM read it: Christ's head at the top, seven faces below
    subs = [Subject("Christus Consolator", (0.078, 0.05, 0.885, 1.0), 5, True, "figure"),
            Subject("woman in dark veil", (0.0, 0.375, 0.215, 1.0), 4, False, "figure"),
            Subject("child", (0.15, 0.665, 0.265, 0.98), 3, False, "figure"),
            Subject("bearded man clinging to Christ's robe", (0.575, 0.595, 0.865, 1.0), 4, False, "figure")]
    faces = [Subject("Christus Consolator", (0.4, 0.05, 0.54, 0.265), 5, True, "head"),
             Subject("woman in dark veil", (0.095, 0.4, 0.17, 0.525), 5, False, "head"),
             Subject("child", (0.17, 0.68, 0.24, 0.8), 5, False, "head"),
             Subject("bearded man clinging to Christ's robe", (0.59, 0.6, 0.68, 0.74), 5, False, "head"),
             Subject("man with blue headband", (0.735, 0.575, 0.805, 0.725), 5, False, "head"),
             Subject("elderly man with staff", (0.275, 0.75, 0.385, 0.93), 5, False, "head")]
    plan = Composer(3840, 2160, 0.97).plan(_img(1937, 1426), _interp(subs, "religious-historical", faces))
    assert _retained(faces[0].box, plan.window) == 1.0 and not plan.detail
    assert _retained(faces[2].box, plan.window) == 1.0          # the child kept whole: the window goes as low as it may
    assert plan.faces_kept == "5/6"                             # only the elderly man, who cannot fit with Christ


def test_a_tie_at_the_top_goes_to_the_subject_the_focal_point_falls_on():
    from neural_canvas.crop import main_subject
    # Bloch, Christ at Gethsemane: Christ and the angel both read as importance 5; the eye goes to Christ
    subs = [Subject("Christ", (0.012, 0.2, 0.62, 0.998), 5, True, "figure"), Subject("angel", (0.44, 0.015, 1.0, 0.96), 5, True, "figure")]
    faces = [Subject("Christ", (0.45, 0.22, 0.58, 0.42), 5, True, "head"), Subject("angel", (0.45, 0.035, 0.6, 0.22), 5, True, "head")]
    assert main_subject(_interp(subs, "religious-historical", faces, focal=(0.48, 0.34))).label == "Christ"


def test_the_main_head_is_found_inside_the_main_subject_when_no_face_bears_its_label():
    from neural_canvas.crop import main_heads, main_subject
    subs = [Subject("saint", (0.2, 0.0, 0.6, 1.0), 5, True, "figure"), Subject("donor", (0.7, 0.4, 0.9, 1.0), 3, False, "figure")]
    faces = [Subject("haloed face", (0.35, 0.05, 0.45, 0.2), 5, True, "head"), Subject("donor", (0.75, 0.45, 0.85, 0.6), 3, False, "head")]
    it = _interp(subs, faces=faces)
    assert main_subject(it).label == "saint" and [f.label for f in main_heads(it)] == ["haloed face"]


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
    assert "Summer Morning (c. 1891) (detail)" in [t for t, _, _ in lines] and not fails


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
    assert text[-2:] == ["Oil paint on canvas – Impressionism", "Los Angeles County Museum of Art"]
    assert text == ["Paul Gauguin", "France, 1848–1903", "The Field of Derout-Lollichon (1886)",
                    "Oil paint on canvas – Impressionism", "Los Angeles County Museum of Art"]   # Sir's five lines
    x0, y0, x1, y1 = pl.box
    assert (x1, y1) == (3840 - 40, 2160 - 40) and (x1 - x0, y1 - y0) == card_size(0.5) == (411, 171)   # fixed, half of 822


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


def test_every_card_is_the_same_size_and_prints_title_with_its_date():
    short = _card({"artist": "Jim Musil", "title": "Towering Storm", "date": "2025"})
    full = _card({"artist": "James Tissot", "artist_nationality": "France", "artist_born": "1836",
                  "artist_died": "1902", "title": "The Miracle of the Loaves and Fishes", "date": "1886–1894",
                  "medium": "Opaque watercolor over graphite", "movement": "Academic Art",
                  "collection": "Brooklyn Museum"})
    for pl in (short, full):
        x0, y0, x1, y1 = pl.box
        assert (x1 - x0, y1 - y0) == (411, 171) and pl.passed, pl.failures
    assert "The Miracle of the Loaves and Fishes (1886–1894)" in full.lines


def test_a_long_title_shrinks_then_wraps_and_an_impossible_one_fails():
    mid = _card({"artist": "Alphonse Mucha", "title": "Slav Epic: The Apotheosis of the Slavs", "date": "1926"})
    assert mid.passed and "Slav Epic: The Apotheosis of the Slavs (1926)" in mid.lines   # set smaller, one line
    long = _card({"artist": "Alfons Mucha", "title": "Slavs in their Original Homeland: Between the Turanian Whip "
                  "and the Sword of the Goths", "date": "1912"})
    assert long.passed and sum(1 for s in long.lines if "Slavs" in s or "Goths" in s) == 2 and long.box[3] - long.box[1] == 171
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


def _structured_art(w=600, h=400, seed=3):
    """A picture with real structure (ORB needs corners): coloured rectangles and circles on noise."""
    import cv2
    rng = np.random.default_rng(seed)
    a = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(a, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), col, -1)
        else:
            cv2.circle(a, (x, y), int(rng.integers(5, 30)), col, -1)
    return a


def _cast(a, rgb_gain):
    return Image.fromarray(np.clip(a.astype(float) * np.array(rgb_gain), 0, 255).astype(np.uint8))


def test_colour_moves_to_the_middle_ground_of_several_copies():
    from neural_canvas.colour import _lab, rebalance
    a = _structured_art()
    src = _cast(a, (1.25, 1.0, 0.8))                          # the input: too warm
    copies = [("m1", _cast(a, (1.0, 1.0, 1.0))), ("m2", _cast(a, (0.98, 1.0, 1.03))),
              ("bw", Image.fromarray(a).convert("L").convert("RGB"))]
    out, rec = rebalance(src, copies)
    assert rec["applied"] and len(rec["copies_used"]) == 2
    assert any("greyscale" in s["why"] for s in rec["copies_skipped"])
    b_in, b_out, b_ref = (_lab(im)[..., 2].mean() for im in (src, out, Image.fromarray(a)))
    assert abs(b_out - b_ref) < abs(b_in - b_ref)           # moved toward the copies' colour


def test_one_copy_is_not_a_middle_ground():
    from neural_canvas.colour import rebalance
    a = _structured_art()
    src = _cast(a, (1.2, 1.0, 0.9))
    out, rec = rebalance(src, [("m1", Image.fromarray(a))])
    assert not rec["applied"] and "needed" in rec["why_not"] and out is src


def test_upscale_drift_is_matched_back_without_blurring_detail():
    from neural_canvas.colour import match_low_frequency
    from neural_canvas.upscale import _mean_delta_e
    ref = Image.fromarray(_structured_art(800, 450))
    drifted = _cast(np.asarray(ref), (1.1, 0.95, 0.9))
    fixed = match_low_frequency(drifted, ref)
    assert _mean_delta_e(fixed, ref) < 0.4 * _mean_delta_e(drifted, ref)
    edges = lambda im: np.abs(np.diff(np.asarray(im.convert("L"), float), axis=1)).mean()
    assert edges(fixed) > 0.9 * edges(drifted)


def test_page_reliability_and_non_titles():
    from neural_canvas.sources import classify, non_title
    assert classify("https://www.mutualart.com/Artwork/x") == "market"           # not "art.com"
    assert classify("https://www.metmuseum.org/art/collection/search/436121") == "curated"
    assert classify("https://www.pinterest.com/pin/123/") == "social"
    assert classify("https://jmcnwhistler.wordpress.com/2020/06/x") == "blog"
    assert non_title("200 Klee REd ideas") and non_title("Venice") and non_title("Shin Paintings for Sale - Fine Art America")
    assert non_title("Giftbeeren") == "" and non_title("Shot in the Dark") == ""


def test_titles_dates_and_media_agree_across_wording():
    from neural_canvas.catalogue import same_date, same_medium, same_title
    assert same_title("Étretat: The Beach and the Falaise d'Amont", "Etretat: the beach and the falaise d'Amont")
    assert same_title("Te raau rahi (The Big Tree)", "The Big Tree (Te raau rahi)")
    assert not same_title("Wind from the Sea", "Airborne")
    assert same_date("1886–1894", "1890") and same_date("1860s", "1864") and not same_date("1919", "1930")
    assert same_medium("Oil paint on canvas", "Oil on jute canvas") and not same_medium("Oil on canvas", "Watercolour on paper")


def test_one_vote_per_institution():
    from neural_canvas.catalogue import institution
    assert institution("https://images.nga.gov/x") == institution("https://www.nga.gov/y") == "nga"
    assert institution("artic") == institution("https://www.artic.edu/artworks/1") == "artic"
    assert institution("https://www.vam.ac.uk/item/O1") == "vam"
    assert institution("https://www.meisterdrucke.us/x") == institution("https://www.meisterdrucke.uk/y")


def _rec(source, title, **kw):
    from neural_canvas.catalogue import Record
    return Record(source, f"https://{source}/x", title, **kw)


def test_title_and_artist_need_two_sources_and_other_facts_one_reliable_witness():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    ident = {"artist": "Frederic Remington", "inscribed_title": ""}
    wd = _rec("wikidata", "A Dash for the Timber", artist="Frederic Remington", date="1889", medium="Oil paint on canvas",
              collection="Amon Carter Museum of American Art", artist_born="1861", artist_died="1909", artist_nationality="United States")
    page = _rec("cartermuseum.org", "A Dash for the Timber", text="a dash for the timber frederic remington 1861 1909 1889 oil on canvas")
    ru = cat.rule([wd, page], ident)
    assert ru["facts"]["title"] == "A Dash for the Timber" and ru["facts"]["artist"] == "Frederic Remington"
    assert ru["facts"]["date"] == "1889" and not ru["failures"]
    assert ru["facts"]["artist_nationality"] == "American"    # one reliable witness (Wikidata) prints it (Sir, 2026-11-10)
    ru1 = cat.rule([wd], ident)
    assert ru1["failures"] and "title not ruled" in ru1["failures"][0]


def test_an_inscribed_title_that_differs_is_shown_beside_the_ruled_one():
    from neural_canvas.catalogue import Catalogue
    from neural_canvas.placard import _layout
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    a = _rec("wikidata", "Poisonous Berries", titles=["Giftbeeren"], artist="Paul Klee", date="1920")
    b = _rec("zpk.org", "Giftbeeren", text="paul klee giftbeeren 1920")
    ru = cat.rule([a, b], {"artist": "Paul Klee", "inscribed_title": "Giftbeeren"})
    assert ru["facts"]["title"] == "Poisonous Berries" and ru["facts"]["inscribed_title"] == "Giftbeeren"
    lines, _ = _layout({"artist": "Paul Klee", **ru["facts"]}, 0.3, 221)
    assert "Poisonous Berries 'Giftbeeren' (1920)" in [t for t, _, _ in lines]


def test_leads_keep_search_text_and_drop_furniture():
    from neural_canvas.catalogue import leads
    L = leads({"title": "Venice", "inscribed_title": "Giftbeeren", "evidence": {"web_references": [
        "file: 1a666d5a 7d4b 3dfc4 e1cf 5 e6633 | page: 200 Klee REd ideas | paul klee | site: pinterest.com",
        "file: Airborne Andrew Wyeth  | page: Wind from the Sea (1948) by Andrew Wyeth – Artchive | site: artchive.com"]}})
    assert L[0] == "Giftbeeren" and "Airborne Andrew Wyeth" in L and "Klee REd" in L
    assert not any(any(c.isdigit() for c in w) and any(c.isalpha() for c in w) for t in L for w in t.split())


def test_a_near_same_title_in_another_medium_is_another_work():
    from neural_canvas.catalogue import Catalogue, institution
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    ident = {"artist": "James McNeill Whistler", "inscribed_title": ""}
    yale = _rec("yale", "The Riva—Sunset: Red and Gold", artist="James McNeill Whistler", medium="Pastel and black chalk")
    salute = _rec("hunterian.gla.ac.uk", "Sunset, Red and Gold: Salute", artist="James McNeill Whistler", medium="Oil on panel")
    assert "title" not in cat.rule([yale, salute], ident)["facts"]          # the titles alone would agree
    wd = _rec("wikidata", "The Riva - Sunset: Red and Gold", artist="James McNeill Whistler", medium="pastel")
    ru = cat.rule([yale, salute, wd], ident)
    assert ru["support"]["title"] == ["wikidata", "yale"]
    assert institution("https://lux.collections.yale.edu/data/object/1") == institution("https://artgallery.yale.edu/x") == "yale"
    assert institution("si") == institution("https://americanart.si.edu/artwork/x") == "si"


def test_a_yale_linked_art_record_reads_into_a_record():
    from neural_canvas.catalogue import Catalogue
    L = lambda s: [{"_label": s}]
    x = {"id": "https://lux.collections.yale.edu/data/object/cef3", "_label": "The Riva—Sunset: Red and Gold",
         "identified_by": [{"type": "Name", "classified_as": L("Primary Title"), "content": "The Riva—Sunset: Red and Gold"},
                           {"type": "Identifier", "classified_as": L("Accession Number"), "content": "1966.9.27"}],
         "produced_by": {"part": [{"classified_as": L("Artist"), "carried_out_by": [
                             {"_label": "Artist: James McNeill Whistler (American, 1834–1903)"}]}],
                         "timespan": {"identified_by": [{"classified_as": L("Display Date"), "content": "1879–80"}]}},
         "referred_to_by": [{"classified_as": L("Medium"), "content": "Pastel and black chalk"},
                            {"classified_as": L("Signed"), "content": "with butterfly"}],
         "current_owner": [{"_label": "Yale University Art Gallery"}],
         "representation": [{"digitally_shown_by": [{"access_point": [{"id": "https://media.collections.yale.edu/thumbnail/yuag/obj/8121"}]}]}]}
    r = Catalogue._lux_rec(x)
    assert (r.title, r.artist, r.artist_bio, r.date) == ("The Riva—Sunset: Red and Gold", "James McNeill Whistler", "American, 1834–1903", "1879–80")
    assert r.medium == "Pastel and black chalk" and r.collection == "Yale University Art Gallery" and r.image_url.endswith("/8121")
    assert r.states("with butterfly") and r.titles == []


def test_a_rijksmuseum_linked_art_record_reads_its_aat_codes():
    from neural_canvas.catalogue import Catalogue
    aat = lambda n: [{"id": f"http://vocab.getty.edu/aat/{n}"}]
    lang = lambda n: [{"id": f"http://vocab.getty.edu/aat/{n}"}]
    x = {"id": "https://id.rijksmuseum.nl/200108369",
         "identified_by": [{"type": "Identifier", "content": "SK-A-2344"},
                           {"type": "Name", "classified_as": aat(300404670), "content": "Het melkmeisje", "language": lang(300388256)},
                           {"type": "Name", "classified_as": aat(300404670), "content": "The Milkmaid", "language": lang(300388277)},
                           {"type": "Name", "classified_as": [{"id": "https://id.rijksmuseum.nl/22015528"}], "content": "De keukenmeid"}],
         "produced_by": {"part": [{"carried_out_by": [{"notation": [{"@language": "nl", "@value": "Johannes Vermeer"},
                                                                     {"@language": "en", "@value": "Johannes Vermeer"}]}]}],
                         "timespan": {"identified_by": [{"content": "ca. 1660", "language": lang(300388256)},
                                                        {"content": "c. 1660", "language": lang(300388277)}]}},
         "referred_to_by": [{"classified_as": aat(300435429), "content": "olieverf op doek", "language": lang(300388256)},
                            {"classified_as": aat(300435429), "content": "oil on canvas", "language": lang(300388277)}],
         "subject_of": [{"digitally_carried_by": [{"access_point": [{"id": "https://iiif.micr.io/QkOGy/full/max/0/default.jpg"}]}]}]}
    r = Catalogue._rijks_rec(x)
    assert (r.title, r.artist, r.date, r.medium) == ("The Milkmaid", "Johannes Vermeer", "c. 1660", "oil on canvas")
    assert "Het melkmeisje" in r.titles and "De keukenmeid" in r.titles
    assert r.image_url == "https://iiif.micr.io/QkOGy/full/800,/0/default.jpg" and r.source == "rijksmuseum.nl"


def test_a_paced_host_keeps_its_gap_across_threads_and_a_retry_after_pauses_them_all():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from neural_canvas import matching
    host = "pace.test"
    starts = []
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda _: (matching._wait_turn(host, 0.05), starts.append(time.time())), range(6)))
    starts.sort()
    # stamped after each thread wakes: wake jitter (up to ~10 ms measured 2026-11-06, HEAD too) shrinks a 50 ms slot gap
    assert all(b - a >= 0.03 for a, b in zip(starts, starts[1:]))
    matching._back_off(host, 0.3)
    t = time.time(); matching._wait_turn(host, 0.0)
    assert time.time() - t >= 0.25
    assert matching._gate("upload.wikimedia.org") is matching._gate("commons.wikimedia.org") and matching._gate("example.org") is None


def test_an_inscription_is_support_and_a_lead_never_a_vote():
    from neural_canvas.catalogue import Catalogue, inscription_is_title
    assert inscription_is_title("Giftbeeren", "Paul Klee") and inscription_is_title("Rote Ideen", "Paul Klee")
    assert not any(inscription_is_title(x, "Paul Klee") for x in ("Klee", "P K", "PK 1920", "1920.92", "Klee 1920", "J. M. W."))
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    one = _rec("artvee.com", "Giftbeeren (Poisonous Berries)", artist="Paul Klee")
    ru = cat.rule([one], {"artist": "Paul Klee", "inscribed_title": "Giftbeeren"})
    assert "title" not in ru["facts"]                                        # one source + an inscription: not ruled
    assert {"source": "inscription", "url": "", "title": "Giftbeeren"} in ru["corroborating"]


def test_an_auction_house_corroborates_and_never_counts():
    from neural_canvas.catalogue import Catalogue
    from neural_canvas.sources import supporting
    assert supporting("https://www.sothebys.com/en/buy/auction/x") and supporting("https://www.smithsonianmag.com/a")
    assert not supporting("https://www.mutualart.com/Artwork/x") and not supporting("https://www.artic.edu/artworks/1")
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    wd = _rec("wikidata", "Giftbeeren", artist="Paul Klee")
    sale = _rec("sothebys.com", "Giftbeeren", artist="Paul Klee", support=True)
    ru = cat.rule([wd, sale], {"artist": "Paul Klee", "inscribed_title": ""})
    assert "title" not in ru["facts"] and ru["corroborating"][0]["source"] == "sothebys.com"


def test_artvee_cards_and_arthur_images_read_into_references():
    from pathlib import Path
    from neural_canvas.artsites import Arthur, Artvee
    from neural_canvas.catalogue import leads
    from neural_canvas.websearch import WebReference
    html = (Path(__file__).parent / "fixtures" / "artvee_card.html").read_text()
    [r] = Artvee.parse(html)
    assert r.meta["title"] == "Giftbeeren (Poisonous Berries)" and r.meta["date"] == "1920" and r.meta["artist"] == "Paul Klee"
    assert r.thumb_url == "https://mdl.artvee.com/ft/102435ab.jpg" and r.page_url.endswith("/dl/giftbeeren-poisonous-berries/")
    assert leads({"evidence": {"web_references": [f"file: 102435ab | page: {r.file_title} | site: artvee.com"]}}) == \
        ["Giftbeeren (Poisonous Berries)"]
    class Web:
        def search(self, q, n):
            img = "https://arthur.io/img/art/jpg/00017344a235a55cd/andrew-wyeth/airborne/small/andrew-wyeth--airborne.jpg"
            return [WebReference("Andrew Wyeth - Airborne, 1996", 1, img, "https://arthur.io/art/andrew-wyeth/airborne", img),
                    WebReference("pin", 2, "https://i.pinimg.com/x.jpg", "https://pinterest.com/x", "https://i.pinimg.com/x.jpg")]
    hits = Arthur(Web()).search("Andrew Wyeth")
    assert len(hits) == 1 and hits[0].image_url.endswith("/large/andrew-wyeth--airborne.jpg")
    assert hits[0].meta["title_from_file"] == "airborne" and hits[0].file_title == "Andrew Wyeth - Airborne, 1996"


def test_an_artist_named_only_in_page_text_is_ruled_by_two_pages():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    md = _rec("meisterdrucke.uk", "Poisonous Berries, 1920", text="poisonous berries 1920 by paul klee fine art print")
    av = _rec("artvee.com", "Giftbeeren (Poisonous Berries)", text="giftbeeren poisonous berries 1920 paul klee german 1879 1940")
    ru = cat.rule([md, av], {"artist": "Paul Klee", "inscribed_title": "Giftbeeren"})
    assert ru["facts"]["artist"] == "Paul Klee" and ru["support"]["artist"] == ["artvee.com", "meisterdrucke.uk"]
    one = cat.rule([md], {"artist": "Paul Klee", "inscribed_title": "Giftbeeren"})
    assert "artist" not in one["facts"] and any("each fact needs 2" in f for f in one["failures"])


def test_rijksmuseum_picks_the_artist_among_makers_and_leads_drop_the_site_name():
    from neural_canvas.catalogue import Catalogue, leads
    x = {"id": "https://id.rijksmuseum.nl/1", "identified_by": [{"type": "Name", "content": "Hauptweg"}],
         "produced_by": {"part": [{"carried_out_by": [{"notation": {"@language": "nl", "@value": "Staatliches Bauhaus"}}]},
                                  {"carried_out_by": [{"notation": {"@language": "nl", "@value": "Paul Klee"}}]}]}}
    r = Catalogue._rijks_rec(x, "klee")
    assert r.artist == "Paul Klee" and r.states("staatliches bauhaus")
    L = leads({"evidence": {"web_references": ["file: a | page: Giftbeeren (Poisonous Berries) - Artvee | site: artvee.com",
                                               "file: b | page: The Riva - Sunset: Red and Gold | site: yale.edu"]}})
    assert L[:1] == ["Giftbeeren (Poisonous Berries)"] and "The Riva - Sunset: Red and Gold" in L


def test_text_search_merges_engines_and_never_logs_a_key(caplog):
    from neural_canvas.websearch import WebImageSearch
    w = WebImageSearch(searxng_url="http://searx.test")
    w._keys = {"serper": "SECRET-KEY-123", "brave": ""}
    w._searx_pages = lambda q, n: [("https://a.org/1", "a"), ("https://b.org/2", "b")]
    def boom(q, n):
        raise RuntimeError("401 for https://google.serper.dev/search?key=SECRET-KEY-123")
    w._serper_pages = boom
    hits = w.pages("klee", 10)
    assert [u for u, _ in hits] == ["https://a.org/1", "https://b.org/2"]
    assert "SECRET-KEY-123" not in caplog.text and "***" in caplog.text


def test_serpapi_is_a_fallback_with_a_reserve_and_never_an_everyday_engine(monkeypatch):
    import pytest
    from neural_canvas.websearch import WebImageSearch
    w = WebImageSearch(searxng_url="http://searx.test")
    w._keys = {"serper": "", "brave": "", "serpapi": "SERP-KEY"}
    called = []
    w._searx_pages = lambda q, n: [("https://a.org/1", "a")]
    monkeypatch.setattr(w, "serpapi_pages", lambda q, n=20: called.append(q) or [])
    assert w.pages("klee") == [("https://a.org/1", "a")] and not called      # everyday search: no SerpAPI
    monkeypatch.setattr(w, "serpapi_left", lambda: WebImageSearch.SERPAPI_RESERVE)
    with pytest.raises(RuntimeError, match="spent"):
        w._serpapi({"engine": "google", "q": "x"})                           # the reserve is never touched


def test_artvee_finds_every_page_an_artist_is_filed_under(tmp_path, monkeypatch):
    from neural_canvas.artsites import Artvee
    av = Artvee(tmp_path)
    page = ('<a href="https://artvee.com/artist/frederic-remington/">x</a>'
            '<a href="https://artvee.com/artist/remington-schuyler/">y</a>')
    monkeypatch.setattr(av, "_get", lambda url: page)
    assert av.artist_pages("Frederick Remington") == ["frederick-remington", "frederic-remington"]


def test_catalogue_pages_read_labelled_fields_and_titles_without_the_byline():
    from neural_canvas.catalogue import labelled
    f = labelled("M.0807 The Riva - Sunset; red and gold Artist: James McNeill Whistler Date: 1880 "
                 "Collection: Yale University Art Gallery Accession Number: 1966.9.27 Medium: chalk and pastel Support: paper")
    assert f["Date"] == "1880" and f["Collection"] == "Yale University Art Gallery" and f["Medium"] == "chalk and pastel"


def test_the_overrule_tier_decides_and_lower_disagreement_is_discarded():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    ident = {"artist": "Andrew Wyeth", "inscribed_title": ""}
    museum = _rec("crystalbridges", "Airborne", artist="Andrew Wyeth", date="1996", overrule=True)
    wikiart = _rec("wikiart.org", "Airborne", artist="Andrew Wyeth", date="1996")
    artchive = _rec("artchive.com", "Wind from the Sea", artist="Andrew Wyeth", date="1947")
    blog_like = _rec("wga.hu", "Wind from the Sea", artist="Andrew Wyeth", date="1947")
    ru = cat.rule([museum, wikiart, artchive, blog_like], ident)
    # two lower-tier sources agree on the wrong title, but the overrule tier states the fact: it decides
    assert ru["facts"]["title"] == "Airborne" and ru["facts"]["date"] == "1996"
    assert "artchive.com: Wind from the Sea" in ru["discarded"]["title"]
    # with no overrule source, the ordinary votes decide
    ru2 = cat.rule([artchive, blog_like], ident)
    assert ru2["facts"]["title"] == "Wind from the Sea"


def test_overrule_sources_vote_among_themselves():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    a = _rec("yale", "The Riva—Sunset: Red and Gold", artist="James McNeill Whistler", overrule=True)
    b = _rec("gla", "The Riva - Sunset; red and gold", artist="James McNeill Whistler", overrule=True)
    c = _rec("artvee.com", "Venice", artist="James McNeill Whistler")
    ru = cat.rule([c, a, b], {"artist": "James McNeill Whistler", "inscribed_title": ""})
    assert ru["support"]["title"] == ["gla", "yale"] and ru["discarded"]["title"] == ["artvee.com: Venice"]


def test_nga_open_data_indexes_and_finds_an_artists_works(tmp_path):
    from neural_canvas.nga import NGA, build_index
    w = lambda name, text: (tmp_path / f"{name}.csv").write_text(text, encoding="utf-8")
    w("objects", "objectid,title,displaydate,medium,attribution,inscription,accessionnum,classification,wikidataid\n"
                 '17,Wind from the Sea,1947,tempera on hardboard,Andrew Wyeth,,1975.1.1,Painting,Q1\n')
    w("constituents", "constituentid,forwarddisplayname,preferreddisplayname,lastname,displaydate,nationality,wikidataid,ulanid\n"
                      "5,Andrew Wyeth,\"Wyeth, Andrew\",Wyeth,\"American, 1917 - 2009\",American,Q2,500\n")
    w("objects_constituents", "objectid,constituentid,role,prefix\n17,5,artist,\n")
    w("published_images", "depictstmsobjectid,iiifurl,viewtype,sequence\n17,https://api.nga.gov/iiif/abc,primary,0\n")
    counts = build_index(tmp_path, tmp_path / "nga.sqlite")
    assert counts == {"works": 1, "artists": 1, "work_artists": 1, "images": 1}
    [r] = NGA(tmp_path / "nga.sqlite").works_by("Andrew Wyeth")
    assert r["title"] == "Wind from the Sea" and NGA.image(r) == "https://api.nga.gov/iiif/abc/full/!500,500/0/default.jpg"


def _svc(tmp_path, **pulse):
    from neural_canvas import service as svc
    from neural_canvas.config import DEFAULTS, Config, _merge
    cfg = Config(_merge(DEFAULTS, {"paths": {k: str(tmp_path / k) for k in ("watch_dir", "work_dir", "output_dir",
                                                                             "state_dir")}, "pulse": pulse}))
    return svc.Service(cfg)


def test_runner_runs_works_in_parallel_up_to_its_workers(tmp_path, monkeypatch):
    import threading
    s = _svc(tmp_path, workers=2)
    gate, started, claims = threading.Event(), [], []
    monkeypatch.setattr(s.pulse, "list", lambda label: [{"id": t, "labels": ["staging:wait"], "metadata": {"run_id": t}}
                                                       for t in ("t1", "t2", "t3")])
    # like Pulse: a claimed task is no longer staging:wait, so a second claim of it fails
    monkeypatch.setattr(s.pulse, "claim", lambda tid, label: tid not in claims and (claims.append(tid) or True))
    monkeypatch.setattr(s.pulse, "get", lambda tid: {"id": tid})
    monkeypatch.setattr(s.photos, "check", lambda: "")
    monkeypatch.setattr(s, "run_task", lambda task: started.append(task["id"]) or gate.wait(10))
    assert s.run_ready() == 2 and claims == ["t1", "t2"]
    assert s.run_ready() == 0 and claims == ["t1", "t2"]      # both workers busy: nothing more claimed
    gate.set()
    for f in list(s._running.values()):
        f.result(10)
    assert s.run_ready() == 1 and claims[-1] == "t3"


def test_runner_error_blocks_its_task_instead_of_stranding_it(tmp_path, monkeypatch):
    s = _svc(tmp_path)
    added, notes = [], []
    monkeypatch.setattr(s.pulse, "get", lambda tid: {"id": tid})
    monkeypatch.setattr(s.pulse, "labels", lambda tid, add=(), remove=(): added.append((list(add), list(remove))))
    monkeypatch.setattr(s.pulse, "note", lambda tid, text, metadata=None, status=None: notes.append((text, status)))
    monkeypatch.setattr(s.pulse, "alert", lambda *a: None)

    def boom(task):
        raise ValueError("bad record")
    monkeypatch.setattr(s, "run_task", boom)
    s._run_guarded("t1")
    assert added == [(["blocked:yes", "reason:docent-error", "active:no"], ["active:running"])]
    assert "ValueError" in notes[0][0] and notes[0][1] == "open"


def test_service_start_requeues_runs_stranded_by_a_stop(tmp_path, monkeypatch):
    s = _svc(tmp_path)
    changed = []
    monkeypatch.setattr(s.pulse, "list", lambda label: [
        {"id": "a", "status": "in_progress", "labels": ["active:running", "staging:done", "docent:identify"]},
        {"id": "b", "status": "open", "labels": ["staging:wait"]},
        {"id": "c", "status": "closed", "labels": ["active:running"]}])
    monkeypatch.setattr(s.pulse, "labels", lambda tid, add=(), remove=(): changed.append((tid, set(add), set(remove))))
    monkeypatch.setattr(s.pulse, "note", lambda *a, **k: None)
    assert s.requeue_stranded() == ["a"]
    tid, add, remove = changed[0]
    _pulse_labels_valid(sorted(add))
    assert "staging:wait" in add and {"active:running", "docent:identify"} <= remove


def test_a_source_agreeing_in_other_words_is_not_listed_as_discarded():
    # drop work 2 (Velasco), 2026-11-05: the ruling was voted on the Spanish wording and printed in Wikidata's
    # English; Artvee's title was the printed one word for word, yet it was listed as discarded
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    en, es = "The Valley of Mexico from the Santa Isabel Mountain Range", "Valle de México desde el cerro de Santa Isabel"
    wd = _rec("wikidata", en, titles=[es], artist="José María Velasco", overrule=True)
    commons = _rec("commons.wikimedia.org", es, artist="José María Velasco", overrule=True)
    wikiart = _rec("wikiart.org", es + ", 1882", artist="José María Velasco")
    artvee = _rec("artvee.com", en, artist="José María Velasco")
    other = _rec("meisterdrucke.com", "Popocatepetl at dawn", artist="José María Velasco")
    ru = cat.rule([wd, commons, wikiart, artvee, other], {"artist": "José María Velasco", "inscribed_title": ""})
    assert ru["facts"]["title"] == en
    assert ru["discarded"]["title"] == ["meisterdrucke.com: Popocatepetl at dawn"]


def test_footprint_finds_the_part_of_a_picture_a_detail_shows():
    # the input is often a detail: verification compares it with the part of the record's picture it shows
    from PIL import ImageFilter
    from neural_canvas.matching import footprint
    rng = np.random.default_rng(7)
    small = Image.fromarray(rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)).resize((1200, 900), Image.NEAREST)
    big = small.filter(ImageFilter.GaussianBlur(1.5))
    detail = big.crop((300, 200, 900, 700))
    part = footprint(big, detail)
    assert part is not None
    assert abs(part.width - 600) < 40 and abs(part.height - 500) < 40
    noise = Image.fromarray(rng.integers(0, 255, (500, 600, 3), dtype=np.uint8))
    assert footprint(big, noise) is None or footprint(big, noise).width < 1200


def test_a_museum_record_brings_in_its_own_wikidata_item(monkeypatch):
    from neural_canvas.catalogue import Catalogue, Record
    cat = Catalogue.__new__(Catalogue)
    cat.consulted, cat.t, cat.cfg = [], {}, None
    calls = []
    nga = Record("nga", "https://www.nga.gov/x", "Five Butterflies", artist="Odilon Redon", verified_by="image",
                 wikidata="Q64618525")
    wd = Record("wikidata", "https://www.wikidata.org/wiki/Q64618525", "Five Butterflies", artist="Odilon Redon")
    monkeypatch.setattr(cat, "_wikidata_record", lambda qid: calls.append(qid) or (wd, {}))
    for name in ("_wikidata_items", "_artic_search", "_met_search", "_cleveland_search", "_europeana", "_harvard",
                 "_smithsonian", "_yale", "_rijksmuseum", "_registry", "_artvee", "_curated_web", "_web_text",
                 "_translations"):
        monkeypatch.setattr(cat, name, lambda *a, **k: [])
    monkeypatch.setattr(cat, "_nga", lambda *a, **k: [nga])
    monkeypatch.setattr(cat, "_artist_living", lambda a: (False, ""))
    monkeypatch.setattr(cat, "_get", lambda *a, **k: {})
    cat.emb = type("E", (), {"embed": staticmethod(lambda ims: np.zeros((1, 512)))})()
    recs = cat.gather(Image.new("RGB", (8, 8)), {"artist": "Odilon Redon"}, ["Five Butterflies"])
    assert calls == ["Q64618525"]
    got = next(r for r in recs if r.source == "wikidata")
    assert got.verified_by == "link from nga" and got.overrule


def test_a_task_closed_on_the_board_stops_its_run_before_the_next_stage(tmp_path, monkeypatch):
    from neural_canvas.pipeline import RunResult, RunStopped
    s = _svc(tmp_path)
    notes, labels = [], []
    status = {"v": "in_progress"}
    monkeypatch.setattr(s.pulse, "get", lambda tid: {"id": tid, "status": status["v"]})
    monkeypatch.setattr(s.pulse, "note", lambda tid, text, metadata=None, status=None: notes.append(text))
    monkeypatch.setattr(s.pulse, "labels", lambda tid, add=(), remove=(): labels.append((list(add), list(remove))))
    monkeypatch.setattr(s.pulse, "close", lambda *a: (_ for _ in ()).throw(AssertionError("closed twice")))

    class FakeDocent:
        def run(self, rid, src, resume, on_stage):
            on_stage("identify")
            status["v"] = "closed"                     # Sir closes it on the board mid-run
            try:
                on_stage("catalogue")
            except RunStopped as e:
                return RunResult(rid, "stopped", "catalogue", [str(e)])
            raise AssertionError("run went on past a closed task")
    s._local.docent = FakeDocent()
    s.run_task({"id": "t1", "metadata": {"run_id": "r1", "source_path": str(tmp_path / "x.jpg")}, "labels": []})
    assert "STOPPED before catalogue" in notes[-1]
    assert not any("blocked:yes" in add for add, _ in labels)


def test_walled_catalogue_titles_come_from_googles_index_and_vote_on_the_title_only(monkeypatch):
    from neural_canvas.artsites import WalledCatalogue, index_title
    from neural_canvas.catalogue import Catalogue
    from neural_canvas.websearch import WebReference
    assert index_title("Crows in Winter – Works – eMuseum") == "Crows in Winter"
    assert WalledCatalogue.sites("Newell Convers Wyeth") == ["collections.brandywine.org"]
    asked = []

    class Web:
        def index_images(self, q, limit=20):
            asked.append(q)
            return [WebReference("Crows in Winter – Works – eMuseum", 1, "https://encrypted-tbn0.gstatic.com/x",
                                 "https://collections.brandywine.org/objects/11335", "https://encrypted-tbn0.gstatic.com/x",
                                 {"site": "collections.brandywine.org"}),
                    WebReference("Crows - elsewhere", 2, "https://t/y", "https://example.com/y", "https://t/y",
                                 {"site": "example.com"})]
    refs = WalledCatalogue(Web()).search("Newell Convers Wyeth", ["Crows in Winter"])
    assert [r.meta["catalogue"] for r in refs] == ["collections.brandywine.org"]     # only the catalogue's own pictures
    assert all("Newell" not in q and "Wyeth" in q for q in asked)                    # the surname: Google needs every word
    cat = Catalogue.__new__(Catalogue)
    cat.consulted = []
    cat.idf = type("I", (), {"web": Web()})()
    monkeypatch.setattr(cat, "_verify", lambda img, vec, url: (url.startswith("https://encrypted-tbn0"), 0.96, 1488))
    recs = cat._walled(None, None, "Newell Convers Wyeth", ["Crows in Winter"])
    assert len(recs) == 1 and recs[0].title == "Crows in Winter"
    assert not (recs[0].artist or recs[0].date or recs[0].medium)          # Sir: a title vote, nothing more
    assert "Google" in recs[0].verified_by


def test_lens_pool_errors_never_carry_a_key(monkeypatch):
    import requests
    from neural_canvas import lens_pool
    lens_pool._KEYS.add("sekret-key-123456")

    def boom(*a, **k):      # a connection error quotes the whole URL, query-string key included
        raise requests.ConnectionError("HTTPSConnectionPool: /search.json?engine=google_lens&api_key=sekret-key-123456")
    monkeypatch.setattr(requests, "request", boom)
    import pytest
    with pytest.raises(RuntimeError) as e:
        lens_pool._req("GET", "https://serpapi.com/search.json", params={"api_key": "sekret-key-123456"})
    assert "sekret" not in str(e.value) and "api_key=***" in str(e.value)


def _pool(tmp_path, specs):
    from neural_canvas.lens_pool import LensPool, Ledger, Provider
    providers = []
    for name, allowance, live, refs in specs:
        providers.append(Provider(name, lambda w, r=refs, n=name: (_ for _ in ()).throw(RuntimeError("down")) if r == "error" else r,
                                  allowance, "monthly", balance=(lambda v=live: v) if live is not None else None))
    return LensPool(providers, Ledger(tmp_path / "pool.json"), lambda img: "https://tmp/x.jpg")


def test_lens_pool_picks_largest_share_and_skips_providers_under_the_10_percent_floor(tmp_path):
    pool = _pool(tmp_path, [("a", 250, 208, ["r"]), ("b", 100, 60, ["r"]), ("c", 1000, 99, ["r"])])
    assert pool.pick().name == "a"                      # 83% left beats 60%; c (9.9%) is under the floor
    assert [s["eligible"] for s in pool.status()] == [True, True, False]


def test_lens_pool_spends_monthly_pools_before_one_time_reserves(tmp_path):
    pool = _pool(tmp_path, [("monthly", 250, 50, ["r"]), ("reserve", 2500, 2400, ["r"])])
    pool.providers[1].renews = "one-time"
    assert pool.pick().name == "monthly"                # 20% left, but it renews; the 96% reserve waits
    pool.providers[0].balance = lambda: 20
    pool.ledger.update("monthly", remaining=20)
    assert pool.pick().name == "reserve"                # the monthly pool hit its floor


def test_lens_pool_never_runs_a_provider_to_its_last_credits(tmp_path):
    pool = _pool(tmp_path, [("a", 100, 11, ["r"])])
    assert pool.pick().name == "a"                      # 11 - 1 >= 10
    assert pool.ledger.reserve(pool.providers[0])
    pool.providers[0].balance = lambda: 10
    assert pool.pick() is None                          # 10 - 1 < 10: skipped, never paid
    pool.ledger.update("a", remaining=10)
    assert not pool.ledger.reserve(pool.providers[0])   # the locked check refuses it too


def test_lens_pool_rereads_real_balances_when_only_one_provider_is_left(tmp_path):
    pool = _pool(tmp_path, [("a", 100, 50, ["r"]), ("b", 100, 5, ["r"])])
    reads = []
    pool.providers[1].balance = lambda: reads.append(1) or 100    # b's month renewed since the last check
    pool.ledger.update("b", remaining=5, checked=__import__("time").time())
    assert pool.pick().name == "b" and reads             # only a was above the floor -> live re-read found b reset


def test_lens_pool_treats_errors_and_empty_replies_as_failures_and_alerts_when_all_fail(tmp_path):
    from neural_canvas.lens_pool import LensSlotUnavailable
    pool = _pool(tmp_path, [("a", 100, 90, "error"), ("b", 100, 80, []), ("c", 100, 50, ["r"])])
    prov, refs, errs = pool.search("img")
    assert prov == "c" and refs == ["r"] and errs == ["a: down", "b: empty reply"]
    alerts = []
    pool2 = _pool(tmp_path / "2", [("a", 100, 90, "error")])
    pool2.alert = lambda text, data: alerts.append(text)
    import pytest
    with pytest.raises(LensSlotUnavailable):
        pool2.search("img")
    assert alerts and "a: down" in alerts[0]


def test_vision_counts_locally_and_stops_at_the_free_floor(tmp_path, monkeypatch):
    import time
    from neural_canvas.lens_pool import Ledger
    from neural_canvas.vision_web import VisionWebDetection
    from PIL import Image
    led = Ledger(tmp_path / "pool.json")
    led.update("vision", month=time.strftime("%Y-%m"), used_month=900)
    v = VisionWebDetection({"credentials_path": "x", "key_path": "y"}, tmp_path / "vw", led)
    import pytest
    with pytest.raises(RuntimeError, match="spill into paid"):
        v.detect(Image.new("RGB", (8, 8)))


def test_nationality_prints_demonyms_and_a_compound_only_when_the_sources_state_each_part():
    from neural_canvas.catalogue import Catalogue, Record
    cat = Catalogue.__new__(Catalogue)
    answers = {}

    class VLM:
        def ask_json(self, prompt, required=(), num_predict=0):
            return {"label": answers["label"]}
    cat.idf = type("I", (), {"vlm": VLM()})()
    wd = Record("wikidata", "u", "t", artist_nationality="Switzerland", artist_nationalities=["Switzerland", "France"])
    bios = [Record("artvee.com", "u", "t", artist_bio="French, 1865-1925"), Record("rkd.nl", "u", "t", artist_bio="French")]
    answers["label"] = "Swiss-French"
    assert cat._nationality("Félix Vallotton", wd, bios)[:2] == ("Swiss-French", ["artvee.com", "rkd.nl", "wikidata"])
    answers["label"] = "Swiss-Italian"                    # a part no source states is never printed
    assert cat._nationality("Félix Vallotton", wd, bios)[0] == ""
    answers["label"] = "United States"                    # a country name is not a demonym: not ruled
    us = Record("wikidata", "u", "t", artist_nationality="United States", artist_nationalities=["United States"])
    assert cat._nationality("x", us, [Record("artic", "u", "t", artist_bio="American, 1849-1921")])[0] == ""
    answers["label"] = "American"
    assert cat._nationality("x", us, [Record("artic", "u", "t", artist_bio="American, 1849-1921")])[0] == "American"
    assert cat._nationality("x", us, [])[0] == "American"  # one reliable witness rules it (Sir, 2026-11-10)


def test_commons_thumbnails_use_the_widths_wikimedia_serves():
    from neural_canvas.references import file_thumb, thumb_step
    assert thumb_step(800) == 960 and thumb_step(250) == 250 and thumb_step(5000) == 3840
    assert "/960px-" in file_thumb("A b.jpg", 800)


def test_a_historical_region_is_its_own_nationality_and_flemish_equals_netherlandish():
    # van Eyck, 2026-11-05: Wikidata "Southern Netherlands" had become "Dutch", and Artvee's "Flemish" was not read
    from neural_canvas.catalogue import Catalogue, Record, _country_demonym
    assert _country_demonym("Southern Netherlands") == "Flemish"
    assert _country_demonym("Kingdom of the Netherlands") == "Dutch" and _country_demonym("Russian Empire") == "Russian"
    cat = Catalogue.__new__(Catalogue)
    cat.idf = type("I", (), {"vlm": type("V", (), {"ask_json": lambda self, p, required=(), num_predict=0: {"label": "Flemish"}})()})()
    wd = Record("wikidata", "u", "t", artist_nationality="Southern Netherlands", artist_nationalities=["Southern Netherlands"])
    for bio in ("Flemish, c. 1390 – 1441", "Netherlandish, active 1422; died 1441"):
        assert cat._nationality("Jan van Eyck", wd, [Record("x", "u", "t", artist_bio=bio)])[:2] == ("Flemish", ["wikidata", "x"])


def test_first_sight_queues_an_artist_once_and_remembers_the_result(tmp_path):
    from neural_canvas.indexer import ArtistIndexer
    ix = ArtistIndexer(tmp_path, lambda: None)
    assert ix.enqueue("Félix Vallotton") and not ix.enqueue("félix vallotton")      # queued once
    assert ix._pop() == "Félix Vallotton"
    ix._finish("Félix Vallotton", {"wikidata": 450})
    assert ix.indexed("Félix Vallotton") and ix._pop() is None
    assert not ix.enqueue("Félix Vallotton")                                         # indexed: never again


def test_a_run_never_embeds_a_cold_catalogue(tmp_path, monkeypatch):
    from neural_canvas.identify import Identifier
    from neural_canvas import identify as ident
    idf = Identifier.__new__(Identifier)
    idf.cache_dir, idf.ua = tmp_path, "ua"
    idf.wiki = type("W", (), {"oeuvre": lambda self, q: [{"qid": "Q1", "label": "A", "file": "a.jpg"}]})()
    fetched = []
    monkeypatch.setattr(ident, "fetch_image", lambda *a: fetched.append(a))
    works, vecs = idf._oeuvre_vectors("Q9")
    assert works == [] and fetched == []                    # nothing cached, nothing fetched: no wait


def test_hits_are_rejudged_by_site_trust_when_reused_copies_carry_the_wrong_name():
    from neural_canvas.identify import _site_rank
    assert _site_rank("https://www.metmuseum.org/art/collection/search/1") == 0
    assert _site_rank("https://artvee.com/dl/colour-harmony-and-contrast/") == 1
    assert _site_rank("https://open.spotify.com/album/x") == 7       # album covers reuse paintings
    assert _site_rank("https://www.pinterest.at/pin/1/") == 6
    assert _site_rank("https://www.etsy.com/listing/1") == 5
    assert _site_rank("https://example.org/page") == 3


def test_a_commons_file_name_is_never_printed_as_a_title():
    from neural_canvas.catalogue import display_title as d
    assert d("File:Alexej von Jawlensky Landschaft mit orangener Wolke 1909.jpg", "Alexej von Jawlensky") == \
        "Landschaft mit orangener Wolke"
    assert d("File:László Moholy-Nagy, croce rossa e palle bianche, 1921 (museum kunstpalast, düsseldorf).jpg",
             "László Moholy-Nagy") == "croce rossa e palle bianche"
    assert d("File:Château de Chillon by Courbet.jpg", "Gustave Courbet") == "Château de Chillon"
    assert d("Three Worlds", "M. C. Escher") == "Three Worlds"


# ── 2026-11-10: abstracts with nothing to locate, the text-search backfill, the larger source ──

class _FakeVLM:
    def __init__(self, d):
        self.d = d

    def ask_json(self, *a, **k):
        return self.d


def test_an_abstract_with_nothing_to_locate_is_a_valid_reading_and_other_genres_are_not():
    from neural_canvas.interpret import interpret
    d = {"genre": "abstract", "scene": "", "artistic_intent": "", "focal_point": [500, 500], "horizon_y": None,
         "subjects": [], "faces": [], "label_text": "word " * 30}
    it = interpret(_FakeVLM(d), _img(300, 200), {})
    assert it.genre == "abstract" and it.subjects == []
    plan = Composer(3840, 2160, 0.97).plan(_img(1800, 1186), it)     # centred on visual weight; nothing to cut
    assert plan.passed and not plan.detail and _is_16x9_full_bleed(plan)
    with pytest.raises(ValueError):
        interpret(_FakeVLM({**d, "genre": "landscape"}), _img(300, 200), {})


def _text_pool(tmp_path, providers):
    from neural_canvas.lens_pool import LensPool, Ledger
    return LensPool(providers, Ledger(tmp_path / "pool.json"), lambda im: "")


def test_text_search_goes_to_the_next_service_with_credits(tmp_path):
    from neural_canvas.lens_pool import Provider
    asked = []

    def prov(name, left, renews):
        def pages(q, n):
            asked.append(name)
            return [(f"https://{name}.example/{q}", name)]
        return Provider(name, lambda w: [], allowance=100, renews=renews, balance=lambda: left, pages=pages)
    pool = _text_pool(tmp_path, [prov("serpapi", 10, "monthly"), prov("serper", 80, "one-time"), prov("oxylabs", 40, "one-time")])
    assert pool.pages("q", 5) == [("https://serper.example/q", "serper")]      # SerpAPI at its floor: Serper, most left
    assert asked == ["serper"]


def test_text_search_with_every_service_at_its_floor_raises_and_alerts(tmp_path):
    from neural_canvas.lens_pool import Provider, TextSearchUnavailable
    alerts = []
    pool = _text_pool(tmp_path, [Provider("serpapi", lambda w: [], 100, "monthly", balance=lambda: 10, pages=lambda q, n: [("u", "")])])
    pool.alert = lambda msg, ctx: alerts.append(msg)
    with pytest.raises(TextSearchUnavailable):
        pool.pages("q")
    assert alerts


def test_an_engine_low_on_credits_is_backfilled_in_web_pages():
    from neural_canvas.websearch import CreditsLow, WebImageSearch
    web = WebImageSearch()
    web._keys = {"serper": "k", "brave": ""}

    def low(q, n):
        raise CreditsLow("Serper credits at the 10% floor")
    web._serper_pages = low
    got = []
    web.backfill = lambda q, n, exclude: got.append(exclude) or [("https://museum.example/work", "the work")]
    assert web.pages("q", 5) == [("https://museum.example/work", "the work")] and got == [{"serper"}]


def test_commons_files_are_traced_from_pages_uploads_and_thumbnails():
    from neural_canvas.larger import _commons_file
    assert _commons_file("https://commons.wikimedia.org/wiki/File:Le_pav%C3%A9_de_Chailly_(Monet).jpg") == \
        "File:Le pavé de Chailly (Monet).jpg"
    assert _commons_file("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Monet_Le_Pave.JPG/960px-Monet_Le_Pave.JPG") == \
        "File:Monet Le Pave.JPG"
    assert _commons_file("https://upload.wikimedia.org/wikipedia/commons/2/21/Monet_Le_Pave.JPG") == "File:Monet Le Pave.JPG"
    assert _commons_file("https://example.com/a.jpg") == ""


def test_a_larger_copy_replaces_the_source_only_when_it_is_the_same_work_and_framing():
    from neural_canvas.larger import Candidate, LargerSource

    class Emb:                      # stands in for CLIP: the same picture scores 1, a different one 0
        def embed(self, ims):
            return [np.asarray(im.convert("L").resize((16, 16)), dtype=float).ravel() for im in ims]

        def similarity(self, a, bs):
            a = a / np.linalg.norm(a)
            return np.array([float(a @ (b / np.linalg.norm(b))) for b in bs])
    rng = np.random.default_rng(1)
    big = Image.fromarray(rng.integers(0, 255, (1200, 1650, 3), dtype=np.uint8)).resize((3300, 2400), Image.NEAREST)
    src = big.resize((850, 618))
    wrong_frame = big.crop((0, 0, 2400, 2400))
    ls = LargerSource(None, None, Emb(), {"id_clip_min": 0.86, "id_inliers_min": 25})
    ls.known = lambda st: [Candidate("https://a/square.jpg", "known", 2400, 2400), Candidate("https://a/big.jpg", "known")]
    ls.searched = lambda a, t: []
    ls._fetch = lambda url: {"https://a/square.jpg": wrong_frame, "https://a/big.jpg": big}[url]
    im, rep = ls.find(src, {}, "Monet", "Le Pavé de Chailly", (3840, 2160), 4.0)
    assert im is not None and im.size == (3300, 2400) and rep.chosen["url"] == "https://a/big.jpg"
    assert rep.tried[0]["verdict"].startswith("framing differs") and rep.needed > 4.0


# ── 2026-11-10: complete cards (WikiArt labels, Wikipedia infobox, Commons Artwork template, life, movement) ──

def test_wikiart_labels_no_longer_run_on_into_the_date():
    from neural_canvas.catalogue import labelled
    f = labelled("Date: 1905 - 1907 Style: Art Nouveau (Modern) Genre: landscape Media: oil, canvas "
                 "Location: Private Collection Dimensions: 110 x 110 cm")
    assert f["Date"] == "1905 - 1907" and f["Media"] == "oil, canvas" and f["Style"] == "Art Nouveau (Modern)"
    assert f["Location"] == "Private Collection"


def test_the_wikipedia_infobox_and_the_commons_artwork_template_are_read():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue)
    infobox = ('<table class="infobox vevent"><tr><th class="infobox-label">Year</th><td class="infobox-data">1948'
               '<sup class="reference">[1]</sup></td></tr><tr><th class="infobox-label">Medium</th><td>Egg tempera on '
               'gessoed panel</td></tr><tr><th class="infobox-label">Location</th><td><style>.x{}</style><a>Museum of '
               'Modern Art</a>, New York City</td></tr></table>')
    commons = ('<td id="fileinfotpl&#95;art&#95;medium" class="fileinfo-paramfield">Medium</td> <td> <a>oil on canvas</a>'
               '<div style="display: none;">medium QS:P186,Q296955</div></td> <td id="fileinfotpl&#95;art&#95;gallery">'
               'Current location</td> <td>Musée d&#39;Orsay</td> <td id="fileinfotpl&#95;art&#95;authority">Authority</td>'
               '<td>: Q17493132 Joconde</td>')
    cat._get = lambda url, **p: {"parse": {"text": infobox if "wikipedia" in url else commons}}
    box = cat._infobox("Christina's World")
    assert box == {"year": "1948", "medium": "Egg tempera on gessoed panel", "location": "Museum of Modern Art, New York City"}
    art = cat._commons_artwork("Le pavé de Chailly (Monet).jpg")
    assert art["art_medium"] == "oil on canvas" and art["art_gallery"] == "Musée d'Orsay" and art["item"] == "Q17493132"


def test_a_commons_template_filled_from_the_runs_wikidata_item_is_not_a_second_witness():
    from neural_canvas.catalogue import record_items
    from neural_canvas.evidence import Ledger
    wd = _rec("wikidata", "Le Pavé de Chailly", medium="oil on canvas", collection="Musée d'Orsay")
    wd.url = "https://www.wikidata.org/wiki/Q17493132"
    echo = _rec("commons.wikimedia.org", "Le Pavé de Chailly", medium="oil on canvas", collection="Musée d'Orsay",
                wikidata="Q17493132", echo=["date", "medium", "collection"])
    own = _rec("commons.wikimedia.org", "Le Pavé de Chailly", medium="oil on canvas", wikidata="Q999")
    local = _rec("commons.wikimedia.org", "Le Pavé de Chailly", medium="oil on canvas", date="1860s",
                 wikidata="Q17493132", echo=["date"])       # its medium is typed on the file page: its own
    led3 = Ledger([])
    record_items(led3, [wd, local])
    assert led3.items[1].claims.get("medium") == "oil on canvas" and not led3.items[1].claims.get("date")
    led = Ledger([])
    record_items(led, [wd, echo])
    assert not led.items[1].claims.get("medium") and not led.items[1].claims.get("collection")
    led2 = Ledger([])
    record_items(led2, [own])
    assert led2.items[0].claims["medium"] == "oil on canvas"


def test_the_artists_life_is_ruled_from_their_own_wikidata_item_when_the_works_is_not_found():
    from neural_canvas.catalogue import Catalogue, Record
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    artist = Record("wikidata", "https://www.wikidata.org/wiki/Q547055", "", [], "Carl Bloch", authority=True,
                    artist_born="1834", artist_died="1890", artist_nationality="Denmark", artist_nationalities=["Denmark"])
    rkd = Record("rkd.nl", "https://rkd.nl/artists/1", "", [], "Carl Bloch", "Danish, 1834–1890", authority=True,
                 artist_born="1834", artist_died="1890", artist_nationality="Danish")
    page = _rec("artvee.com", "Christus Consolator", artist="Carl Bloch", artist_bio="Danish, 1834-1890")
    ru = cat.rule([artist, rkd, page], {"artist": "Carl Bloch"})
    assert ru["facts"]["artist_born"] == "1834" and ru["facts"]["artist_died"] == "1890"
    assert ru["facts"].get("artist_nationality") == "Danish"


def test_a_movement_stated_by_two_sources_in_different_words_is_ruled():
    from neural_canvas.catalogue import Catalogue
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    a = _rec("wikidata", "Bauerngarten", artist="Gustav Klimt", movement="Art Nouveau")
    b = _rec("wikiart.org", "Bauerngarten", artist="Gustav Klimt", movement="Art Nouveau (Modern)")
    ru = cat.rule([a, b], {"artist": "Gustav Klimt"})
    assert ru["facts"]["movement"] == "Art Nouveau"


def test_page_furniture_and_unknown_values_are_not_read_as_fields():
    from neural_canvas.catalogue import labelled
    f = labelled("Media: oil, canvas Location: unknown SHOP FOR POSTERS AND MORE FROM JUST $24 Share: Tags: flowers")
    assert f["Media"] == "oil, canvas" and "Location" not in f
    g = labelled("Location: Private collection SHOP FOR HANDMADE OIL PAINTING REPRODUCTIONS")
    assert g["Location"] == "Private collection"


def test_a_medium_that_names_no_material_never_reaches_the_card():
    from neural_canvas.catalogue import record_items
    from neural_canvas.evidence import Ledger
    led = Ledger([])
    record_items(led, [_rec("commons.wikimedia.org", "Christus Consolator", medium="painting")])
    assert not led.items[0].claims.get("medium")


def test_an_identification_made_before_the_ledger_still_gives_the_catalogue_its_verified_copy():
    from neural_canvas.catalogue import identification_ledger
    old = {"tier": "oeuvre", "reference": "File:Hilma af Klint - Group IV No 2.jpg", "confirmed": True, "clip": 0.9135,
           "inliers": 796, "page_url": "https://commons.wikimedia.org/wiki/File:Hilma.jpg"}
    led = identification_ledger(old)
    assert led.verified() and led.items[0].inliers == 796
    assert not identification_ledger({**old, "confirmed": False}).items         # an unconfirmed match is no copy


def test_kunstindeks_danmark_records_print_in_english():
    from neural_canvas.catalogue import Catalogue, kid_medium
    assert kid_medium("Olie på lærred") == "Oil on canvas" and kid_medium("Radering") == "Etching"
    rec = Catalogue._kid_rec({"id": "fab8", "title": "Kristus i Gethsemane have", "creatorsNamesExact": ["Carl Bloch"],
                              "creatorsNationalities": ["Dansk"], "creatorsBirthDates": ["1834-05-23"],
                              "creatorsDeathDates": ["1890-02-22"], "productionStartString": "1879",
                              "productionEndString": "1879", "techniquePosition": ["Olie på lærred"],
                              "museum": "Ny Carlsberg Glyptotek"})
    assert (rec.artist, rec.artist_bio, rec.date, rec.medium, rec.collection) == \
        ("Carl Bloch", "Danish, 1834–1890", "1879", "Oil on canvas", "Ny Carlsberg Glyptotek")
    assert rec.url.endswith("/samlinger/fab8") and rec.overrule


def test_first_sight_backfills_and_an_alias_is_not_indexed_twice(tmp_path):
    import threading
    from types import SimpleNamespace
    from neural_canvas.indexer import ArtistIndexer
    ix = ArtistIndexer(tmp_path, lambda: None)
    ix.mark("Carl Heinrich Bloch", {"qid": "Q547055", "wikidata": 69})
    assert ix.backfill(["Carl Bloch", "Carl Heinrich Bloch", "", "Carl Bloch"]) == 1       # queued once; the indexed one not
    stop = threading.Event()
    ix._docent = SimpleNamespace(identifier=SimpleNamespace(wiki=SimpleNamespace(find_artist=lambda a: "Q547055")))
    ix.index_artist = lambda *a, **k: (_ for _ in ()).throw(AssertionError("an alias must not be re-indexed"))
    orig = ix._finish
    ix._finish = lambda a, r: (orig(a, r), stop.set())
    ix.run(stop, idle_s=0)
    done = ix._load(ix.done_path, {})
    assert done["carl bloch"]["alias_of"] == "Carl Heinrich Bloch" and ix._load(ix.queue_path, []) == []


def test_the_whole_works_listing_is_cached(tmp_path):
    from types import SimpleNamespace
    from neural_canvas.indexer import works_listing
    calls = []
    wiki = SimpleNamespace(works_by=lambda q: calls.append(q) or [{"qid": "Q2465911", "labels": ["Christina's World"], "file": ""}])
    assert works_listing(tmp_path, wiki, "Q316325")[0]["qid"] == "Q2465911"
    assert works_listing(tmp_path, wiki, "Q316325")[0]["qid"] == "Q2465911" and calls == ["Q316325"]   # second read: disk


def test_a_wikidata_name_kept_only_under_mul_is_still_read():
    from neural_canvas.references import Wikimedia
    assert Wikimedia.label({"labels": {"mul": {"value": "Félix Vallotton"}}}) == "Félix Vallotton"
    assert Wikimedia.label({"labels": {"en": {"value": "A"}, "mul": {"value": "B"}}}) == "A"


def test_the_artists_movement_is_ruled_when_wikidata_and_their_wikipedia_article_agree():
    from neural_canvas.catalogue import Catalogue, Record
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    who = Record("wikidata", "https://www.wikidata.org/wiki/Q34661", "", [], "Gustav Klimt", authority=True,
                 movement="Symbolism|Art Nouveau|Vienna Secession", artist_born="1862", artist_died="1918",
                 text="movement symbolism art nouveau vienna secession", article_movement="Symbolism, Art Nouveau")
    ru = \
        cat.rule([who, _rec("artvee.com", "Bauerngarten", artist="Gustav Klimt")], {"artist": "Gustav Klimt"})
    assert ru["facts"]["movement"] == "Symbolism"
    # Wikidata alone is no witness for the artist's movement: it gives Andrew Wyeth "French Realism" (2026-11-12)
    wyeth = Record("wikidata", "u", "", [], "Andrew Wyeth", authority=True, movement="French Realism", artist_born="1917",
                   text="movement realism regionalism", article_movement="Realism, Regionalism")
    ru = cat.rule([wyeth], {"artist": "Andrew Wyeth"})
    assert ru["facts"]["movement"] == "Realism"                 # an article that says Realism confirms the core
    magritte = Record("wikidata", "u", "", [], "René Magritte", authority=True, movement="Belgian surrealism",
                      artist_born="1898", text="movement surrealism", article_movement="Surrealism")
    assert cat.rule([magritte], {"artist": "René Magritte"})["facts"]["movement"] == "Surrealism"
    wrong = Record("wikidata", "u", "", [], "Andrew Wyeth", authority=True, movement="French Realism", artist_born="1917",
                   text="movement regionalist", article_movement="Regionalist")
    ru = cat.rule([wrong], {"artist": "Andrew Wyeth"})           # Wyeth's real article: the article's own movement
    assert ru["facts"]["movement"] == "Regionalism" and ru["support"]["movement"] == ["wikipedia (artist)"]
    surreal = Record("wikidata", "u", "", [], "René Magritte", authority=True, movement="Surrealism", artist_born="1898",
                     text="movement surrealist", article_movement="Surrealist")
    assert cat.rule([surreal], {"artist": "René Magritte"})["facts"]["movement"] == "Surrealism"
    nabi = Record("wikidata", "u", "", [], "Félix Vallotton", authority=True, movement="Les Nabis|Symbolism", artist_born="1865")
    ru = cat.rule([nabi], {"artist": "Félix Vallotton"})        # the article names no movement: Wikidata's first
    assert ru["facts"]["movement"] == "Les Nabis" and ru["support"]["movement"] == ["wikidata (artist)"]


def test_an_unlisted_label_ends_a_field():
    from neural_canvas.catalogue import labelled
    assert labelled("Style: Northern Renaissance Series: The Ghent Altarpiece")["Style"] == "Northern Renaissance"
    assert labelled("Style: Realism Theme: Angel Genre: religious")["Style"] == "Realism"


def test_a_kid_study_or_copy_is_not_the_work():
    from neural_canvas.catalogue import KID_VERSION
    assert KID_VERSION.search("Kristus i Gethsemane have Forarbejde til alterbilledet i St. Hans Kirke")
    assert KID_VERSION.search("Prometheus' ørn. Skitse")
    assert not KID_VERSION.search("Christus i Gethsemane")


def test_a_google_photos_write_waits_out_a_429_and_retries(monkeypatch):
    from types import SimpleNamespace
    from neural_canvas import deliver
    monkeypatch.setattr(deliver.time, "sleep", lambda s: None)
    replies = [SimpleNamespace(status_code=429, headers={"Retry-After": "3"}, raise_for_status=lambda: None),
               SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None, text="ok")]
    calls = []
    s = SimpleNamespace(post=lambda url, **kw: calls.append(url) or replies.pop(0))
    gp = deliver.GooglePhotos.__new__(deliver.GooglePhotos)
    assert gp._post(s, "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate").text == "ok" and len(calls) == 2


def test_a_card_prints_one_clean_movement():
    from neural_canvas.catalogue import clean_movement
    assert clean_movement("Art Nouveau (Modern) , Symbolism") == "Art Nouveau"
    assert clean_movement("pointillism") == "Pointillism" and clean_movement("") == ""


def test_a_date_range_prints_with_an_en_dash_and_a_long_title_wraps_evenly():
    from neural_canvas.placard import _layout, date_range
    assert date_range("1895 - 1917") == "1895–1917" and date_range("c. 1891") == "c. 1891"
    lines, fails = _layout({"artist": "Alphonse Mucha", "title": "Whitman's chocolates and confections. Philadelphia",
                            "date": "1895 - 1917", "detail": True}, 0.3, 221)
    text = [t for t, _, _ in lines if t]
    assert not fails and "(detail)" not in text and any("1895–1917" in t for t in text)


def test_a_living_painters_life_and_nationality_are_read_beside_the_name_and_most_sites_win():
    # ground truth, 2026-11-12: Fink, Sanchez, Tomioka and Musil have no Wikidata item; their pages print the life
    from neural_canvas.catalogue import life_from_ledger
    from neural_canvas.evidence import Ledger
    led = Ledger()
    led.add("engine", "omkar", "Artwork by Waldemar Theophil Fink (Swiss, 1893\u20131948)", "", "x.com")
    led.add("copy", "yandex", "Sold at Auction: Waldemar Fink, FINK, WALDEMAR (1883 Bern 1948) Abendstimmung",
            "https://www.invaluable.com/auction-lot/fink-1", verified=True, clip=0.95, inliers=80)
    led.add("caption", "omkar", "Waldemar Fink (Swiss Artist, 1883\u20131948) ...", "https://www.instagram.com/p/1/")
    led.add("caption", "omkar", "Fink | Evening in Adelboden (1913)", "https://primaopoi.tumblr.com/post/1")
    led.add("caption", "bing", "Waldemar Fink (Swiss, 1883-1948) canvas print", "https://www.etsy.com/listing/1")  # a shop
    got = life_from_ledger(led.items, "Waldemar Fink")
    assert (got["born"], got["died"], got["nationality"]) == ("1883", "1948", "Swiss")
    assert set(got["life_ids"]) == {"E2", "E3"}
    led2 = Ledger()
    led2.add("engine", "serpapi", "Tom\u00e1s S\u00e1nchez (Cuban b. 1948), Finding the Meditator", "https://www.christies.com/en/lot/lot-1")
    assert life_from_ledger(led2.items, "Tomás Sánchez")["born"] == "1948"
    assert life_from_ledger(led2.items, "Tomás Sánchez")["died"] == "" and life_from_ledger(led2.items, "Tomás Sánchez")["nationality"] == "Cuban"
    led3 = Ledger()
    led3.add("engine", "omkar", "Soichiro Tomioka, Japanese, 1922\u20131994, born in Takada", "https://www.facebook.com/p/1")
    led3.add("engine", "omkar", "Tomioka Soichiro, Japan (1922-1994) Trees (1961)", "https://www.instagram.com/p/2/")
    got = life_from_ledger(led3.items, "Soichiro Tomioka")
    assert (got["born"], got["died"], got["nationality"]) == ("1922", "1994", "Japanese")


def test_a_verified_copy_on_a_lot_page_makes_the_collection_private():
    from neural_canvas.catalogue import sale_page
    assert sale_page("https://www.christies.com/en/lot/lot-6293966")
    assert not sale_page("https://www.christies.com/en/stories/painting-the-french-riviera-at-the-royal-academy-d24a")
    assert sale_page("https://www.kollerauktionen.ch/de/319324-0001-----5039-FINK.htm")
    assert sale_page("https://www.invaluable.com/auction-lot/fink-waldemar-1883-bern-1948")
    assert not sale_page("https://artmuseum.princeton.edu/art/collections/objects/99647")
    assert not sale_page("")


def test_a_living_painter_prints_born():
    from neural_canvas.placard import _layout
    rows, _ = _layout({"artist": "Tom\u00e1s S\u00e1nchez", "artist_nationality": "Cuban", "artist_born": "1948",
                       "title": "Finding the Meditator"}, 1.0, 2000)
    assert any(t == "Cuban, b. 1948" for t, *_ in rows)


def test_an_edition_has_no_single_holder():
    from neural_canvas.catalogue import in_editions
    assert in_editions("Colour lithograph") and in_editions("woodblock print") and not in_editions("Oil on board")


def test_one_painter_prints_one_life_and_nationality_on_every_card(tmp_path):
    # Maggiori (2026-11-12): voted work by work he printed "French" twice and "American" once; Tumblr's
    # "French-American" had been read as "French"
    from neural_canvas.catalogue import ArtistLives, life_votes
    from neural_canvas.evidence import Ledger
    a, b = Ledger(), Ledger()
    a.add("caption", "omkar", "Mark Maggiori (French-American,b.1977) Under The Eye Of Yuttahih", "https://x.tumblr.com/post/1")
    a.add("caption", "omkar", "Mark Maggiori (French, b. 1977) Journey", "https://www.artsy.net/artwork/1")
    b.add("copy", "bing", "Mark Maggiori (American, b. 1977) Sonoran Magnetism, 2022", "https://www.pinterest.com/pin/1/",
          verified=True, clip=0.95, inliers=80)
    assert life_votes(a.items, "Mark Maggiori")[1]["tumblr"][0] == "French-American"
    store = ArtistLives(tmp_path)
    first = store.merge("Mark Maggiori", "run1", *life_votes(a.items, "Mark Maggiori"))
    second = store.merge("Mark Maggiori", "run2", *life_votes(b.items, "Mark Maggiori"))
    assert first["born"] == second["born"] == "1977"
    assert second["nationality"] in ("French", "French-American", "American")
    again = store.merge("Mark Maggiori", "run1", *life_votes(a.items, "Mark Maggiori"))
    assert again["nationality"] == second["nationality"]          # every card reads the same pooled vote


def test_the_works_year_is_read_after_its_title_and_never_before_the_painter_could_paint():
    from neural_canvas.catalogue import date_from_ledger
    from neural_canvas.evidence import Ledger
    led = Ledger()
    led.add("engine", "omkar", "Richard Savoie, Canada (1959-) There Was a Time (2015) ...", "https://www.instagram.com/p/1/")
    led.add("engine", "omkar", "“There Was A Time” Richard Savoie (b. 1959) 2015", "https://www.facebook.com/p/2")
    led.add("caption", "bing", "Richard Savoie - There Was a Time 1959", "https://www.reddit.com/r/Art/1")   # the life misread
    assert date_from_ledger(led.items, "Richard Savoie", "There Was a Time", "1959") == ("2015", ["E1"])
    assert date_from_ledger(led.items, "Someone Else", "There Was a Time", "1959") == ("", [])


def test_an_artist_described_by_their_movement_or_as_a_naturalist_is_found():
    from neural_canvas.references import Wikimedia
    w = Wikimedia.__new__(Wikimedia); w.wikidata_api = "x"
    def found(*hits):
        w._get = lambda url, params: {"search": [{"id": i, "description": d} for i, d in hits]}
        return w.find_artist("x")
    assert found(("Q7836", "Belgian surrealist (1898–1967)"), ("Q1", "1965 exhibition")) == "Q7836"
    assert found(("Q2", "American zoologist")) == "Q2"
    assert found(("Q3", "American zoologist"), ("Q4", "American painter")) == "Q4"     # the artist first
    assert found(("Q5", "Surgeon-General of the UK")) is None


def test_an_oil_painting_prints_its_material():
    from neural_canvas.catalogue import card_medium
    assert card_medium("Oil painting") == "Oil" and card_medium("watercolor painting") == "Watercolor"
    assert card_medium("painting") == "painting" and card_medium("Oil on canvas") == "Oil on canvas"
    assert card_medium("Oil paint on canvas") == "Oil on canvas" and card_medium("oil paint on panel") == "Oil on panel"
    assert card_medium("oil, canvas") == "Oil on canvas" and card_medium("oil, copper") == "Oil on copper"
    assert card_medium("Watercolor paint on wove paper") == "Watercolor on wove paper"
    assert card_medium("oil on canvas") == "Oil on canvas"
    assert card_medium("ink, watercolor, paper") == "Ink, watercolor, paper"     # only a simple pair is reworded


def test_a_movement_inside_another_never_confirms_it():
    # Matisse (2026-11-12): Wikidata's "Impressionism" was confirmed by the article's "Post-Impressionism"
    from neural_canvas.catalogue import Catalogue, Record
    cat = Catalogue.__new__(Catalogue); cat.consulted = []
    matisse = Record("wikidata", "u", "", [], "Henri Matisse", authority=True, artist_born="1869",
                     movement="Impressionism|Fauvism|Modernism", article_movement="Fauvism, Modernism, Post-Impressionism",
                     text="movement fauvism modernism post impressionism")
    ru = cat.rule([matisse], {"artist": "Henri Matisse"})
    assert ru["facts"]["movement"] == "Fauvism" and ru["support"]["movement"] == ["wikipedia (artist)", "wikidata (artist)"]


def test_a_movement_prints_as_a_proper_noun():
    from neural_canvas.catalogue import clean_movement
    assert clean_movement("American realism") == "American Realism"
    assert clean_movement("Hudson River school") == "Hudson River School"
    assert clean_movement("post-impressionism") == "Post-Impressionism" and clean_movement("ukiyo-e") == "Ukiyo-e"
    assert clean_movement("Shin-hanga") == "Shin-hanga" and clean_movement("pre-raphaelite") == "Pre-Raphaelite"
    assert clean_movement("Founder of Tonalism") == "Tonalism" and clean_movement("co-founder of the Nabis") == "Nabis"
    assert clean_movement("Italian Baroque Painting") == "Italian Baroque" and clean_movement("Abstract Art") == "Abstract Art"
    assert clean_movement("Les Nabis") == "Les Nabis" and clean_movement("Art Nouveau (Modern) , Symbolism") == "Art Nouveau"


def test_wikidatas_worded_ranges_print_as_ranges():
    from neural_canvas.placard import date_range
    assert date_range("between 1600 and 1620") == "1600–1620" and date_range("from 1873 until 1874") == "1873–1874"
    assert date_range("1895 - 1917") == "1895–1917" and date_range("c. 1905") == "c. 1905"


def test_a_printmaker_with_no_medium_is_not_given_a_private_collection():
    from neural_canvas.catalogue import _PRINTMAKER
    assert _PRINTMAKER.search("Czechoslovak poster artist, lithographer, photographer")
    assert not _PRINTMAKER.search("French Impressionist painter")


def test_print_media_name_their_material():
    from neural_canvas.catalogue import _materials
    for m in ("chromolithograph", "serigraph", "aquatint", "linocut", "Oil on masonite"):
        assert _materials(m), m
    assert not _materials("Drawing") and not _materials("painting")


def test_a_collection_prints_without_acronym_or_address():
    from neural_canvas.catalogue import clean_collection
    assert clean_collection("Los Angeles County Museum of Art (LACMA), Los Angeles, CA, US") == "Los Angeles County Museum of Art"
    assert clean_collection("Cooper Hewitt, Smithsonian Design Museum") == "Cooper Hewitt, Smithsonian Design Museum"
    assert clean_collection("Albright-Knox Art Gallery, Buffalo, NY, US") == "Buffalo AKG Art Museum"
    assert clean_collection("Brandywine Conservancy and Museum of Art") == "Brandywine Museum of Art"
    assert clean_collection("Brandywine River Museum of Art") == "Brandywine Museum of Art"
    assert clean_collection("Impressionism: The Hasso Plattner Collection") == "Museum Barberini"
    # one holder, one name: no leading "The", no trailing city, unless the name needs it
    assert clean_collection("The Metropolitan Museum of Art") == "Metropolitan Museum of Art"
    assert clean_collection("Museum of Modern Art, New York") == "Museum of Modern Art"
    assert clean_collection("Musée d'Orsay, Paris") == "Musée d'Orsay"
    assert clean_collection("Museum of Fine Arts, Boston, Massachusetts") == "Museum of Fine Arts, Boston"
    assert clean_collection("Huntington Library, Art Museum, and Botanical Gardens") == \
        "Huntington Library, Art Museum, and Botanical Gardens"
    assert clean_collection("National History Museum, Frederiksborg Castle, Denmark") == \
        "Museum of National History at Frederiksborg Castle"
    assert clean_collection("Austrian Gallery Belvedere") == "Belvedere"
    assert clean_collection("Private collection") == "Private collection"
    assert clean_collection("museum collection of the Prague City Gallery") == "Prague City Gallery"
    assert clean_collection("The Wallace Collection") == "Wallace Collection"   # its name survives, as Wikidata prints it


def test_one_painter_one_name_and_never_a_namesake():
    from neural_canvas.catalogue import same_person
    assert same_person("Carl Bloch", "Carl Heinrich Bloch") and same_person("N. C. Wyeth", "Newell Convers Wyeth")
    assert same_person("Edgar Degas", "Hilaire Germain Edgar Degas") and same_person("James Tissot", "James Jacques Joseph Tissot")
    assert not same_person("Andrew Wyeth", "Newell Convers Wyeth") and not same_person("John Singer Sargent", "Richard Sargent")


def test_a_file_names_furniture_is_never_a_search_lead():
    from neural_canvas.catalogue import leads
    got = leads({"title": "Jesus Cleansing the Temple", "evidence": {"web_references": [
        "file: default | page: default | site: smk.dk", "file: main | page: Jesus Cleansing the Temple | site: artvee.com"]}})
    assert "default" not in got and "main" not in got and "Jesus Cleansing the Temple" in got


def test_movement_start_years_are_keyed_as_folded_names():
    from neural_canvas.catalogue import MOVEMENT_START, fold
    assert MOVEMENT_START[fold("De Stijl")] == 1917 and MOVEMENT_START[fold("Post-Impressionism")] == 1878


def test_a_reproductions_process_is_never_the_works_medium():
    from neural_canvas.catalogue import _REPRODUCTION
    assert _REPRODUCTION.search("photomechanical print") and _REPRODUCTION.search("Giclée on canvas")
    assert not _REPRODUCTION.search("chromolithograph") and not _REPRODUCTION.search("Oil on canvas")


def test_one_painter_keeps_one_nationality_wording(tmp_path):
    from neural_canvas.catalogue import _kept_nationality
    assert _kept_nationality(tmp_path, "Paul Klee") == ""
    assert _kept_nationality(tmp_path, "Paul Klee", "German-Swiss") == "German-Swiss"
    assert _kept_nationality(tmp_path, "Paul Klee", "Swiss-German") == "German-Swiss"     # the first wording stays


def test_a_title_keeps_its_sources_accents():
    from neural_canvas.catalogue import accented
    pages = ["Claude Monet, Nympheas, 1897", "Nymphéas (Water Lilies), LACMA", "Monet — Nymphéas, c. 1897–98"]
    assert accented("Nympheas", pages) == "Nymphéas"
    assert accented("NYMPHEAS", pages) == "NYMPHÉAS"                       # the ruled form's case stays
    assert accented("Nympheas", pages[:2]) == "Nympheas"                   # one accented witness is not enough
    assert accented("Water Lilies", pages) == "Water Lilies"
    assert accented("Cafe", ["Le Café de nuit", "Café Terrace", "Cafeteria"]) == "Café"
    assert accented("Nymphéas", pages) == "Nymphéas"                       # an accented title is left alone


def test_one_painter_keeps_one_name(tmp_path):
    from neural_canvas.catalogue import _kept_name
    long, short = "José María Velasco Gómez Obregón", "José María Velasco"
    assert _kept_name(tmp_path, long) == long                     # no Wikidata yet: the first ruled name is kept
    assert _kept_name(tmp_path, short) == long
    assert _kept_name(tmp_path, long, short) == short             # Wikidata's label replaces a ruled name
    assert _kept_name(tmp_path, long) == short                    # and every later card prints it
    assert _kept_name(tmp_path, "José Velasco", "José Maria Velasco") == short   # the first label stays
    assert _kept_name(tmp_path, "Andrew Wyeth", "Andrew Wyeth") == "Andrew Wyeth"
    assert _kept_name(tmp_path, "N. C. Wyeth", "Newell Convers Wyeth") == "Newell Convers Wyeth"
    assert _kept_name(tmp_path, "Jamie Wyeth") == "Jamie Wyeth"   # another Wyeth is another painter
    assert _kept_name(tmp_path, "Wyeth") == "Wyeth"               # a bare surname matching several keeps its form
