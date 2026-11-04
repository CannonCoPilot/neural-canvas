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


def _interp(subjects, genre="genre-scene"):
    return Interpretation(genre, "", "", (0.5, 0.5), None, subjects, "")


def test_full_bleed_keeps_essential_and_is_16x9():
    img = _img(2000, 1500)   # 4:3 → crop vertically
    plan = Composer(3840, 2160, 0.97).plan(img, _interp([Subject("face", (0.4, 0.05, 0.6, 0.25), 5, True)]))
    assert plan.passed and plan.mode == "full-bleed"
    x0, y0, x1, y1 = plan.window_px
    assert abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.01
    assert plan.window[1] <= 0.05 + 1e-6            # slid up to keep the face


def test_tall_work_with_tall_subject_is_matted_not_cut():
    img = _img(1000, 1500)
    s = Subject("figure", (0.3, 0.1, 0.7, 0.95), 5, True)
    plan = Composer(3840, 2160, 0.97).plan(img, _interp([s]))
    assert plan.passed and plan.mode == "pillarbox"
    assert _retained(s.box, plan.window) >= 0.97
    assert plan.art_size[1] == 2160 and plan.art_size[0] < 3840


def test_matted_window_found_when_span_fills_it_exactly():
    # regression: the only feasible offset was a single point the step grid skipped
    img = _img(1035, 1535)
    plan = Composer(3840, 2160, 0.97).plan(img, _interp([Subject("girl", (0.3, 0.35, 0.7, 0.85), 5, True)]))
    assert plan.passed, plan.failures


def test_wide_panorama_letterboxed_when_essentials_span_width():
    img = _img(4000, 1000)
    subs = [Subject("a", (0.02, 0.3, 0.1, 0.7), 5, True), Subject("b", (0.9, 0.3, 0.98, 0.7), 5, True)]
    plan = Composer(3840, 2160, 0.97).plan(img, _interp(subs))
    assert plan.passed and plan.mode == "letterbox"


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


def test_label_moves_to_mat_when_every_corner_covers_a_protagonist():
    from neural_canvas.crop import CropPlan
    from neural_canvas.placard import compose
    art = _img(3840, 2160)
    plan = CropPlan("full-bleed", (0, 0, 1, 1), (0, 0, 3840, 2160), (3840, 2160), (0, 0), passed=True)
    ident = {"artist": "A. Painter", "title": "Crowd", "date": "1900"}
    frame, pl = compose(art, plan, (3840, 2160), ident, "A paragraph. " * 8, [(0.0, 0.0, 1.0, 1.0)], 7.0)
    assert frame.size == (3840, 2160)
    assert pl.position == "mat-right" and pl.essential_overlap == 0 and pl.passed, pl.failures
