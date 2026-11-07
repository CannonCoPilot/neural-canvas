"""Direct Bing / Yandex reverse-image search: parsers and failure rules on saved pages (no network).

Fixtures in tests/fixtures/reverse_image/ are trimmed from live captures of 2026-11-06 (Leighton, "The Feigned
Death of Juliet"); the captcha pages are synthetic, reduced to the markers the parser checks."""
import html
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from neural_canvas import reverse_image as ri
from neural_canvas.reverse_image import ReverseImage, parse_bing, parse_yandex

FX = Path(__file__).parent / "fixtures" / "reverse_image"
BCID = "dummy000000000000000000000000000000"
BING_URL = f"https://www.bing.com/images/search?q=The%20Feigned%20Death%20Of%20Juliet%20Painting&bcid={BCID}&FORM=SBIESR"
CBIR = "2099415/B1G92n12Bvan5D6LUEWcUg1909"
YANDEX_URL = f"https://yandex.com/images/search?rpt=imageview&cbir_id={CBIR}&cbir_page=sites"


def fx(name):
    return (FX / name).read_text()


def yandex_page(**edit):
    """The Yandex fixture with its page state edited (key -> new value of initialState[key])."""
    page = fx("yandex_results.html")
    m = [m for m in re.finditer(r'data-state="([^"]*)"', page) if "initialState" in m.group(1)][0]
    s = json.loads(html.unescape(m.group(1)))
    s["initialState"].update(edit)
    return page[:m.start(1)] + html.escape(json.dumps(s, ensure_ascii=False), quote=True) + page[m.end(1):]


# ── Bing ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_bing_parser_reads_murl_purl_and_title_from_the_tiles():
    refs = parse_bing(fx("bing_results.html"), BING_URL, BCID)
    assert len(refs) == 3
    r = refs[0]
    assert r.image_url.startswith("https://c8.alamy.com/") and r.image_url.endswith("W3KXDC.jpg")
    assert r.page_url == "https://www.alamy.com/stock-photo/romeo-and-juliet-8.html"
    assert r.thumb_url.startswith("https://ts4.mm.bing.net/th?id=")
    assert r.file_title == "Romeo and juliet 8 hi-res stock photography and images - Alamy"
    assert r.meta == {"source": "bing", "site": "alamy.com"}
    assert r.page_id == hash(r.image_url) & 0x7FFFFFFFFFFF
    assert all("" not in x.file_title and "" not in x.file_title for x in refs)


def test_bing_homepage_raises_even_though_it_carries_tiles():
    with pytest.raises(RuntimeError, match="redirected"):
        parse_bing(fx("bing_home.html"), "https://www.bing.com/images", BCID)


def test_bing_captcha_raises():
    with pytest.raises(RuntimeError, match="captcha"):
        parse_bing(fx("bing_captcha.html"), BING_URL, BCID)
    with pytest.raises(RuntimeError, match="captcha"):
        parse_bing(fx("bing_results.html"), "https://www.bing.com/challenge/verify?x=1", BCID)


def test_bing_page_for_another_upload_or_without_a_grid_raises():
    with pytest.raises(RuntimeError, match="bcid"):
        parse_bing(fx("bing_results.html"), BING_URL.replace(BCID, "other"), BCID)
    with pytest.raises(RuntimeError, match="no result grid"):
        parse_bing("<html><title>x - Search Images</title></html>", BING_URL, BCID)


def test_bing_results_page_with_an_empty_grid_is_a_genuine_no_match():
    page = re.sub(r"<li data-idx.*</ul>", "</ul>", fx("bing_results.html"), flags=re.S)
    assert parse_bing(page, BING_URL, BCID) == []


# ── Yandex ────────────────────────────────────────────────────────────────────────────────────────────────
def test_yandex_parser_reads_sites_sizes_and_lookalikes_in_order():
    refs, tags = parse_yandex(fx("yandex_results.html"), YANDEX_URL, CBIR)
    kinds = [r.meta["kind"] for r in refs]
    assert kinds == sorted(kinds, key=["site", "size", "similar"].index)       # sites first, look-alikes last
    site = refs[0]
    assert site.image_url == "https://i.ytimg.com/vi/SapVYK-hiFU/maxresdefault.jpg"
    assert site.page_url == "https://www.youtube.com/watch?v=SapVYK-hiFU"      # utm_* tracking stripped
    assert site.thumb_url.startswith("https://avatars.mds.yandex.net/i?id=")   # Yandex's own thumb is KEPT
    assert site.file_title == "Folk Harp Music: Sleeping Alone - YouTube" and site.meta["source"] == "yandex"
    assert refs[1].file_title.startswith("Английские разночтения")             # Cyrillic titles kept as given
    sim = [r for r in refs if r.meta["kind"] == "similar"]
    assert sim and all(r.image_url.startswith("http") and "yandex" not in r.image_url for r in sim)  # img_url, not the thumb
    assert tags[0] == "the feigned death of juliet" and any(re.search("[а-я]", t) for t in tags)


def test_yandex_drops_only_our_own_upload_and_its_ui_art():
    refs, _ = parse_yandex(fx("yandex_results.html"), YANDEX_URL, CBIR)
    assert not any("get-images-cbir" in r.image_url or "yastatic.net" in r.image_url for r in refs)
    assert len({r.image_url for r in refs}) == len(refs)
    assert len([r for r in refs if r.meta["kind"] == "site"]) == 3


def test_yandex_captcha_raises():
    with pytest.raises(RuntimeError, match="captcha"):
        parse_yandex(fx("yandex_captcha.html"), "https://yandex.com/showcaptcha?retpath=x", CBIR)
    with pytest.raises(RuntimeError, match="captcha"):
        parse_yandex(fx("yandex_captcha.html"), YANDEX_URL, CBIR)


def test_yandex_homepage_shell_and_expired_pages_raise():
    with pytest.raises(RuntimeError, match="redirected"):
        parse_yandex(fx("yandex_results.html"), "https://yandex.com/images/", CBIR)
    with pytest.raises(RuntimeError, match="no results state"):
        parse_yandex("<html><title>Yandex Images</title></html>", YANDEX_URL, CBIR)
    with pytest.raises(RuntimeError, match="not registered"):        # the empty shell Imify called "no results"
        parse_yandex(yandex_page(cbirPreview={"cbirId": "", "isCbirEmpty": True}), YANDEX_URL, CBIR)
    with pytest.raises(RuntimeError, match="not registered"):
        parse_yandex(fx("yandex_results.html"), YANDEX_URL, "1/someone-else")
    with pytest.raises(RuntimeError, match="expired"):
        parse_yandex(yandex_page(cbirPreview={"cbirId": CBIR, "imageExpired": True}), YANDEX_URL, CBIR)


def test_yandex_genuine_page_with_nothing_found_is_empty():
    page = yandex_page(cbirSites={"sites": []}, cbirSimilar={"thumbs": []}, cbirOtherSizes={"items": {}})
    assert parse_yandex(page, YANDEX_URL, CBIR)[0] == []


# ── the client, over a fake transport ─────────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, text, url, status=200):
        self.text, self.url, self.status_code = text, url, status

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, routes):
        self.routes, self.headers = routes, {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _go(self, url):
        for prefix, resp in self.routes:
            if url.startswith(prefix):
                return resp
        raise AssertionError(url)

    def get(self, url, **kw):
        return self._go(url)

    def post(self, url, **kw):
        return self._go(url)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ri, "PACE_S", 0.0)
    def use(routes):
        monkeypatch.setattr(ri.requests, "Session", lambda: _Session(routes))
        return ReverseImage("test-agent")
    return use


IMG = Image.new("RGB", (64, 48), (120, 80, 40))


def test_bing_client_returns_references_and_the_caption(client):
    c = client([("https://www.bing.com/images/kblob", _Resp(fx("bing_kblob.json"), "")),
                ("https://www.bing.com/images/search", _Resp(fx("bing_results.html"), BING_URL)),
                ("https://www.bing.com/images", _Resp("", "https://www.bing.com/images"))])
    refs, caption = c.bing(IMG)
    assert len(refs) == 3 and caption == "The Feigned Death Of Juliet Painting"


def test_bing_client_raises_when_the_upload_is_refused_or_redirected_home(client):
    c = client([("https://www.bing.com/images/kblob", _Resp('{"error":"x"}', "")),
                ("https://www.bing.com/images", _Resp("", ""))])
    with pytest.raises(RuntimeError, match="upload not accepted"):
        c.bing(IMG)
    c = client([("https://www.bing.com/images/kblob", _Resp(fx("bing_kblob.json"), "")),
                ("https://www.bing.com/images/search", _Resp(fx("bing_home.html"), "https://www.bing.com/images")),
                ("https://www.bing.com/images", _Resp("", ""))])
    with pytest.raises(RuntimeError, match="redirected"):
        c.bing(IMG)


def test_yandex_client_returns_references_and_tags(client):
    c = client([("https://yandex.com/images-apphost/image-download", _Resp(fx("yandex_upload.json"), "")),
                ("https://yandex.com/images/search", _Resp(fx("yandex_results.html"), YANDEX_URL))])
    tags = []
    refs = c.yandex(IMG, tags=tags)
    assert refs[0].meta["kind"] == "site" and tags[0] == "the feigned death of juliet"


def test_yandex_client_raises_on_http_error_and_unregistered_upload(client):
    c = client([("https://yandex.com/images-apphost/image-download", _Resp("", "", status=429))])
    with pytest.raises(RuntimeError, match="yandex: HTTPError"):
        c.yandex(IMG)
    c = client([("https://yandex.com/images-apphost/image-download", _Resp("{}", ""))])
    with pytest.raises(RuntimeError, match="not registered"):
        c.yandex(IMG)


def test_calls_to_one_engine_are_paced(monkeypatch):
    import time
    monkeypatch.setattr(ri, "PACE_S", 0.3)
    c = ReverseImage("test-agent")
    t0 = time.monotonic()
    for _ in range(3):
        c._paced("bing", lambda: None)
    assert time.monotonic() - t0 >= 0.6
