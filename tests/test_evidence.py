"""The evidence ledger, the agents' round loop and the mechanical rules (Sir's redesign, 2026-11-07)."""
from neural_canvas.agents import Agents, rule_facts
from neural_canvas.evidence import Ledger, engine_texts, site_of


def test_engine_answers_are_read_from_every_providers_text_fields():
    # Visser (withheld bench): her name was only in fields the adapters dropped
    omkar = {"visual_matches": [{"title": "Fishers of bluebirds"}],
             "web_results": [{"title": "Hummingbird and fish - Suzan Visser (1967 -) : r/oilpaintings",
                              "snippet": "Modern Paintings (1900-present).", "link": "https://www.google.com/goto?url=x",
                              "source": "Reddit · r/oilpaintings"}], "entities": []}
    letscrape = {"data": {"knowledge_graph": [{"title": "Suzan Visser", "subtitle": "Dutch painter"}],
                          "organic_results": [{"title": "SUSAN VISSER - Roodborstje", "url": "/goto?url=y"}]}}
    serpapi = {"ai_overview": {"text_blocks": [{"snippet": "Hummingbird and Fish by Dutch painter Suzan Visser"}]}}
    texts = [t["text"] for r in (omkar, letscrape, serpapi) for t in engine_texts(r)]
    assert any("Suzan Visser (1967" in t and "Modern Paintings" in t for t in texts)
    assert "Suzan Visser — Dutch painter" in texts
    assert any(t.startswith("Google AI Overview:") for t in texts)
    assert not any("bluebirds" in t for t in texts)          # image matches are scored, not read here


def test_one_site_seen_through_two_engines_is_one_origin():
    led = Ledger()
    a = led.add("engine", "google_lens:rapidapi_omkar", "Hummingbird and fish - Suzan Visser",
                "https://www.google.com/goto?url=x", "Reddit · r/oilpaintings")
    b = led.add("copy", "google_lens:apify", "Hummingbird and fish - Suzan Visser (1967 -)",
                "https://www.reddit.com/r/museum/comments/1", verified=True, clip=0.95, inliers=80)
    assert a.url == "" and a.origin == b.origin == "reddit"   # an opaque Google hop names no page
    assert led.origins([a.id, b.id]) == {"reddit"}
    assert site_of("https://collections.artuk.org.uk/x") == "artuk.org.uk"
    assert led.add("engine", "google_lens:rapidapi_omkar", "  Hummingbird and  fish - Suzan Visser ",
                   "https://www.google.com/goto?url=z", "Reddit") is a      # spacing and the hop do not make a new item
    n = len(led.items)
    led.add("copy", "bing", "Hummingbird and fish - Suzan Visser (1967 -)", "https://www.reddit.com/r/museum/comments/1")
    assert len(led.items) == n                               # the same page twice is one item


def test_a_caption_upgrades_to_a_verified_copy_in_place():
    led = Ledger()
    c = led.add("copy", "yandex", "Tower of Goubing", "https://example.org/a", verified=False, clip=0.7)
    c2 = led.add("copy", "yandex", "Tower of Goubing", "https://example.org/a", verified=True, clip=0.93, inliers=60)
    assert c2 is c and c.verified and c.inliers == 60


def _visser_ledger():
    led = Ledger()
    led.add("copy", "google_lens:apify", "Hummingbird and fish - Suzan Visser (1967 -)", "https://www.reddit.com/r/museum/1",
            verified=True, clip=0.95, inliers=80)                                                              # E1
    led.add("engine", "google_lens:omkar", "Hummingbird and fish - Suzan Visser (1967 -) : r/oilpaintings", "",
            "Reddit · r/oilpaintings")                                                                         # E2
    led.add("copy", "google_lens:apify", "Hummingbird and Fish. Painted by Suzan Visser (1967– ), 2019", "https://suzanvisser.art/work/1",
            verified=True, clip=0.94, inliers=70)                                                              # E3
    led.add("lead", "model", "model guess: Sandra Younes")                                                    # E4
    return led


def _ruling(artist_cites, title_cites, artist="Suzan Visser", title="Hummingbird and Fish"):
    return {"artist": {"value": artist, "cites": artist_cites}, "title": {"value": title, "cites": title_cites}}


def _check(ids):
    return {f: {"supports": ids, "unsupported": [], "contradicts": []} for f in ("artist", "title")}


def test_a_fact_needs_two_independent_origins_and_leads_never_count():
    led = _visser_ledger()
    one_site = rule_facts(led, _ruling(["E1", "E2"], ["E1", "E2"]), _check(["E1", "E2"]))
    assert "artist" not in one_site["facts"] and one_site["failures"]          # Reddit twice is one witness
    with_lead = rule_facts(led, _ruling(["E1", "E4"], ["E1", "E4"]), _check(["E1", "E4"]))
    assert "artist" not in with_lead["facts"]                                  # a model guess is not a witness
    ok = rule_facts(led, _ruling(["E1", "E2", "E3"], ["E1", "E3"]), _check(["E1", "E2", "E3"]))
    assert ok["facts"] == {"artist": "Suzan Visser", "title": "Hummingbird and Fish"} and not ok["failures"]
    assert ok["support"]["artist"] == ["E1", "E2", "E3"]


def test_words_decide_where_they_can_and_the_checker_only_where_they_cannot():
    led = _visser_ledger()
    chk = _check(["E1"])
    chk["artist"]["unsupported"] = ["E3"]
    v = rule_facts(led, _ruling(["E1", "E3"], ["E1", "E3"]), chk)
    assert v["facts"].get("artist") == "Suzan Visser"          # its words carry the name: the 8b veto does not stand
    silent = rule_facts(led, _ruling(["e1", 3], ["E1", "E3"]), _check(["E1"]))   # E3 unconfirmed, not rejected
    assert silent["facts"].get("artist") == "Suzan Visser"
    cyr = led.add("copy", "yandex", "Сюзан Виссер колибри и рыба", "https://ya.ru/1", verified=True, clip=0.9, inliers=50)
    v2 = rule_facts(led, _ruling(["E1", cyr.id], ["E1", "E3"]), _check(["E1"]))
    assert "artist" not in v2["facts"]                          # another script needs the checker's word


def test_no_verified_copy_means_no_identification_whatever_the_text_says():
    led = Ledger()
    led.add("engine", "bing", "Bing's best guess: Suzan Visser hummingbird")
    led.add("caption", "yandex", "Suzan Visser - Hummingbird and fish", "https://www.pinterest.com/pin/1")
    v = rule_facts(led, _ruling(["E1", "E2"], ["E1", "E2"]), _check(["E1", "E2"]))
    assert any("no copy of the picture verified" in f for f in v["failures"])


def test_card_facts_need_a_record_and_a_curated_record_can_veto():
    led = _visser_ledger()
    led.add("record", "wikidata", "Wikidata item Q1: date 1890", "https://www.wikidata.org/wiki/Q1", overrule=True,
            claims={"artist": "Sandra Younes", "date": "1890"})                                                    # E5
    led.add("copy", "bing", "Suzan Visser, Hummingbird and Fish (2019)", "https://www.saatchiart.com/art/1", verified=True,
            clip=0.95, inliers=70)                                                                                 # E6
    ruling = {**_ruling(["E1", "E3"], ["E1", "E3"]), "date": {"value": "2019", "cites": ["E3", "E6"]}}
    chk = {**_check(["E1", "E3"]), "date": {"supports": ["E3", "E6"], "contradicts": []}}
    v = rule_facts(led, ruling, chk, record_facts=("date",))
    # one reliable witness (Sir, 2026-11-10): a verified copy stating the date rules it, with no catalogue record
    assert v["facts"]["date"] == "2019" and {"E3", "E6"} <= set(v["support"]["date"])
    chk["artist"]["contradicts"] = ["E5"]
    v2 = rule_facts(led, ruling, chk)
    assert "artist" not in v2["facts"] and "curated record" in v2["unruled"]["artist"]


class _Model:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def ask_json(self, prompt, images=None, **kw):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_rounds_gather_more_evidence_and_stop_when_nothing_new_comes():
    led = _visser_ledger()
    adj = _Model([_ruling(["E1", "E2"], ["E1"]) | {"searches": ["Suzan Visser Hummingbird and fish"]},
                  _ruling(["E1", "E2", "E5"], ["E1", "E5"])])
    chk = _Model([_check(["E1", "E2"]) | {"verdict": "revise"}, _check(["E1", "E2", "E5"]) | {"verdict": "accept"}])
    asked = []

    def search(ruling, queries):
        asked.append(queries)
        led.add("copy", "agent_search", "Suzan Visser, Hummingbird and fish, oil", "https://suzanvisser.art/work/1",
                verified=True, clip=0.96, inliers=90)                                                         # E5
        return 1
    v = Agents(adj, chk).settle(None, led, search=search)
    assert asked == [["Suzan Visser Hummingbird and fish"]] and not v["failures"] and len(v["rounds"]) == 2
    assert "E5" in adj.prompts[1] and "VERIFIED COPY" in adj.prompts[0]

    adj2 = _Model([_ruling(["E1"], ["E1"], "Sandra Younes"), _ruling(["E1", "E2", "E3"], ["E1", "E3"])])
    fixed = Agents(adj2, _Model([_check(["E1"]), _check(["E1", "E2", "E3"])])).settle(None, _visser_ledger(),
                                                                                       search=lambda r, q: 0)
    assert not fixed["failures"] and fixed["rounds"][0]["new_items"] == 0       # re-ruled on the rules' feedback
    assert "PREVIOUS RULING FAILED" in adj2.prompts[1] and "Sandra Younes" in adj2.prompts[1]
    stuck = Agents(_Model([_ruling(["E1"], ["E1"])] * 2), _Model([_check(["E1"])] * 2)).settle(None, _visser_ledger(),
                                                                                              search=lambda r, q: 0)
    assert stuck["failures"] and len(stuck["rounds"]) == 2                       # the cap ends it, still unruled


def test_catalogue_records_join_the_ledger_as_witnesses():
    from neural_canvas.catalogue import Record, record_items
    led = _visser_ledger()
    recs = [Record("wikidata", "https://www.wikidata.org/wiki/Q9", "The Tower of Goubing", artist="Félix Vallotton",
                   date="1919", overrule=True, verified_by="image"),
            Record("rkd.nl", "https://rkd.nl/a", "", artist="Félix Vallotton", authority=True),
            Record("christies.com", "https://www.christies.com/lot/1", "Tower", support=True)]
    assert record_items(led, recs) == 2                       # the authority record rules the life only
    rec = next(i for i in led.items if i.source == "wikidata")
    assert rec.kind == "record" and rec.overrule and rec.claims["date"] == "1919" and rec.origin == "wikidata"
    assert next(i for i in led.items if i.source == "christies.com").kind == "caption"


def test_lens_work_keeps_engine_answers_with_their_provider():
    from neural_canvas.lens_pool import Work
    w = Work(None, lambda im: "")
    w.note("google_lens:serpapi", {"organic_results": [{"title": "The Valley of Mexico", "link": "https://en.wikipedia.org/x"}]})
    assert w.texts == [{"text": "The Valley of Mexico", "url": "https://en.wikipedia.org/x", "site": "",
                        "source": "google_lens:serpapi"}]


def test_a_citation_must_carry_the_value_whatever_the_models_say():
    from neural_canvas.agents import states
    led = Ledger()
    other = led.add("caption", "x", "Painting of Fruit and Ice Cream with Hummingbirds - Suzan Visser", "https://p.com/1")
    named = led.add("caption", "x", "Hummingbird and fish - Suzan Visser (1967 -)", "https://r.com/1")
    cyr = led.add("engine", "yandex", "Сюзан Виссер колибри и рыба")
    assert not states(other, "title", "Hummingbird and Fish") and states(named, "title", "Hummingbird and Fish")
    assert states(other, "artist", "Suzan Visser") and not states(named, "artist", "Sandra Younes")
    page = led.add("caption", "x", "José María Velasco - The Valley of Mexico", "https://w.org/1")
    assert states(page, "artist", "José María Velasco Gómez Obregón")     # two parts of a long name
    assert states(cyr, "artist", "Suzan Visser") is None      # another script: the checker decides
    rec = led.add("record", "wikidata", "Wikidata item Q2", claims={"date": "1919", "medium": "oil on canvas"})
    assert states(rec, "date", "c. 1919") and states(rec, "medium", "Oil on canvas") and not states(rec, "date", "1920")


def test_page_less_answers_are_one_witness_and_a_stray_script_character_bypasses_nothing():
    from neural_canvas.agents import states
    led = Ledger()
    led.add("copy", "yandex", "Still life", "https://a.org/1", verified=True, clip=0.95, inliers=60)
    a = led.add("engine", "bing", "Bing's best guess for this picture: Jan Steen")
    b = led.add("engine", "yandex", "Yandex tag for this picture: Jan Steen")
    v = rule_facts(led, _ruling([a.id, b.id], [a.id, b.id], "Jan Steen", "Still life"), _check([a.id, b.id]))
    assert "artist" not in v["facts"]                       # two page-less answers are one origin
    mixed = led.add("caption", "x", "Hummingbird and fish 鳥 - by somebody", "https://b.org/1")
    assert not states(mixed, "artist", "Suzan Visser")


def test_a_caption_rescored_as_verified_becomes_the_copy():
    led = Ledger()
    c = led.add("caption", "yandex", "Tower of Goubing", "https://example.org/a", clip=0.0)
    c2 = led.add("copy", "yandex", "Tower of Goubing", "https://example.org/a", verified=True, clip=0.93, inliers=60)
    assert c2 is c and c.kind == "copy" and c.verified and len(led.items) == 1


def test_engine_answers_and_captions_support_but_never_anchor():
    # pass 4: a James Ward colour study was ruled "Josef Albers" on engines' look-alike answers
    led = Ledger()
    led.add("copy", "yandex", "colour study", "https://a.org/1", verified=True, clip=0.95, inliers=60)          # E1
    e = led.add("engine", "google_lens:serpapi", "Josef Albers Homage to the Square", "https://moma.org/x")      # E2
    c = led.add("caption", "yandex", "Josef Albers - Homage to the Square", "https://pinterest.com/pin/2", clip=0.9)  # E3
    v = rule_facts(led, _ruling([e.id, c.id], [e.id, c.id], "Josef Albers", "Homage to the Square"), _check([e.id, c.id]))
    assert "artist" not in v["facts"] and "no verified copy or catalogue record" in v["unruled"]["artist"]


def test_a_title_cut_off_by_a_caption_is_never_ruled():
    led = _visser_ledger()
    v = rule_facts(led, _ruling(["E1", "E2", "E3"], ["E1", "E3"], title="Hummingbird and..."), _check(["E1", "E2", "E3"]))
    assert "title" not in v["facts"] and "cut off" in v["unruled"]["title"]


def test_a_caption_beside_another_picture_does_not_speak_for_this_one():
    # bench pass 7: a Pinterest pin titled after Eric Wert, plus an article on Wert's own paintings, ruled a Visser "Eric Wert"
    led = Ledger()
    pin = led.add("copy", "yandex", "Eric Wert: Masterful Dutch-Inspired Still Life Paintings", "https://www.pinterest.com/pin/1",
                  verified=True, clip=0.97, inliers=80)
    far = led.add("caption", "agent_search", "The hyper-realism of Eric Wert's still-life paintings", "https://visualflood.com/x",
                  clip=0.66)
    similar = led.add("caption", "agent_search", "Eric Wert still life", "https://ericwert.com/1", clip=0.84)
    copy = led.add("caption", "google_lens", "Eric Wert, still life", "https://www.artsy.net/artwork/1", clip=0.87)
    v = rule_facts(led, _ruling([pin.id, far.id, similar.id], [pin.id], "Eric Wert", "x"), _check([]))
    assert "artist" not in v["facts"] and "1 independent" in v["unruled"]["artist"]  # similar pictures, not this one
    v2 = rule_facts(led, _ruling([pin.id, copy.id], [pin.id], "Eric Wert", "x"), _check([]))
    assert v2["facts"].get("artist") == "Eric Wert"           # a near-copy's caption does speak for it


def test_the_lens_picker_prefers_providers_that_return_text():
    from neural_canvas.lens_pool import LensPool, Provider
    big = Provider("vision", lambda w: [], allowance=1000, renews="monthly")
    rich = Provider("serpapi", lambda w: [], allowance=250, renews="monthly", texts=True)
    pool = LensPool([big, rich], None, lambda im: "")
    pool.refresh = lambda p, force=False: None
    pool.remaining = lambda p: {"vision": 990, "serpapi": 60}[p.name]     # vision has the larger share
    assert pool.pick().name == "serpapi"
    assert pool.pick({"serpapi"}).name == "vision"                     # the image-only provider is the next witness


def test_uncited_catalogue_records_whose_own_field_states_a_date_support_it():
    # Wyeth, Christina's World (2026-11-10): three records read 1948; the adjudicator cited none of them
    led = Ledger()
    led.add("copy", "bing", "Christina's World, 1948", "https://www.moma.org/a", verified=True, clip=0.95, inliers=90)   # E1
    led.add("record", "wikidata", "wikidata catalogue record: date '1948'", "https://www.wikidata.org/wiki/Q2465911",
            claims={"date": "1948", "medium": "Egg tempera on gesso"}, overrule=True)                                 # E2
    led.add("record", "wikipedia.org", "wikipedia catalogue record: date '1948'", "https://en.wikipedia.org/wiki/C",
            claims={"date": "1948"})                                                                                   # E3
    led.add("record", "wikiart", "wikiart catalogue record", "https://www.wikiart.org/x", claims={"date": "1950"})   # E4
    ruling = {**_ruling(["E1"], ["E1"], "Andrew Wyeth", "Christina's World"), "date": {"value": "1948", "cites": ["E1"]}}
    v = rule_facts(led, ruling, _check(["E1"]), record_facts=("date", "medium", "collection"))
    assert v["facts"]["date"] == "1948" and set(v["support"]["date"]) == {"E1", "E2", "E3"}     # not E4: another year


def test_a_co_author_or_a_title_in_another_language_is_no_contradiction():
    # the Ghent Altarpiece (2026-11-10): Commons credits Hubert van Eyck; Commons and KIK-IRPA title it in German and Dutch
    from types import SimpleNamespace as NS
    from neural_canvas.agents import contradicts
    rec = lambda **c: NS(claims=c, text="")                                                          # noqa: E731
    assert not contradicts(rec(artist="Hubert van Eyck"), "artist", "Jan van Eyck")
    assert contradicts(rec(artist="Hans Memling"), "artist", "Jan van Eyck")
    assert not contradicts(rec(title="Lam Gods: Aanbidding van het Lam"), "title", "Adoration of the Lamb from the Ghent Altarpiece")
    assert not contradicts(rec(title="LUX: Yale Collections Discovery"), "title", "Adoration of the Lamb from the Ghent Altarpiece")
    assert contradicts(rec(title="The Arnolfini Portrait"), "title", "Adoration of the Lamb from the Ghent Altarpiece")
    assert not contradicts(rec(), "title", "Adoration of the Lamb")                                   # states nothing


def test_a_date_a_medium_or_a_collection_is_contradicted_only_by_a_different_one():
    from types import SimpleNamespace as NS
    from neural_canvas.agents import contradicts
    rec = lambda **c: NS(claims=c, text="")                                                          # noqa: E731
    assert not contradicts(rec(date="before 1426-1432"), "date", "1432")
    assert contradicts(rec(date="1890"), "date", "1432")
    assert not contradicts(rec(medium="oil on panel"), "medium", "Oil on oak panel")
    assert contradicts(rec(medium="tempera on panel"), "medium", "Oil on panel")
    assert not contradicts(rec(collection="Kathedrale St. Bavo"), "collection", "St Bavo's Cathedral")


def test_a_fact_other_than_title_and_artist_needs_one_record_and_a_conflict_is_chosen():
    # Sir, 2026-11-10: one reliable witness for the other facts; the holder's own record rules its collection;
    # true but conflicting records: choose and fill
    led = Ledger()
    led.add("copy", "bing", "Cowry Shells", "https://example.org/a", verified=True, clip=0.95, inliers=90)          # E1
    led.add("record", "rijksmuseum", "Rijksmuseum record", "https://www.rijksmuseum.nl/en/collection/RP-T-1",
            claims={"collection": "Rijksmuseum"}, overrule=True)                                                    # E2
    led.add("record", "met", "Met record", "https://www.metmuseum.org/art/collection/1",
            claims={"collection": "The Metropolitan Museum of Art"}, overrule=True)                                # E3
    ruling = {**_ruling(["E1"], ["E1"], "Johann Gustav Hoch", "Cowry Shells"),
              "collection": {"value": "Rijksmuseum", "cites": ["E2"]}}
    chk = {**_check(["E1"]), "collection": {"supports": ["E2"], "contradicts": ["E3"]}}
    v = rule_facts(led, ruling, chk, record_facts=("collection",))
    assert v["facts"]["collection"] == "Rijksmuseum" and v["contradicted"]["collection"] == ["E3"]
    caption_only = {**ruling, "date": {"value": "1750", "cites": ["E1"]}}
    assert "date" not in rule_facts(led, caption_only, chk, record_facts=("date", "collection"))["facts"]   # no record


def test_a_page_naming_the_work_witnesses_its_card_facts_and_a_shop_never_does():
    # ground truth, 2026-11-12: Fink's medium was in Google's answer and on Pinterest, never in a catalogue record
    led = Ledger()
    led.add("copy", "omkar", "FINK, WALDEMAR Abendstimmung in Adelboden. 1913.", "https://www.kollerauktionen.ch/de/1.htm",
            verified=True, clip=0.95, inliers=90)                                                                       # E1
    led.add("copy", "yandex", "Waldemar Fink, Evening in Adelboden", "https://www.instagram.com/p/1/", verified=True,
            clip=0.93, inliers=60)                                                                                       # E2
    led.add("engine", "omkar", "Evening in Adelboden (1913) By Fink Waldemar; oil on board; 80 x 103 cm", "", "Artnet")  # E3
    led.add("caption", "omkar", "Evening in Adelboden by Waldemar Fink, 1913. Oil on board", "https://es.pinterest.com/pin/1/",
            clip=0.5)                                                                                                    # E4
    led.add("caption", "bing", "Evening in Adelboden Waldemar Fink canvas print", "https://www.etsy.com/listing/1", clip=0.9)  # E5
    led.add("caption", "bing", "Fall Collection '25", "https://example.org/fall", clip=0.9)                           # E6
    ruling = {**_ruling(["E1", "E2"], ["E2", "E3"], "Waldemar Fink", "Evening in Adelboden"),
              "medium": {"value": "Oil on board", "cites": ["E3", "E4"]},
              "date": {"value": "1913", "cites": ["E3"]},
              "collection": {"value": "Private collection", "cites": ["E6"]}}
    chk = _check(["E1", "E2", "E3"])
    v = rule_facts(led, ruling, chk, record_facts=("date", "medium", "collection"))
    assert v["facts"]["medium"] == "Oil on board" and set(v["support"]["medium"]) == {"E3", "E4"}
    assert v["facts"]["date"] == "1913" and "E1" in v["support"]["date"]        # the uncited verified lot page joins
    assert "collection" not in v["facts"]               # "Collection" alone names no holder, and E6 names no work
    shop = {**ruling, "medium": {"value": "Canvas", "cites": ["E5"]}}
    assert "medium" not in rule_facts(led, shop, chk, record_facts=("medium",))["facts"]


def test_a_title_proven_at_identification_keeps_its_witnesses_when_ruled_again():
    led = _visser_ledger()
    ruling = _ruling(["E1"], ["E1"])                          # the catalogue's adjudicator cited one site
    assert "title" not in rule_facts(led, ruling, _check(["E1"]))["facts"]
    carry = {"artist": ("Suzan Visser", ["E1", "E3"]), "title": ("Hummingbird and Fish", ["E1", "E3"])}
    v = rule_facts(led, ruling, _check(["E1"]), carry=carry)
    assert v["facts"]["title"] == "Hummingbird and Fish" and set(v["support"]["title"]) == {"E1", "E3"}
    other = _ruling(["E1"], ["E1"], title="Fishers of Bluebirds")    # another value keeps nothing
    assert "title" not in rule_facts(led, other, _check(["E1"]), carry=carry)["facts"]


def test_the_works_own_record_fills_a_card_fact_the_adjudicator_left_blank():
    # Signac, Cassis, Cap Lombard (2026-11-12): Wikidata's record stated the medium and the museum; neither printed
    led = Ledger()
    led.add("copy", "bing", "Paul Signac - Cassis, Cap Lombard, Opus 196", "https://www.reddit.com/r/museum/1", verified=True,
            clip=0.95, inliers=90)                                                                                  # E1
    led.add("copy", "bing", "Cassis, Cap Lombard, Opus 196 by Paul Signac", "https://www.wikiart.org/x", verified=True,
            clip=0.94, inliers=80)                                                                                  # E2
    led.add("record", "wikidata", "wikidata record", "https://www.wikidata.org/wiki/Q24046510", overrule=True,
            claims={"title": "Cassis, Cap Lombard, Opus 196", "medium": "Oil paint on canvas",
                    "collection": "Kunstmuseum Den Haag"})                                                         # E3
    led.add("record", "wikidata", "wikidata record of another work", "https://www.wikidata.org/wiki/Q2", overrule=True,
            claims={"title": "Le Pont des Arts", "date": "1928"})                                                  # E4
    ruling = _ruling(["E1", "E2"], ["E1", "E2"], "Paul Signac", "Cassis, Cap Lombard, Opus 196")
    v = rule_facts(led, ruling, _check(["E1", "E2"]), record_facts=("date", "medium", "collection"))
    assert v["facts"]["medium"] == "Oil paint on canvas" and v["facts"]["collection"] == "Kunstmuseum Den Haag"
    assert "date" not in v["facts"]                   # another work's record never speaks for this one
