# Identification and attribution plan (November 2026)

Synthesis of an eight-agent research swarm run on 2026-11-05 at Sir's order, plus what was built the
same day. Raw agent reports live in `.claude/scratch/swarm/` (local only, gitignored):
`A-museum-apis.md`, `B-yale-iiif-nrm.md`, `C-aggregators.md`, `D-cooperhewitt-docent.md`,
`E-methods-audit.md`, `F-museum-repos.md`, `G-catalogues-iiif.md`, `H-sample-dossiers.md`, and
`URL-COVERAGE.md` (every URL Sir supplied, mapped to the agent that covered it).

## 1. The rule this plan serves

- **Facts:** title, artist, date, medium and collection are printed only when RULED.
- **Ruled:** two or more credible sources, from different institutions, agree in general. Variant
  wording, translations, alternate titles and overlapping dates all count as agreement.
- **Disagreement allowed:** other credible sources may disagree. An AI review step decides what the
  card shows, and may show a credible variant ("also known as").
- **Leads only:** file names, social media, blogs, shops, print sellers and auction aggregators steer
  the search. They never rule.
- **Filename-blind:** the pipeline never reads the input's file name. Intake copies the pixels to
  `var/work/<run>/source.<ext>` and strips embedded tags.
- **No silent degradation:** a work that cannot be ruled stays blocked and says what it consulted.
  It is never printed on a guess.

## 2. What the research found

### 2.1 Why the four titles were wrong (agent H)

In three of the four, the right answer was already in the pipeline's own evidence. A
higher-scoring but non-credible page outranked it.

| Work | Printed | Credible consensus | What went wrong |
|---|---|---|---|
| Klee | "Red Ideas" | *Giftbeeren* (Poisonous Berries), 1920 | A Pinterest board name won on image score. The inscription on the work reads "Giftbeeren". |
| Wyeth | "Wind from the Sea" | *Airborne*, 1996 (Crystal Bridges) | An Artchive page title for a different painting (NGA, 1947) won. "Airborne" was in the same evidence list. |
| Whistler | "Venice" | *The Riva – Sunset; red and gold*, 1879–80, pastel, Yale 1966.9.27 (Glasgow cat. M.0807) | A blog caption won. The matched copy's file name read `venice-yale.jpg`. |
| Rockwell | "THE LITTLE MODEL 1919" | *Wet Paint* (Girl Running with Wet Canvas), Post cover, 1930 | One MutualArt page, which serves no content to a plain fetch. |

The blocked three:

| Work | Consensus | Where |
|---|---|---|
| Teichert | *The Promised Land*, c. 1949–51, oil on masonite | BYU Museum of Art |
| Remington | *Warrior's Last Ride*, 1908 (oil burned by the artist; survives as a lithograph) | Buffalo Bill Center of the West, Remington catalogue raisonné |
| Signac | *Les Andelys, Côte d'Aval*, 1886 | Art Institute of Chicago (blocked only on colour, since fixed) |

The catalogue stage built on 2026-11-05 already stops leads from ruling. H's evidence predates it.
What remains is reach: finding two credible sources for works that have them.

### 2.2 Sources: live, dead, and decided

| Source | Status (verified 2026-11-05) | Use |
|---|---|---|
| Art Institute of Chicago | API live; fielded search precise; images at 843 px; 60 req/min | **Voting** (built) |
| Smithsonian (EDAN) | Key works; unit filter only inside `q`; Freer\|Sackler is `NMAA` | **Voting** (built) |
| Yale LUX | Linked Art, keyless; Yale metadata CC0 (LUX Open Access page) | **Voting** (built) |
| Rijksmuseum | Old API 410; new keyless Linked Art service live | **Voting** (built) |
| Harvard | Key works; `person` + `title` together is a fielded AND | **Voting** (improved) |
| Met | v1.1 search noisy; one field flag per call; in-copyright works have no image | **Voting** (artist ∩ title built) |
| Cleveland, Europeana, RKD, Wikidata, Wikipedia, Commons, meisterdrucke | Live | Voting (existing) |
| Cooper Hewitt | Reached through EDAN (`CHNDM`); a design museum with no paintings | No separate adapter |
| National Gallery of Art | `www.nga.gov` and its manifests are Cloudflare-walled; `api.nga.gov` images are open; open data needed subset ≈ 241 MB | **Local index** (Phase 6) |
| Glasgow Whistler catalogue raisonné | Plain HTTP, stable `?mid=m0XXX&xml=bib` records | **Adapter** (Phase 2) |
| Remington catalogue raisonné (Center of the West) | Plain HTTP, `/artworks/view/<id>` | **Adapter** (Phase 2) |
| Norman Rockwell Museum | Collection site bot-walled; `prints.nrm.org` / `store.nrm.org` load | Store pages as curated pages (Phase 2) |
| Crystal Bridges | Main domain loads; the `.emuseum.com` mirror is walled | Curated pages (exists) |
| Arthur.io | Pages and sub-sitemaps Cloudflare-challenged; image server open; robots.txt allows `*` for search and reference (blocks AI training and named AI crawlers) | **Lead** via site-restricted image search (built) |
| Artvee | Search and artist pages load (plain `neural-canvas/1.0` agent); robots.txt allows `*` | **Lead**: search + whole-artist catalogue by embedding, parsed with artvee-scraper (built) |
| artera.ae | Re-hosts Artvee records | Lead only (classified commercial, built) |
| api-evangelist profiles | README signposts, not APIs | Read once; never a source |
| Getty ULAN | Server-side TLS failure | Unavailable (remediable) |

### 2.3 Speed (agent E, corrected by measurement)

- **Wrong order:** the worst run (Degas, 69.7 min identify) spent 89% of its time on three cold
  Wikidata oeuvre sweeps of wrong artists. Web detection, tried last, answered in about a second.
- **Not latency:** Wikimedia thumbnails cost 0.12 s each one at a time, but 0.54 s each with six in
  flight (HTTP 429, Retry-After 11 s). The historical 1.4 s per thumbnail was rate-limit waiting.
- **VLM is cheap:** 1.8% of the worst run.

### 2.4 Methods (agents D and E)

- **Two stages:** cheap candidate generation, then strict verification. The two-instrument image
  gate (CLIP + ORB/RANSAC) stays exactly as strict.
- **Join keys:** Wikidata museum properties give exact routes from the work's item to a museum record:
  P4610 AIC, P3634 Met, P350 RKD, P347 Joconde, P4659 Orsay, P1212 Louvre, P4704 SAAM. P10626 is
  not AIC; it is a German ceiling-painting database.
- **Attribution qualifiers:** "attributed to", "workshop of", "circle of", "follower of", "after"
  (CDWA; Linked Art AttributeAssignment). A qualifier is never flattened into a bare name.
- **State what was checked:** every ruling carries the sources consulted, not only the ones that agreed.
- **Families of near-identical works:** go to the catalogue raisonné first. It lists every
  near-duplicate side by side with its accession numbers.

## 3. Built on 2026-11-05

| Commit | Change |
|---|---|
| `cb20936` | Smithsonian and Yale adapters; fielded AIC search; medium gate on title agreement; per-host pacing |
| `1e8b76a` | Rijksmuseum Linked Art adapter; artera.ae classed commercial |
| `3404f7e` | Web detection before a cold oeuvre sweep; no post-match sweep; host-wide Retry-After; one-in-flight hosts |
| `5222bbb` | Freer\|Sackler unit fixed (`NMAA`); fielded Harvard; Met artist ∩ title |

## 4. Build phases

Each phase ends with its tests passing and a commit. Phases 1 to 5 are code only. Phase 6 downloads
data into `~/Claude/Staging/` under the bulk-download policy.

### Phase 1. Leads that point at institutions
- **Institution tokens:** when a matched copy's file name or caption carries an institution token
  ("yale", a museum acronym, an accession-number shape), search that institution first.
- **Inscription first:** the signature pass's text, fuzzy-cleaned, becomes the first literal query.
  Foreign titles are searched as written; translation is only for the card.
- **Live content only:** a page that answers with a challenge, a script shell or an error is no
  source, however it was captioned. The catalogue stage already enforces this; identification's
  leads get the same check before they steer.

### Phase 2. Catalogues raisonnés and institution pages
- **Glasgow Whistler** (`whistlerpaintings.gla.ac.uk`): search, then parse the bib record (title,
  date, medium, support, collection, accession number, inscription). Image-verify as usual.
- **Remington** (`remington.centerofthewest.org`): same pattern.
- **Registry:** a small table maps artist to catalogue-raisonné domain and query template, so adding
  the next one is a data change.
- **Institution store pages:** `prints.nrm.org`, BYU Museum of Art press and store pages, through the
  existing curated-page reader, with `site:` queries.

### Phase 3. Join keys and second artist sources
- **Wikidata walk:** follow every museum property on the work's item through its formatter URL to the
  museum record (generalises the existing linked-pages walk).
- **AIC artist records:** `/agents/{id}` gives a second source for life dates and ULAN/Wikidata IDs.
- **EDAN record fetch:** `/content/edanmdm:<id>` for a known Smithsonian record.

### Phase 4. Attribution qualifier and a narrower medium gate
- **Qualifier field:** `Record.attribution` takes named / attributed / workshop / circle / follower /
  after / anonymous, read from each source's own wording. The card prints it ("Attributed to X").
- **Medium gate:** two records whose media clash never agree on a title (built). When only one states
  a medium, a near-title (not an exact or contained one) needs the date or collection to agree too.
  Wikipedia and Commons rarely state a medium, so "one side silent" cannot simply block.

### Phase 5. The AI review step (adjudicator)
- **Trigger:** only when credible, image-verified records disagree in a way the matching rules cannot
  settle.
- **Model:** the local model, schema-constrained, using agent D's prompt.
- **Holding institution first:** prefer the owning museum's own record over an inventory, finding aid
  or aggregator.
- **Outputs, one of three:** ruled (a display value plus "also known as" variants), shown both (each
  with its source), or unruled. It never invents a value. It never counts as a vote.
- **Recorded:** the rationale and the consulted list go into the run record.
- **Test case:** Van Gogh's wheatfield. Wikidata gives "Wheatfield with sheaves and reaper"; Commons
  gives "Wheat Stacks with Reaper".

### Phase 6. Local indexes (Staging, serialized)
- **NGA:** `objects`, `constituents` and `published_images` CSVs (≈ 241 MB) to
  `~/Claude/Staging/incoming/`. Verify, move to `verified/`, then build a SQLite lookup (artist +
  title to object ID and IIIF image). Images come from `api.nga.gov`, which is not walled. Refresh
  monthly.
- **AIC:** `allArtworks.jsonl` (23 MB) as a pre-filter for candidate titles.

### Phase 7. Serial publications (illustrators)
- **Post covers:** for Rockwell and similar, match against a cover index (date, title, props)
  before trusting a market page. This is the one new capability H found no shortcut for.

### Phase 8. Retest, then reprocess
- **The seven sample works:** each run fresh through the real pipeline (the catalogue stage has never
  yet run inside it).
- **Dry run v3:** all 61 identified works, compared with v2 (10 of 61 ruled).
- **Then Sir's standing order:** clear the album, move `inbox/ingested` back to `inbox`, reprocess,
  monitor, review.

### Future. The AI Art Docent
- **One schema:** the normalized `Record`, its citation and its ruling serve the placard, an
  interactive docent chat and the adjudicator alike.
- **MCP tools** (agent D): `identify_candidates`, `catalogue_lookup`, `rule_card`, `cite`. Every
  response says what was consulted and what agreed.

## 5. Sir's decisions (2026-11-05)

1. **Inscriptions are credible:** a title written on the work is one credible vote, unless it is the
   artist's name, initials, a date or a number (built: `inscription_is_title`).
2. **Auction houses and press are leads and support:** a verified page is listed as corroboration
   beside the ruling and never counts toward the two (built: `sources.supporting`). Auction
   aggregators stay leads only.
3. **Artvee and Arthur.io are pipeline tools:** built as identification tiers (`artsites.py`). Artvee
   is read with artvee-scraper's parsers. Arthur.io's index and artist pages are Cloudflare-challenged,
   which the pipeline does not work around; its pictures are reached through search engines that
   indexed them, and its image server.
