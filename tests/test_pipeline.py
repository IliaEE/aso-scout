"""
Logic tests. No network.

The important assertions are the negative ones: nonsense pairings must die on
coherence, and dead niches must die on the monetization gate. Those two are
what make verb x object safe without a language model.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import features as feat
from src.pipeline import score as scoring
from src.pipeline.cluster import build_clusters, eligible_head, overlap
from src.pipeline.coherence import coherence_score, dedup_terms, top1_relevance
from src.seeds import expand
from src.seeds.verbs import SEED_VERBS
from src.sources.itunes import normalize_app
from tests import fixtures

PASS, FAIL = "  ok  ", " FAIL "
results: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> None:
    results.append((bool(cond), label))
    print(f"[{PASS if cond else FAIL}] {label}")


def norm(raw_list):
    return [normalize_app(r) for r in raw_list]


print("\n--- coherence: does the nonsense filter work? ---")
noise = norm(fixtures.NOISE_METER)
pressure = norm(fixtures.SCAN_PRESSURE)
obscure = norm(fixtures.OBSCURE_CONVERTER)
habit = norm(fixtures.HABIT_TRACKER)

coh_good = coherence_score("noise meter", noise)
coh_bad = coherence_score("scan pressure", pressure)
check(coh_good > 0.4, f"real niche is coherent: 'noise meter' = {coh_good:.2f}")
check(coh_bad < 0.25, f"nonsense pairing is incoherent: 'scan pressure' = {coh_bad:.2f}")
check(coh_good > coh_bad * 2, "coherence separates the two by a wide margin")
check(coherence_score("anything", []) == 0.0, "empty SERP scores zero, not an error")

rel = top1_relevance("noise meter", noise)
check(rel > 0.5, f"relevant #1 scores high: {rel:.2f}")


print("\n--- gates ---")
f_noise = feat.extract("noise meter", "us", "en", noise, suggest_position=2)
s_noise = scoring.score(f_noise)
check(s_noise.verdict == scoring.Verdict.QUEUED,
      f"weak+stale+monetized SERP queues: {s_noise.verdict.value} score={s_noise.score}")

f_habit = feat.extract("habit tracker", "us", "en", habit, suggest_position=1)
s_habit = scoring.score(f_habit)
check(s_habit.verdict == scoring.Verdict.WATCHLIST,
      f"dominant leader -> watchlist, not bin: {s_habit.verdict.value}")

f_press = feat.extract("scan pressure", "us", "en", pressure, suggest_position=None)
s_press = scoring.score(f_press)
check("incoherent" in s_press.gates_failed,
      f"nonsense fails INCOHERENT: {s_press.gates_failed}")

f_obsc = feat.extract("cubit converter", "us", "en", obscure, suggest_position=3)
s_obsc = scoring.score(f_obsc)
check("no_traction" in s_obsc.gates_failed,
      f"dead niche fails the traction gate: {s_obsc.gates_failed}")
check(f_obsc.top1_reviews < 300,
      f"...because its leader has almost no reviews: {f_obsc.top1_reviews}")


print("\n--- the dead-niche guard: product vs sum ---")
# A dead niche has a gorgeous weakness score. If scoring were additive it
# would outrank a real opportunity. Verify the product collapses it.
check(s_obsc.factors["weakness"] > 0.3,
      f"dead niche looks weak/takeable: weakness={s_obsc.factors['weakness']}")
check(s_obsc.factors["market"] == 0.0,
      f"...but market factor is zero: {s_obsc.factors['market']}")
check(s_obsc.score < s_noise.score,
      f"product collapses it: dead={s_obsc.score} < real={s_noise.score}")
naive_sum = sum(s_obsc.factors.values())
check(naive_sum > 0.5,
      f"a naive SUM would have ranked it high ({naive_sum:.2f}) — why we multiply")


print("\n--- install proxy honesty ---")
check(f_noise.est_top1_installs_monthly is None,
      "no install estimate without two data points (refuses to fabricate)")
f_noise.top1_weekly_review_delta = 310
est = f_noise.est_top1_installs_monthly
check(est is not None and est > 0, f"estimate appears once deltas exist: ~{est}/mo")


print("\n--- clustering by SERP overlap ---")
serps = {
    "noise meter": noise,
    "noise level meter": noise[:9],
    "noise detector": noise[1:],
    "habit tracker": habit,
    "scan pressure": pressure,
}
clusters = build_clusters(serps)
by_head = {c.head: c for c in clusters}

noise_cluster = next((c for c in clusters if "noise" in c.head), None)
check(noise_cluster is not None and noise_cluster.size >= 3,
      f"noise variants merge: {noise_cluster.head} -> {noise_cluster.members}")
check(all("habit" not in m for m in (noise_cluster.members if noise_cluster else [])),
      "unrelated niche does NOT merge into the noise cluster")

ov_same = overlap({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 9})
ov_diff = overlap({1, 2, 3}, {8, 9, 10})
check(ov_same >= 0.6, f"same-niche overlap high: {ov_same:.2f}")
check(ov_diff == 0.0, f"different-niche overlap zero: {ov_diff:.2f}")

check(not eligible_head("fitness"), "broad single word cannot lead a cluster")
check(not eligible_head("best app"), "all-broad phrase cannot lead a cluster")
check(eligible_head("noise meter"), "two-token specific phrase can lead")


print("\n--- dedup ---")
variants = ["noise meter", "noisemeter", "noise metre", "decibel meter"]
canon = dedup_terms(variants)
groups = len(set(canon.values()))
check(groups == 2, f"spelling variants collapse, distinct terms survive: {groups} groups")
check(canon["noisemeter"] == canon["noise meter"], "noisemeter folds into noise meter")
check(canon["decibel meter"] != canon["noise meter"], "decibel meter stays separate")


print("\n--- lexicon self-expansion (no LLM) ---")
sugg = [h["term"] for h in fixtures.SUGGEST_SCAN["hints"]]
objects = expand.extract_objects("scan", sugg)
check("pdf" in objects and "document" in objects,
      f"objects harvested from lineage, not invented: {list(objects)[:6]}")
check("scan" not in objects, "the probe root is excluded from its own objects")

frontier = expand.next_probes(objects, already_probed={"scan"}, min_count=1)
check(len(frontier) > 0, f"harvested objects become next probes: {frontier[:5]}")
check("app" not in frontier, "broad/stopword tokens never become probes")

verbs = expand.discover_verbs(
    ["convert pdf", "convert heic", "convert video", "merge pdf"],
    known_verbs=set(SEED_VERBS) - {"convert"},
    min_distinct=3,
)
check("convert" in verbs, f"new verb discovered by frequency, not grammar: {verbs}")
verbs_thin = expand.discover_verbs(["frobnicate pdf"], set(SEED_VERBS), min_distinct=3)
check("frobnicate" not in verbs_thin, "single sighting is not enough evidence")

check(expand.singularize("documents") == "document", "plural normalization works")


print("\n--- guard: verb x object never fabricates a query ---")
# The point: we never construct "scan pressure" ourselves. It only exists in
# this test because we hand-wrote it. Real probes only yield what Apple returns.
apple_returned = set(sugg)
check("scan pressure" not in apple_returned,
      "Apple never suggests the nonsense pairing — so it never enters the funnel")
check(all(s.startswith("scan") for s in apple_returned),
      "every harvested suggestion extends the probed root")


print("\n--- storefront header wiring (the empty-hints bug) ---")
from src.config import STOREFRONT_IDS, Storefront
from src.sources.suggest import HINTS_URL, _parse_hints
import asyncio as _aio
from src.sources.http import FixtureClient as _FC

_sf = Storefront("us", "en", "United States", 143441)
check(_sf.store_front_header == "143441-1,29",
      f"header value built correctly: {_sf.store_front_header}")
check(STOREFRONT_IDS["de"] == 143443, "German storefront ID present")


class _HeaderSpy(_FC):
    """Captures the headers probe() sends, so the fix cannot silently regress."""

    def __init__(self):
        super().__init__({})
        self.seen_headers = None
        self.seen_params = None

    async def fetch_json(self, url, params, headers=None):
        self.seen_headers = headers
        self.seen_params = params
        return {"hints": [{"term": "scan pdf"}]}


_spy = _HeaderSpy()
_terms = _aio.run(__import__("src.sources.suggest", fromlist=["probe"]).probe(
    _spy, "scan", "us", "en_us", 143441))
check(_spy.seen_headers is not None
      and _spy.seen_headers.get("X-Apple-Store-Front") == "143441-1,29",
      f"probe sends the storefront header: {_spy.seen_headers}")
check(_spy.seen_params.get("clientApplication") == "Software",
      "probe pins clientApplication=Software (music index otherwise)")
check(len(_terms) == 1 and _terms[0][0] == "scan pdf",
      f"suggestions parse through: {_terms}")


print("\n--- game queries (the live-run noise) ---")
games = norm(fixtures.BLOCK_PUZZLE)
f_game = feat.extract("block puzzle", "us", "en", games, suggest_position=1)
s_game = scoring.score(f_game)
check(f_game.games_share >= 0.5,
      f"games share detected: {f_game.games_share:.2f}")
check(f_game.coherence > 0.5,
      f"game query IS coherent — coherence cannot catch it: {f_game.coherence:.2f}")
check(f_game.top1_reviews > 300,
      f"game query HAS traction — the traction gate cannot catch it: "
      f"{f_game.top1_reviews}")
check("game_dominated" in s_game.gates_failed,
      f"dedicated games gate catches it: {s_game.gates_failed}")

print("\n--- root matching (blockchain/blurams leak) ---")
from src.sources.suggest import root_matches
for _r, _t, _exp in [
    ("block", "block puzzle", True),
    ("block", "blockchain", False),
    ("blur", "blur photo", True),
    ("blur", "blurams camera", False),
    ("backup", "backup contacts", True),
    ("backup", "back-up care", False),
]:
    check(root_matches(_r, _t) is _exp,
          f'root "{_r}" vs "{_t}" -> {_exp}')

print("\n--- seed root ordering ---")
from src.seeds.verbs import SEED_VERBS as _SV
check(_SV[:3] == ["scan", "convert", "compress"],
      f"top roots are high-value, not alphabetical: {_SV[:3]}")
check("blur" not in _SV[:10] and "block" not in _SV[:10],
      "the roots that produced games and brands are demoted")


print("\n--- monetization is NOT measurable from the iTunes API ---")
# Live run rejected 39/39 keywords on a monetization gate computed from
# `advisories`, which carries content ratings for music and film — never an
# IAP flag for apps. These assertions pin the correct behaviour.
_free_iap_app = fixtures.app(
    99, "Remove BG: Background Eraser", 4.6, 48_000, 15, price=0.0, iap=True
)
_norm = normalize_app(_free_iap_app)
check(_norm["is_monetized"] is False,
      "a free app is not claimed to be monetized (we cannot know)")
check(_norm["is_paid"] is False, "is_paid stays truthful for a free app")

_paid = normalize_app(fixtures.app(98, "PDF Expert", 4.7, 20_000, 10, price=9.99))
check(_paid["is_paid"] is True and _paid["is_monetized"] is True,
      "a genuinely paid app is detected — price is real data")

check("no_monetization" not in [g.value for g in scoring.Gate],
      "the unmeasurable monetization gate is gone")
check("no_traction" in [g.value for g in scoring.Gate],
      "replaced by a gate built on data we actually have")

# Regression on the exact live failure: real niches must survive.
_real = feat.extract("remove background", "us", "en",
                     [_norm] + noise[1:], suggest_position=1)
_real_s = scoring.score(_real)
check("no_traction" not in _real_s.gates_failed,
      f"a free-with-subscription niche is no longer rejected: {_real_s.gates_failed}")


print("\n--- platform giants (Google Translate was candidate #2) ---")
from src.pipeline.score import is_giant
check(is_giant("Google LLC") and is_giant("Microsoft Corporation"),
      "giants recognised by seller name")
check(not is_giant("Readdle Inc.") and not is_giant("Compress Photos Inc"),
      "ordinary developers are not flagged as giants")

_giant_serp = [dict(a) for a in noise]
_giant_serp[0]["seller"] = "Google LLC"
_giant_serp[0]["title"] = "Google Translate"
_f_giant = feat.extract("translate on screen", "us", "en", _giant_serp, 1)
_s_giant = scoring.score(_f_giant)
check(_f_giant.top1_seller == "Google LLC", "top-1 seller captured in features")
check("giant_dominated" in _s_giant.gates_failed,
      f"giant-owned keyword is gated out: {_s_giant.gates_failed}")

print("\n--- report groups by niche, not by keyword ---")
from src.report.render import Candidate, group_into_clusters

def _cand(term, score, apps):
    return Candidate(term=term, storefront="us", score=score, notes=[],
                     factors={}, apps=apps, suggest_pos=1)

# Three spellings of one niche (shared SERP) plus one unrelated niche.
_cands = [
    _cand("compress image", 47.6, noise),
    _cand("compress photos", 42.8, noise[:9]),
    _cand("compress photos & pictures", 30.4, noise[1:]),
    _cand("habit tracker", 40.0, habit),
]
_clusters = group_into_clusters(_cands)
check(len(_clusters) == 2,
      f"4 keywords collapse into 2 niches: {[c.head.term for c in _clusters]}")
_big = next(c for c in _clusters if "compress" in c.head.term)
check(_big.size == 3, f"the compress variants group together: {_big.size}")
check(_big.head.term == "compress image",
      f"highest-scoring member leads the cluster: {_big.head.term}")
check(all("habit" not in v.term for v in _big.variants),
      "unrelated niche is not absorbed")
check(_clusters[0].score >= _clusters[1].score, "clusters ordered by score")

_single = group_into_clusters([_cand("noise meter", 70.0, noise)])
check(len(_single) == 1 and _single[0].size == 1,
      "a lone candidate still yields one cluster")


print("\n--- clustering: transitivity vs over-merging ---")
from src.pipeline.cluster import build_clusters as _bc

# Under-merge case from the live run: two variants whose pairwise overlap sits
# below threshold but which both link through a third.
_a = noise[:8]            # shares 8 apps with _b via _c
_b = noise[3:]            # direct overlap with _a is partial
_c = noise[1:9]           # bridges them
_cl = _bc({"compress image": _a, "compress photos": _b, "compress pictures": _c})
_merged = [c for c in _cl if c.size > 1]
check(len(_merged) == 1 and _merged[0].size == 3,
      f"single linkage joins A-B-C through the bridge: "
      f"{[(c.head, c.size) for c in _cl]}")

# Over-merge guard: same #1 app but otherwise disjoint SERPs must NOT merge.
_shared_leader = noise[:1] + habit[1:]      # same #1, rest unrelated
_cl2 = _bc({"noise meter": noise, "unrelated query": _shared_leader})
check(all(c.size == 1 for c in _cl2),
      f"a shared #1 alone does not merge unrelated niches: "
      f"{[(c.head, c.members) for c in _cl2]}")

# Broad terms must not act as bridges between distinct niches.
_cl3 = _bc({"noise meter": noise, "app": noise[:5] + habit[:5], "habit tracker": habit})
_giant_cluster = [c for c in _cl3 if c.size >= 3]
check(not _giant_cluster,
      f"broad term does not glue niches together: "
      f"{[(c.head, c.members) for c in _cl3]}")


print("\n--- tracked-app refresh keeps deltas alive ---")
# Root rotation only moves forward, so an app snapshotted today is never seen
# again once its root is spent. Without a refresh step every app would sit at
# one reading and find_alerts (which needs 3 in 30 days) could never fire.
import tempfile as _tf
from pathlib import Path as _P
from src import db as _db
from src.report.render import find_alerts as _alerts

_tmp = _P(_tf.mkdtemp()) / "alerts.db"
_db.init_db(str(_tmp))
with _db.connect(str(_tmp)) as _c:
    _c.execute("INSERT INTO apps VALUES (1,'us','Decibel X','Utilities','{}','2026-09-13')")
    for _d, _rt in [("2026-08-25", 4.7), ("2026-09-01", 4.5),
                    ("2026-09-08", 4.3), ("2026-09-13", 4.1)]:
        _c.execute("INSERT OR REPLACE INTO app_metrics_daily VALUES (1,'us',?,?,90000)",
                   (_d, _rt))
    _fired = _alerts(_c)
    check(len(_fired) == 1 and "Decibel X" in _fired[0],
          f"decaying leader raises an alert: {_fired}")

    _c.execute("INSERT INTO apps VALUES (2,'us','Stable','Utilities','{}','2026-09-13')")
    for _d, _rt in [("2026-08-25", 4.6), ("2026-09-01", 4.6),
                    ("2026-09-08", 4.55), ("2026-09-13", 4.5)]:
        _c.execute("INSERT OR REPLACE INTO app_metrics_daily VALUES (2,'us',?,?,10000)",
                   (_d, _rt))
    check(not [a for a in _alerts(_c) if "Stable" in a],
          "a stable leader stays quiet")

    _c.execute("DELETE FROM app_metrics_daily WHERE track_id = 1 "
               "AND day <> '2026-09-13'")
    check(not [a for a in _alerts(_c) if "Decibel" in a],
          "a single reading cannot produce an alert — hence the refresh step")


print("\n--- brand queries that look like niches ---")
from src.pipeline.features import name_match as _nm
for _q, _t, _brand in [
    ("count the kicks", "Count the Kicks!", True),
    ("scan halal", "Scan Halal", True),
    ("sign now", "SignNow: e-Signature app", True),   # concatenated brand
    ("sign me", "Sign.Me", True),
    ("monitor your weight", "Monitor Your Weight", True),
    ("resize image", "Image Size", False),
    ("compress image", "Compress Photos & Pictures", False),
    ("scan pokemon cards", "Collectr - TCG Collector App", False),
]:
    _v = _nm(_q, _t)
    check((_v >= 0.75) is _brand,
          f'"{_q}" vs "{_t}" -> {_v:.2f} ({"бренд" if _brand else "ниша"})')

print("\n--- star rating ---")
from src.report.rating import rate as _rate
_r5 = _rate(49.3, 4, name_match=0.33)
check(_r5.stars == 5, f"strong score + real cluster = 5 stars: {_r5.bar}")
_rb = _rate(42.9, 1, name_match=1.0)
check(_rb.stars <= 2 and "бренд?" in _rb.flags,
      f"a brand query is pushed down despite a high score: {_rb.bar} {_rb.flags}")
_rs = _rate(42.0, 1, name_match=0.2)
_rc = _rate(42.0, 4, name_match=0.2)
check(_rc.stars > _rs.stars,
      f"same score, bigger cluster rates higher: {_rs.bar} vs {_rc.bar}")
check(_rate(10.0, 1).stars == 1 and _rate(99.0, 5).stars == 5,
      "stars stay inside 1..5")
check(_rate(30.0, 2).reason, "every rating carries a human-readable reason")

print("\n--- storefronts ---")
import os as _os, importlib as _il
import src.config as _cfg
_os.environ["STORES"] = "us,gb,ca,au"
_il.reload(_cfg)
_st = _cfg.storefronts_from_env()
check([s.key for s in _st] == ["us", "gb", "ca", "au"],
      f"STORES selects storefronts: {[s.key for s in _st]}")
check(all(s.store_id > 143000 for s in _st), "each has a numeric storefront ID")
_os.environ["STORES"] = "us,bogus"
check([s.key for s in _cfg.storefronts_from_env()] == ["us"],
      "unknown codes are dropped, not fatal")
_os.environ["STORES"] = "us"
_il.reload(_cfg)


total = len(results)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{'=' * 58}\n{passed}/{total} checks passed\n{'=' * 58}")
sys.exit(0 if passed == total else 1)
