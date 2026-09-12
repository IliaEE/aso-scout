"""
Recorded-shape Apple responses for offline testing.

IMPORTANT: these are hand-built to match the *shape* of Apple's responses, not
real data. The numbers are invented. They exist to exercise pipeline logic —
never read them as market intelligence.

Four scenarios are covered deliberately:
  1. noise meter    — weak, stale, monetized SERP  -> should QUEUE
  2. habit tracker  — strong leader                -> should WATCHLIST
  3. scan pressure  — nonsense pairing             -> should fail INCOHERENT
  4. obscure conv   — coherent but unmonetized     -> should fail NO_MONETIZATION
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.sources.http import FixtureClient
from src.sources.itunes import SEARCH_URL
from src.sources.suggest import HINTS_URL


def _ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")


def app(track_id, name, rating, reviews, upd_days, age_days=1500,
        price=0.0, iap=True, langs=("EN",), desc="A useful app."):
    return {
        "trackId": track_id,
        "bundleId": f"com.example.a{track_id}",
        "trackName": name,
        "sellerName": "Example Inc",
        "description": desc,
        "genres": ["Utilities"],
        "primaryGenreName": "Utilities",
        "averageUserRating": rating,
        "userRatingCount": reviews,
        "price": price,
        "advisories": ["In-App Purchases"] if iap else [],
        "currentVersionReleaseDate": _ago(upd_days),
        "releaseDate": _ago(age_days),
        "languageCodesISO2A": list(langs),
        "contentAdvisoryRating": "4+",
    }


# --- 1. noise meter: the good case --------------------------------------
NOISE_METER = [
    app(1, "Decibel X: dB Sound Level Meter", 4.7, 92000, 60,
        desc="Measure noise levels with decibel accuracy."),
    app(2, "Sound Meter: Decibel Detector", 3.9, 1240, 420,
        desc="Simple sound meter for noise measurement."),
    app(3, "dB Meter Pro: Noise Measurement", 4.1, 340, 780,
        desc="Professional noise meter and spl analyzer."),
    app(4, "Noise Detector & Sound Analyzer", 3.6, 95, 930,
        desc="Detect noise around you."),
    app(5, "Sonometer: SPL Audio Level Tool", 4.0, 610, 270,
        desc="Sound level meter with spl readings."),
    app(6, "Noise Meter Simple", 3.4, 48, 1100, desc="Noise meter."),
    app(7, "Quiet: Noise Level Monitor", 4.2, 2100, 150, desc="Monitor noise level."),
    app(8, "dB Sound Meter Free", 3.8, 180, 600, price=0.0, iap=False,
        desc="Free db sound meter."),
    app(9, "Audio Level Meter", 3.5, 72, 1400, desc="Audio level meter tool."),
    app(10, "Noise Watch", 4.0, 310, 500, desc="Noise watch and meter."),
]

# --- 2. habit tracker: strong leader, goes to watchlist ------------------
HABIT_TRACKER = [
    app(21, "Streaks: Habit Tracker", 4.8, 210000, 20,
        desc="The habit tracker that helps you build good habits."),
    app(22, "Habitica: Gamify Your Habits", 4.7, 88000, 35,
        desc="Habit tracker and todo list."),
    app(23, "Way of Life: Habit Tracker", 4.6, 61000, 90,
        desc="Track habits daily."),
    app(24, "Done: A Simple Habit Tracker", 4.7, 54000, 110,
        desc="Simple habit tracker."),
    app(25, "Productive: Habit Tracker", 4.6, 120000, 45,
        desc="Build habits that stick."),
]

# --- 3. scan pressure: nonsense pairing, incoherent SERP -----------------
# Apple returns a grab bag because the query means nothing. Note that none of
# these titles contain both query tokens — that is what collapses coherence.
SCAN_PRESSURE = [
    app(31, "Blood Pressure Companion", 4.3, 5400, 200,
        desc="Log your blood pressure readings."),
    app(32, "Tire Gauge Helper", 3.9, 120, 800, desc="Check tire inflation."),
    app(33, "Weather Barometer", 4.1, 890, 300, desc="Barometric readings."),
    app(34, "Heart Health Log", 4.4, 2300, 150, desc="Track cardiac metrics."),
    app(35, "Medical Records Vault", 4.0, 430, 500, desc="Store health documents."),
]

# --- 4. coherent but nobody monetizes -----------------------------------
OBSCURE_CONVERTER = [
    app(41, "Cubit Converter", 3.2, 18, 1200, price=0.0, iap=False,
        desc="Convert ancient cubit units."),
    app(42, "Cubit Unit Tool", 3.0, 6, 1600, price=0.0, iap=False,
        desc="Cubit converter for historians."),
    app(43, "Ancient Units Converter", 3.5, 24, 900, price=0.0, iap=False,
        desc="Convert cubit and other ancient units."),
]


# --- 5. game query: passes coherence and money, still worthless ----------
# This is what probing "block" actually returned from the live endpoint.
BLOCK_PUZZLE = [
    app(51, "Block Blast!", 4.8, 1_200_000, 10, desc="Block puzzle game."),
    app(52, "Block Puzzle - Sudoku Style", 4.7, 240_000, 25, desc="Classic block puzzle."),
    app(53, "Block Crush: Puzzle Game", 4.6, 88_000, 40, desc="Crush blocks."),
    app(54, "Block Craft 3D", 4.5, 410_000, 60, desc="Build with blocks."),
    app(55, "Wood Block Puzzle", 4.7, 150_000, 30, desc="Wooden block puzzle."),
]
for _a in BLOCK_PUZZLE:
    _a["primaryGenreName"] = "Games"
    _a["genres"] = ["Games", "Puzzle"]


SUGGEST_SCAN = {
    "hints": [
        {"term": "scan"},
        {"term": "scanner app"},
        {"term": "scan documents"},
        {"term": "scan pdf"},
        {"term": "scan qr code"},
        {"term": "scan receipts"},
        {"term": "scan barcode"},
        {"term": "scan to pdf"},
    ]
}

SUGGEST_NOISE = {
    "hints": [
        {"term": "noise meter"},
        {"term": "noise level meter"},
        {"term": "noise detector"},
        {"term": "noise meter app"},
    ]
}


def build_client() -> FixtureClient:
    """Wire fixtures to the URL+params keys the sources will request."""
    f: dict = {}

    def suggest_key(q: str, country="us", lang="en_us"):
        return FixtureClient.make_key(
            HINTS_URL,
            {"q": q, "clientApplication": "Software", "country": country, "l": lang},
        )

    def search_key(term: str, country="us", limit=10):
        return FixtureClient.make_key(
            SEARCH_URL,
            {"term": term, "country": country, "entity": "software", "limit": limit},
        )

    f[suggest_key("scan")] = SUGGEST_SCAN
    f[suggest_key("scan ")] = SUGGEST_SCAN
    f[suggest_key("noise")] = SUGGEST_NOISE
    f[suggest_key("noise ")] = SUGGEST_NOISE

    f[search_key("noise meter")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("noise level meter")] = {"resultCount": 10, "results": NOISE_METER[:9]}
    f[search_key("noise detector")] = {"resultCount": 10, "results": NOISE_METER[1:]}
    f[search_key("noise meter app")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("habit tracker")] = {"resultCount": 5, "results": HABIT_TRACKER}
    f[search_key("scan pressure")] = {"resultCount": 5, "results": SCAN_PRESSURE}
    f[search_key("cubit converter")] = {"resultCount": 3, "results": OBSCURE_CONVERTER}
    f[search_key("scan documents")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan pdf")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan qr code")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scanner app")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan receipts")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan barcode")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan to pdf")] = {"resultCount": 10, "results": NOISE_METER}
    f[search_key("scan")] = {"resultCount": 10, "results": NOISE_METER}

    return FixtureClient(f)
