"""Central configuration. Everything tunable lives here, nothing in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """
    Read .env from the project root into os.environ.

    Hand-rolled rather than pulling in python-dotenv: this is twenty lines and
    one less thing to install. Real environment variables always win, so
    Railway's dashboard settings are never overridden by a stray local file.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Storefront:
    """An App Store country + the language its metadata is written in."""
    country: str          # iTunes country code, e.g. "us"
    lang: str             # ISO language of the storefront, e.g. "en"
    label: str
    store_id: int         # numeric storefront ID for X-Apple-Store-Front

    @property
    def key(self) -> str:
        return self.country.lower()

    @property
    def store_front_header(self) -> str:
        """
        Value for the X-Apple-Store-Front header.

        Without this header the hints endpoint returns an empty array: it has
        no store to suggest for, and the country/l query params are ignored.
        The trailing number is a client version marker — 29, 32 and 12 all
        return byte-identical responses, so the choice is arbitrary.
        """
        return f"{self.store_id}-1,29"


# Platform giants. When one of these owns the #1 slot, the keyword is
# effectively theirs regardless of rating: you are not out-ranking Google on
# "translate" with metadata. Matched as a substring of sellerName, lowercased.
GIANT_SELLERS: set[str] = {
    "google", "apple", "microsoft", "meta platforms", "facebook",
    "adobe", "samsung", "amazon", "bytedance", "tencent", "yandex",
}


# Numeric App Store storefront IDs. Needed for the hints endpoint header.
STOREFRONT_IDS: dict[str, int] = {
    "us": 143441, "gb": 143444, "ca": 143455, "au": 143460,
    "de": 143443, "fr": 143442, "it": 143450, "es": 143454,
    "nl": 143452, "se": 143456, "pl": 143478, "tr": 143480,
    "ru": 143469, "ee": 143518, "lv": 143519, "lt": 143520,
    "jp": 143462, "kr": 143466, "cn": 143465, "in": 143467,
    "br": 143503, "mx": 143468,
}


STOREFRONTS: list[Storefront] = [
    Storefront("us", "en", "United States", 143441),
    Storefront("de", "de", "Germany", 143443),
]


@dataclass
class Thresholds:
    """
    Gate and scoring thresholds.

    Start with these, then let calibration move them once Astro facts
    accumulate. Do NOT treat them as ground truth — they are priors.
    """

    # --- Gate: is the top-3 beatable at all? -----------------------------
    # A keyword is dropped when the top-3 is simultaneously well-rated AND
    # heavily reviewed. Either alone is survivable; both together is not.
    # Lowered from 4.6: a live run put Google Translate (4.3*, 83k) forward as
    # candidate #2 because its rating missed the old bar. Rating that high with
    # review counts that large is not a beatable combination.
    unbeatable_rating: float = 4.4
    unbeatable_reviews: int = 50_000

    # --- Gate: has anyone gained traction in this niche? -----------------
    # Replaces the old monetization gate, which was unmeasurable: the iTunes
    # API gives no IAP data for apps (see sources/itunes.py).
    #
    # Traction is the better dead-niche detector anyway. A niche whose leader
    # has 18 reviews is empty because nobody wants it, not because the SERP is
    # weak. Reviews we can actually measure.
    min_top1_reviews: int = 300

    # Review count at which a niche counts as fully proven, for scoring.
    proven_top1_reviews: int = 50_000

    # --- Gate: coherence -------------------------------------------------
    # Share of top-10 apps whose title+subtitle contain the query's tokens.
    # Nonsense queries return a random grab bag and score near zero.
    min_coherence: float = 0.25

    # --- Weakness signals (drive the score, not gates) -------------------
    stale_days: int = 365          # "not updated in over a year"
    weak_rating: float = 4.2
    thin_reviews: int = 2_000

    # --- Dedup -----------------------------------------------------------
    # rapidfuzz threshold for merging spelling variants. 88 sits in a wide
    # safe gap: "noisemeter"/"noisemetre" = 90 (must merge), while the nearest
    # false positive "mergepdf"/"mergeppt" = 75 (must not). Raising it to 92
    # leaves the metre/meter variant uncollapsed.
    fuzzy_dedup_ratio: int = 88

    # --- Clustering ------------------------------------------------------
    # Share of shared apps in the top-10 for two keywords to be one cluster.
    cluster_overlap: float = 0.60
    min_tokens_for_head: int = 2   # single-word terms never lead a cluster

    # Sharing the #1 app is strong evidence of one intent, but not enough on
    # its own: a popular utility can top two unrelated queries. Require some
    # real SERP similarity alongside it, or single linkage will chain
    # unrelated niches together — and an over-merged cluster hides a good
    # niche inside another's shadow, which is worse than an extra report row.
    same_leader_min_overlap: float = 0.30

    # --- Review-delta install proxy --------------------------------------
    # installs ~= weekly_review_delta * multiplier. Wildly imprecise; only
    # ever compare apps within the same category.
    review_to_install_multiplier: int = 45


def _default_sqlite_path() -> str:
    """
    Where the database lives.

    Explicit SQLITE_PATH wins. Otherwise, if Railway has mounted a volume it
    sets RAILWAY_VOLUME_MOUNT_PATH automatically, and we put the database
    there. This matters because the failure it prevents is silent: without a
    volume the database lands on the ephemeral filesystem, gets wiped on every
    deploy, and nothing in the logs says so — you would just keep seeing
    "истории накоплено 0 нед." forever.
    """
    explicit = os.environ.get("SQLITE_PATH")
    if explicit:
        return explicit
    mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if mount:
        return os.path.join(mount, "aso_scout.db")
    return "aso_scout.db"


def on_railway() -> bool:
    return bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME")
                or os.environ.get("RAILWAY_SERVICE_ID")
                or os.environ.get("RAILWAY_PROJECT_ID"))


def volume_mounted() -> bool:
    return bool(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH"))


@dataclass
class Settings:
    db_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    sqlite_path: str = field(default_factory=_default_sqlite_path)

    # Apple rate limiting. ~20 req/min per IP is the observed ceiling on the
    # iTunes endpoints. Staying under it is cheaper than getting the IP burned.
    requests_per_minute: int = field(default_factory=lambda: _env_int("RPM", 18))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 4))
    request_timeout: float = field(default_factory=lambda: _env_float("TIMEOUT", 20.0))
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "USER_AGENT",
            "aso-scout/0.1 (internal keyword research; contact: you@example.com)",
        )
    )

    # How many keywords to keep in the review queue, and how many make the
    # human-facing report. The report is deliberately tiny.
    queue_size: int = field(default_factory=lambda: _env_int("QUEUE_SIZE", 50))
    report_size: int = field(default_factory=lambda: _env_int("REPORT_SIZE", 3))

    # Astro tracking slots available. Drives the cap in the cluster prompt.
    astro_slots: int = field(default_factory=lambda: _env_int("ASTRO_SLOTS", 40))

    thresholds: Thresholds = field(default_factory=Thresholds)
    storefronts: list[Storefront] = field(default_factory=lambda: list(STOREFRONTS))

    @property
    def use_postgres(self) -> bool:
        return bool(self.db_url)


settings = Settings()
