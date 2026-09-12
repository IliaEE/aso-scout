"""
Seed verbs for autocomplete probing.

These are NOT combined with objects to form queries. They are sent to Apple's
autocomplete endpoint as prefixes; Apple returns the objects that real users
actually search for. That is the whole point: we never invent a pairing like
"scan blood pressure", because Apple would never return it.

Hand-written on purpose. WordNet/nltk synonyms ("peruse", "scrutinise") are
literary register and nobody types them into the App Store.

The list grows by itself: objects harvested from suggestions become probes,
their suggestions yield new verbs, and those land in `discovered_verbs` in the
DB. This file is only the cold start.
"""

# Utility / file manipulation — highest density of ASO-driven apps
UTILITY = [
    "scan", "convert", "compress", "merge", "split", "resize", "crop",
    "rotate", "unzip", "extract", "export", "import", "backup", "restore",
    "sign", "fill", "redact", "watermark", "rename", "sync", "transfer",
]

# Measure / count / track — high subscription viability
QUANTIFY = [
    "track", "count", "measure", "calculate", "monitor", "log", "record",
    "detect", "estimate", "compare", "check", "test",
]

# Media editing
MEDIA = [
    "remove", "blur", "enhance", "upscale", "colorize", "restore",
    "trim", "mute", "reverse", "loop", "overlay", "caption", "transcribe",
]

# Behaviour / restriction
CONTROL = [
    "block", "hide", "lock", "limit", "schedule", "remind", "plan",
    "organize", "clean", "unfollow",
]

# Language
LANGUAGE = [
    "translate", "pronounce", "learn", "practice", "read",
]

# Order matters: `--limit-roots N` takes the first N, so the list is ranked by
# expected ASO value, NOT alphabetically. Sorting this alphabetically puts
# "backup", "block" and "blur" first, which yields casual games (block blast,
# block puzzle) and hardware brands (blurams) instead of utility niches.
# High-intent, clearly transactional verbs come first.
_PRIORITY = [
    "scan", "convert", "compress", "merge", "translate", "remove",
    "track", "count", "measure", "split", "record", "transcribe",
    "resize", "crop", "sign", "fill", "calculate", "detect",
    "monitor", "block", "backup", "blur",
]


def _ranked(verbs: list[str]) -> list[str]:
    """Priority verbs first, in order; everything else after, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for v in _PRIORITY + verbs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


SEED_VERBS: list[str] = _ranked(UTILITY + QUANTIFY + MEDIA + CONTROL + LANGUAGE)

# Probe suffixes appended to each verb. Apple's autocomplete is prefix-based,
# so "scan " gets the top completions and "scan a".."scan z" reaches deeper
# into the tail. The space-only probe is cheapest and most valuable.
PROBE_SUFFIXES: list[str] = [""] + [" "] + [f" {c}" for c in "abcdefghijklmnopqrstuvwxyz"]

# Single-word generic terms that must never become cluster heads. SERP
# intersection is meaningless for them: the top-10 for "fitness" is a grab bag
# and would merge with anything. See pipeline/cluster.py.
TOO_BROAD: set[str] = {
    "app", "apps", "free", "best", "pro", "new", "top", "online", "my",
    "fitness", "health", "music", "photo", "photos", "video", "videos",
    "game", "games", "kids", "money", "food", "work", "life", "home",
    "ai", "chat", "camera", "editor", "player", "maker", "manager",
    "tracker", "scanner", "calculator", "converter", "meter", "test",
}

# Tokens that signal a suggestion is a brand query, not a functional one.
# Used to route keywords to ASA (brand bidding) rather than ASO.
KNOWN_BRAND_HINTS: set[str] = {
    "instagram", "tiktok", "whatsapp", "youtube", "spotify", "netflix",
    "capcut", "canva", "notion", "duolingo", "strava", "figma", "adobe",
    "google", "microsoft", "facebook", "snapchat", "telegram", "discord",
    "roblox", "minecraft", "uber", "airbnb", "revolut", "paypal",
}
