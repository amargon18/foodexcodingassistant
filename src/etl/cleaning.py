import re
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd


# ── Compiled regexes ──────────────────────────────────────────────────────────

_LANG_PREFIX_RE = re.compile(r"^([a-z]{2}):(.*)", re.I)

_FUNCTIONAL_QUALIFIER_RE = re.compile(
    r"\s+for\s+(colou?r|flavou?r|aroma|texture|preservation|coloring|colouring)\b.*$",
    re.I,
)
_INTERNAL_MEASURE_RE = re.compile(
    r"\b\d+\s*(tbsp|tsp|cups?|tablespoons?|teaspoons?)\b", re.I
)

_BOILERPLATE_RE = re.compile(
    r"\b(manufactured|distributed|refrigerat|best before|calorie|calories|"
    r"remove|set aside|made from|plant that|company|consumer information|"
    r"see base|packaging|per serving|daily value|directions|warning|"
    r"product is|for distribution|contains allergen|allergen|"
    r"servings?|flush|consume|responsibly|celebrat|irradiat|processed with|"
    r"use by|sell by|keep refrigerat|store in|"
    r"may also contain|trace of|nutrition|suitable for|less than|per package|contains no)\b",
    re.I,
)
_UNIT_TOKEN_RE = re.compile(
    r"^\d[\d\.,]*\s*(g|mg|kg|ml|l|oz|lb|kcal|kj|cal|iu)?$",
    re.I,
)

# Instruction/qualifier phrases that are not actual ingredients
_NOISE_PHRASE_RE = re.compile(
    r"^(to\s+(preserve|promote|maintain|retain)|"
    r"cured\s+with|"
    r"contains\s+live|"
    r"contains\s+\d+|"
    r"contains\s+less\s+than|"
    r"and\s+less\s+of|"
    r"or\s+less\s+of)\b"
    r"|^contains$",
    re.I,
)

# Residual boilerplate that passes is_valid_ingredient but is not an ingredient name:
# storage/handling instructions, address fragments, foreign-language packaging text.
_RESIDUAL_BOILERPLATE_RE = re.compile(
    r"\b("
    r"how\s+to|"
    r"store\s+in\s+a|keep\s+in\s+a|"
    r"dark\s+place|dry\s+place|cool\s+place|cool\s+and\s+dry|"
    r"net\s+weight|net\s+wt|drained\s+weight|gross\s+weight|"
    r"chief\s+medical|medical\s+officer|"
    r"sulphites?\s+are|sulfites?\s+are|"
    r"farbstoffe|farbsloffe|farbstoff|"
    r"zawiera|ingredienti|zutaten|composici"
    r")\b",
    re.I,
)

# Food nouns whose co-occurrence (3+ word string, 2+ nouns) signals a multi-ingredient fragment
_FOOD_CORE_NOUNS: frozenset = frozenset({
    "water", "salt", "sugar", "oil", "milk", "flour", "starch",
    "syrup", "vinegar", "yeast", "butter",
})
_MULTI_INGR_ACID_RE = re.compile(r"\bacid\b.*\bacid\b", re.I | re.DOTALL)

_ECODE_RE = re.compile(
    r"^e\s*-?\s*(\d{3,4})\s*([a-z]?\s*(?:\([ivx]+\))?)?$",
    re.I,
)

_SHORT_INGREDIENT_WHITELIST = {
    "oil", "egg", "rye", "oat", "fat", "fig", "yam", "roe", "gin", "rum",
    "soy", "gum", "ice", "pea", "nut",
    "salt", "milk", "corn", "beef", "pork", "lamb", "rice", "malt", "lard",
    "whey", "wine", "beer", "miso", "tofu", "fish", "lime", "mint", "sage",
    "dill", "clam", "crab", "duck", "agar", "bran", "plum", "date", "oats",
    "eggs", "nuts", "beet", "cane", "palm", "peas", "seed", "tuna", "kale",
    "okra", "figs", "nori", "suet", "veal", "goat", "teff", "kelp", "ghee",
    "cola", "lees",
    "sugar", "cream", "water", "wheat", "honey", "yeast", "onion", "lemon",
    "apple", "cocoa", "flour", "spice", "juice", "olive", "thyme", "basil",
    "cacao", "carob", "chili", "clove", "dates", "herbs", "maple", "pasta",
    "spelt", "sumac", "peach", "grape", "melon", "mango", "berry", "prune",
    "agave", "anise", "brine", "caper", "cress", "cumin", "curry", "grain",
    "guava", "kamut", "kombu", "kvass", "leeks", "lupin", "mirin", "natto",
    "okara", "olein", "pecan", "perch", "prawn", "quark", "squid", "stout",
    "syrup", "umami", "vodka", "wafer",
}

# Open Food Facts taxonomy ancestor terms — present for context, not directly matchable.
# Rows with these values get matchable=False but are preserved for analysis.
BROAD_CONTEXT_TERMS: frozenset = frozenset({
    "fruit",
    "vegetable",
    "root vegetable",
    "taproot vegetable",
    "fruit vegetable",
    "berries",
    "tuber",
    "oil and fat",
    "vegetable oil and fat",
    "added sugar",
    "disaccharide",
    "monosaccharide",
    "cereal",
    "cereal flour",
    "protein",
    "plant protein",
    "animal protein",
    "minerals",
    "vitamins",
    "ferment",
    "microbial culture",
    "colour",
    "color",
    "flavouring",
    "flavoring",
    "stabiliser",
    "stabilizer",
    "preservative",
})


# ── Shared normalization ──────────────────────────────────────────────────────

def normalize_lookup_key(text: str) -> str:
    """Canonical key normalization used for taxonomy lookups and context-term checks.

    Ensures that glucose-fructose syrup, Glucose fructose syrup, and
    glucose_fructose_syrup all map to the same key.
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("-", " ").replace("_", " ")
    return text.strip()


def clean_ingredient_token(text: str) -> str:
    text = re.sub(r"\s*\d+[\.,]?\d*\s*%", "", text)
    text = re.sub(r"^\d+\s+", "", text)
    text = _INTERNAL_MEASURE_RE.sub("", text)
    text = _FUNCTIONAL_QUALIFIER_RE.sub("", text)
    text = re.sub(r"(?<=[a-z])\d+$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return text.strip()


# ── Validation and classification ─────────────────────────────────────────────

def is_valid_ingredient(text: str) -> bool:
    if len(text) < 3:
        return False

    # E-codes pass immediately — digit-ratio and length checks would otherwise reject them.
    if _ECODE_RE.match(text.strip().lower()):
        return True

    words = text.split()
    if len(words) > 5:
        return False
    if _BOILERPLATE_RE.search(text):
        return False

    letters = [c for c in text if unicodedata.category(c)[0] == "L"]
    if letters:
        non_latin = sum(1 for c in letters if ord(c) > 0x024F)
        if non_latin / len(letters) > 0.25:
            return False

    numeric_ratio = sum(1 for w in words if _UNIT_TOKEN_RE.match(w)) / len(words)
    if numeric_ratio > 0.5:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.4:
        return False

    # Single short token: require whitelist entry.
    # The old two-word short-token rule (which blocked "vitamin c", "e 330", etc.)
    # has been removed — E-codes and vitamin names are normalized before this check.
    if len(words) == 1 and len(text) <= 5 and text.lower() not in _SHORT_INGREDIENT_WHITELIST:
        return False

    return True


def is_additive_code(text: str) -> bool:
    """True if text is a food additive E-number (after normalize_lookup_key)."""
    return bool(_ECODE_RE.match(normalize_lookup_key(text)))


def is_noise_phrase(text: str) -> bool:
    """True for instruction/qualifier phrases that are not real ingredients."""
    return bool(_NOISE_PHRASE_RE.search(normalize_lookup_key(text)))


def is_context_term(text: str) -> bool:
    """True for broad Open Food Facts taxonomy ancestor terms."""
    return normalize_lookup_key(text) in BROAD_CONTEXT_TERMS


def has_valid_ingredients_tags(tags_str: str) -> bool:
    if pd.isna(tags_str) or str(tags_str).strip() == "":
        return False
    tokens = [
        re.sub(r"^[a-z]{2}:", "", t.strip())
        for t in str(tags_str).split(",")
        if t.strip()
    ]
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _UNIT_TOKEN_RE.match(t))
    return numeric / len(tokens) < 0.4


# ── E-code handling ───────────────────────────────────────────────────────────

def normalize_ecode(text: str) -> str:
    m = _ECODE_RE.match(text.strip())
    if not m:
        return text
    number = m.group(1)
    suffix = re.sub(r"[\s()\-]", "", (m.group(2) or "")).lower()
    return f"e{number}{suffix}"


# ── Taxonomy ──────────────────────────────────────────────────────────────────

def load_ingredient_taxonomy(path: Path) -> dict:
    taxonomy: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("en:"):
                continue
            parts = [p.strip() for p in line[3:].split(",")]
            if not parts or not parts[0]:
                continue
            canonical = parts[0].lower()
            for synonym in parts:
                synonym = synonym.strip()
                if not synonym:
                    continue
                # normalize_lookup_key ensures hyphen/underscore/case variants resolve to the same key
                taxonomy[normalize_lookup_key(synonym)] = canonical
    return taxonomy


def normalize_with_taxonomy(ingredient: str, taxonomy: dict) -> str:
    key = normalize_lookup_key(ingredient)
    if key in taxonomy:
        return taxonomy[key]
    # Try E-code normalization as a secondary lookup
    normalized = normalize_ecode(key)
    if normalized != key:
        return taxonomy.get(normalized, ingredient)
    return ingredient


# ── Tag parsing ───────────────────────────────────────────────────────────────

def parse_ingredients_text_raw(text: str) -> List[str]:
    """Parse a raw ingredients_text label string into cleaned, matchable tokens.

    Applies the same cleaning and validation filters as the ETL pipeline so
    that the live-analysis fallback path produces ingredient sets comparable
    to the precomputed products_ingredients parquet.

    Steps:
      1. Split on commas / semicolons / newlines.
      2. Strip parenthetical sub-ingredient lists (kept in the parquet as
         separate tags; dropping them here avoids duplicated sub-ingredients).
      3. clean_text → normalize_ecode → clean_ingredient_token.
      4. Reject tokens that fail is_valid_ingredient, is_noise_phrase, or
         is_context_term — mirroring the matchable=True filter.
    """
    if pd.isna(text) or not str(text).strip():
        return []

    result: List[str] = []
    for part in re.split(r"[,;\n]+", str(text)):
        token = re.sub(r"\([^)]*\)", "", part)   # drop nested sub-lists
        token = clean_text(token)
        if not token:
            continue
        token = normalize_ecode(token)
        if not _ECODE_RE.match(token.strip().lower()):
            token = clean_ingredient_token(token)
        if not token:
            continue
        if not is_valid_ingredient(token):
            continue
        if is_noise_phrase(token):
            continue
        if is_context_term(token):
            continue
        result.append(token)
    return result


def parse_ingredient_tags(tags_str: str) -> list:
    """Return cleaned ingredient strings from 'en:'-prefixed tags only (legacy interface)."""
    if pd.isna(tags_str) or str(tags_str).strip() == "":
        return []
    ingredients = []
    for raw in str(tags_str).split(","):
        token = raw.strip()
        if not token.startswith("en:"):
            continue
        token = token[3:]
        token = clean_text(token)
        token = normalize_ecode(token)
        # Skip token-level cleaning for E-codes: clean_ingredient_token strips digits
        if not _ECODE_RE.match(token.strip().lower()):
            token = clean_ingredient_token(token)
        if token:
            ingredients.append(token)
    return ingredients


def parse_ingredient_tags_rich(tags_str: str) -> list:
    """Parse ALL language-prefixed tags and return per-tag metadata dicts.

    Unlike the legacy parse_ingredient_tags(), this keeps non-English tags so
    they can be marked matchable=False rather than silently dropped.

    Each dict has keys:
        original_tag, tag_lang, ingredient,
        is_valid, is_additive, is_noise_phrase, is_context_term
    """
    if pd.isna(tags_str) or str(tags_str).strip() == "":
        return []

    records = []
    for raw in str(tags_str).split(","):
        original_tag = raw.strip()
        if not original_tag:
            continue

        m = _LANG_PREFIX_RE.match(original_tag)
        if m:
            tag_lang = m.group(1).lower()
            token = m.group(2)
        else:
            tag_lang = ""
            token = original_tag

        # Normalize in the correct order: clean → E-code normalize → token clean.
        # E-codes skip clean_ingredient_token because it strips trailing digits (e330→e).
        token = clean_text(token)
        token = normalize_ecode(token)
        if not _ECODE_RE.match(token.strip().lower()):
            token = clean_ingredient_token(token)

        if not token:
            continue

        additive = is_additive_code(token)
        noise = is_noise_phrase(token)
        context = is_context_term(token)
        valid = is_valid_ingredient(token)

        records.append({
            "original_tag": original_tag,
            "tag_lang": tag_lang,
            "ingredient": token,
            "is_valid": valid,
            "is_additive": additive,
            "is_noise_phrase": noise,
            "is_context_term": context,
        })

    return records


# ── Row builder ───────────────────────────────────────────────────────────────

def build_ingredient_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per product into one row per ingredient with diagnostic columns.

    Output columns:
        code, lang_product_name, product_name, product_context,
        original_tag, tag_lang, ingredient, nlp_text,
        is_additive, is_noise_phrase, is_context_term, matchable
    """
    df = df.copy()
    df["product_context"] = df["product_name"].apply(clean_text)
    df["_parsed"] = df["ingredients_tags"].apply(parse_ingredient_tags_rich)

    df = df.explode("_parsed").reset_index(drop=True)
    df = df[df["_parsed"].notna() & (df["_parsed"] != "")].reset_index(drop=True)

    parsed_df = pd.json_normalize(df["_parsed"].tolist())
    df = df.drop(columns=["_parsed"]).reset_index(drop=True)
    df = pd.concat([df, parsed_df], axis=1)

    before = len(df)
    df = df[df["is_valid"]].reset_index(drop=True)
    print(f"  Step B: dropped {before - len(df):,} invalid tokens, {len(df):,} remaining")

    # matchable: English-language ingredient that is neither noise nor a context ancestor
    df["matchable"] = (
        (df["tag_lang"] == "en")
        & ~df["is_noise_phrase"]
        & ~df["is_context_term"]
    )

    df["nlp_text"] = df["ingredient"]

    output_cols = [
        "code", "lang_product_name", "product_name", "product_context",
        "original_tag", "tag_lang", "ingredient", "nlp_text",
        "is_additive", "is_noise_phrase", "is_context_term", "matchable",
    ]
    # Keep only columns that actually exist (lang_product_name may be absent in tests)
    return df[[c for c in output_cols if c in df.columns]]
