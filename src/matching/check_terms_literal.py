"""
Check how key ingredient strings fare against the FoodEx2 term table:
  1. Literal exact match (case-insensitive)
  2. RapidFuzz WRatio top-5 against all termExtendedName values
  3. RapidFuzz WRatio top-5 against the expanded comma-order + spelling variants
     (the same variant set used by the flat_rf index)

Run:
    python3 src/matching/check_terms_literal.py
"""

import sys
from pathlib import Path

import duckdb
from rapidfuzz import fuzz, process as rf_process

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratified_matcher import comma_order_variants  # noqa: E402

DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"

TERMS_TO_CHECK = [
    "salt",
    "sugar",
    "apple juice",
    "burrata cheese",
    "vegetable oil",
    "potato starch",
    "rice milk",
    "oat milk",
    "semi skimmed milk",
    "corn syrup",
    "glucose syrup",
    "wheat flour",
    "olive oil",
    "tomato paste",
    "yogurt",
    "yoghurt",
]

RF_LIMIT = 5
RF_CUTOFF = 50  # show matches above this score


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    rows = con.execute(
        """
        SELECT termCode, termExtendedName, reportHierarchyCode, reportHierarchyLevel
        FROM foodex2_terms
        WHERE reportHierarchyCode LIKE 'Z0001%'
        ORDER BY reportHierarchyLevel
        """
    ).fetchall()
    con.close()

    # Raw name index: (termExtendedName, termCode, level)
    raw_names = [r[1] for r in rows]
    raw_meta  = [(r[0], r[1], r[2], r[3]) for r in rows]  # (code, name, hier, level)

    # Variant index: expand every term with comma_order_variants
    var_strings: list[str] = []
    var_meta:    list[tuple] = []
    for code, name, hier, level in [(r[0], r[1], r[2], r[3]) for r in rows]:
        for v in comma_order_variants(name):
            var_strings.append(v)
            var_meta.append((code, name, hier, level))

    print(f"Loaded {len(rows):,} food terms  |  {len(var_strings):,} variants\n")

    sep = "=" * 100

    for term in TERMS_TO_CHECK:
        print(sep)
        print(f"QUERY: \"{term}\"")

        # ── 1. Exact literal match ────────────────────────────────────────────
        exact_hits = [
            (code, name, hier, level)
            for code, name, hier, level in raw_meta
            if name.lower() == term.lower()
        ]
        if exact_hits:
            print(f"  EXACT  YES  →  " + "  |  ".join(
                f"{name} ({code}, L{level}, {hier})" for code, name, hier, level in exact_hits
            ))
        else:
            print("  EXACT  NO")

        # ── 2. RapidFuzz on raw termExtendedName ─────────────────────────────
        rf_raw = rf_process.extract(
            term, raw_names,
            scorer=fuzz.WRatio,
            limit=RF_LIMIT,
            score_cutoff=RF_CUTOFF,
        )
        print(f"  RF (raw names, top {RF_LIMIT}, cutoff={RF_CUTOFF}):")
        if rf_raw:
            for match_text, score, idx in rf_raw:
                code, name, hier, level = raw_meta[idx]
                print(f"    {score:5.1f}  L{level}  {name}  ({code}, {hier})")
        else:
            print("    (no matches above cutoff)")

        # ── 3. RapidFuzz on expanded variants ────────────────────────────────
        rf_var = rf_process.extract(
            term, var_strings,
            scorer=fuzz.WRatio,
            limit=RF_LIMIT * 3,     # pull more because variants may repeat the term
            score_cutoff=RF_CUTOFF,
        )
        # Deduplicate by termCode, keep best score per term
        seen: dict[str, tuple] = {}
        for match_text, score, idx in rf_var:
            code, name, hier, level = var_meta[idx]
            if code not in seen or score > seen[code][0]:
                seen[code] = (score, match_text, name, hier, level)
        deduped = sorted(seen.values(), key=lambda x: x[0], reverse=True)[:RF_LIMIT]

        print(f"  RF (variants, top {RF_LIMIT}, cutoff={RF_CUTOFF}):")
        if deduped:
            for score, matched_variant, name, hier, level in deduped:
                print(f"    {score:5.1f}  L{level}  {name}  ({hier})  via: \"{matched_variant}\"")
        else:
            print("    (no matches above cutoff)")

    print(sep)


if __name__ == "__main__":
    main()
