"""
Memory-efficient BGE experiment for FoodEx2 ingredient matching.

Goal:
  - English ingredient tags only
  - additives excluded
  - context/noise excluded through products_ingredients.matchable
  - match against FoodEx2 termExtendedName only

Why this version is lighter than run_matching_bge.py:
  - uses BAAI/bge-small-en-v1.5 by default instead of bge-base
  - encodes ingredients in chunks instead of all unique ingredients at once
  - does not keep all ingredient embeddings or all results in memory
  - writes each chunk directly into DuckDB
  - exports parquet outputs from DuckDB at the end

Run:
  python -m src.matching.run_matching_bge_small_chunked

Optional environment overrides:
  MATCHING_MODEL_NAME=BAAI/bge-base-en-v1.5 INGREDIENT_CHUNK_SIZE=1000 ENCODE_BATCH_SIZE=32 python -m src.matching.run_matching_bge_small_chunked
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Make src/matching importable regardless of working directory.
_SRC_MATCHING = Path(__file__).resolve().parent
if str(_SRC_MATCHING) not in sys.path:
    sys.path.insert(0, str(_SRC_MATCHING))

from stratified_matcher import (
    StratifiedMatcherOptimized,
    build_flat_rapidfuzz_index,
    build_hierarchy_index,
    generate_hierarchy_embeddings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"
OUTPUT_DIR  = PROJECT_ROOT / "data"
REPORT_DIR  = PROJECT_ROOT / "src" / "test" / "output"

# bge-small is the best first experiment when the previous bge-base run was killed.
# It is much lighter while keeping the same retrieval-oriented BGE training family.
MODEL_NAME = os.getenv("MATCHING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
MODEL_SLUG = (
    MODEL_NAME.lower()
    .replace("/", "__")
    .replace("-", "_")
    .replace(".", "_")
)


OUTPUT_TABLE_INGREDIENTS  = f"matching_results_ingredients_{MODEL_SLUG}".lower()
OUTPUT_TABLE_PRODUCTS     = f"matching_results_{MODEL_SLUG}".lower()
OUTPUT_INGREDIENT_PARQUET = OUTPUT_DIR  / f"matching_results_ingredients_{MODEL_SLUG}.parquet"
OUTPUT_PRODUCTS_PARQUET   = OUTPUT_DIR  / f"matching_results_{MODEL_SLUG}.parquet"
OUTPUT_REPORT_TXT         = REPORT_DIR  / "bgr_matching_results"

# Scores are not directly comparable across embedding models. Start with permissive
# thresholds and recalibrate after inspection.
THRESHOLD_LEVEL2 = float(os.getenv("THRESHOLD_LEVEL2", "0.55"))
THRESHOLD_LEVEL3 = float(os.getenv("THRESHOLD_LEVEL3", "0.70"))
THRESHOLD_SPECIFIC = float(os.getenv("THRESHOLD_SPECIFIC", "0.85"))

# Keep these conservative to avoid RAM spikes.
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))
INGREDIENT_CHUNK_SIZE = int(os.getenv("INGREDIENT_CHUNK_SIZE", "2000"))

# Reduce beam sizes for the first lightweight experiment. You can increase these
# after confirming the script runs comfortably.
TOP_K_L2 = int(os.getenv("TOP_K_L2", "3"))
TOP_K_L3 = int(os.getenv("TOP_K_L3", "4"))
TOP_K_SPECIFIC = int(os.getenv("TOP_K_SPECIFIC", "6"))

MATCHABLE_WHERE = """
    ingredient IS NOT NULL
    AND matchable = TRUE
    AND tag_lang = 'en'
    AND COALESCE(is_additive, FALSE) = FALSE
"""

RESULT_COLUMNS_SQL = """
    ingredient VARCHAR,
    level2_code VARCHAR,
    level2_name VARCHAR,
    level2_score DOUBLE,
    level3_code VARCHAR,
    level3_name VARCHAR,
    level3_score DOUBLE,
    specific_code VARCHAR,
    specific_name VARCHAR,
    specific_score DOUBLE,
    final_level INTEGER,
    confidence VARCHAR,
    path_score DOUBLE,
    top1_code VARCHAR,
    top1_name VARCHAR,
    top1_score DOUBLE,
    top2_code VARCHAR,
    top2_name VARCHAR,
    top2_score DOUBLE,
    top3_code VARCHAR,
    top3_name VARCHAR,
    top3_score DOUBLE,
    model_name VARCHAR,
    matching_mode VARCHAR,
    rf_used BOOLEAN,
    rf_score DOUBLE,
    rf_matched_variant VARCHAR,
    bge_score DOUBLE,
    lexical_score DOUBLE,
    branch_penalty DOUBLE,
    exact_variant_match BOOLEAN
"""


def batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def result_records(results) -> list[dict]:
    return [
        {
            "ingredient": r.ingredient,
            "level2_code": r.level2_code,
            "level2_name": r.level2_name,
            "level2_score": r.level2_score,
            "level3_code": r.level3_code,
            "level3_name": r.level3_name,
            "level3_score": r.level3_score,
            "specific_code": r.specific_code,
            "specific_name": r.specific_name,
            "specific_score": r.specific_score,
            "final_level": r.final_level,
            "confidence": r.confidence,
            "path_score": r.path_score,
            "top1_code": r.top1_code,
            "top1_name": r.top1_name,
            "top1_score": r.top1_score,
            "top2_code": r.top2_code,
            "top2_name": r.top2_name,
            "top2_score": r.top2_score,
            "top3_code": r.top3_code,
            "top3_name": r.top3_name,
            "top3_score": r.top3_score,
            "model_name": MODEL_NAME,
            "matching_mode": r.matching_mode,
            "rf_used": r.rf_used,
            "rf_score": r.rf_score,
            "rf_matched_variant": r.rf_matched_variant,
            "bge_score": r.bge_score,
            "lexical_score": r.lexical_score,
            "branch_penalty": r.branch_penalty,
            "exact_variant_match": r.exact_variant_match,
        }
        for r in results
    ]

def write_matching_report(con: duckdb.DuckDBPyConnection) -> None:
    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("=" * 100)
            print("MATCHING EXPERIMENT REPORT")
            print("=" * 100)

            print(f"\nMODEL_NAME: {MODEL_NAME}")
            print(f"MODEL_SLUG: {MODEL_SLUG}")
            print(f"DB_PATH: {DB_PATH}")
            print(f"OUTPUT_TABLE_INGREDIENTS: {OUTPUT_TABLE_INGREDIENTS}")
            print(f"OUTPUT_INGREDIENT_PARQUET: {OUTPUT_INGREDIENT_PARQUET}")
            print(f"OUTPUT_REPORT_TXT: {OUTPUT_REPORT_TXT}")

            print("\n" + "=" * 100)
            print("PARAMETERS")
            print("=" * 100)
            print(f"THRESHOLD_LEVEL2: {THRESHOLD_LEVEL2}")
            print(f"THRESHOLD_LEVEL3: {THRESHOLD_LEVEL3}")
            print(f"THRESHOLD_SPECIFIC: {THRESHOLD_SPECIFIC}")
            print(f"ENCODE_BATCH_SIZE: {ENCODE_BATCH_SIZE}")
            print(f"INGREDIENT_CHUNK_SIZE: {INGREDIENT_CHUNK_SIZE}")
            print(f"TOP_K_L2: {TOP_K_L2}")
            print(f"TOP_K_L3: {TOP_K_L3}")
            print(f"TOP_K_SPECIFIC: {TOP_K_SPECIFIC}")

            total = con.execute(
                f"SELECT COUNT(*) FROM {OUTPUT_TABLE_INGREDIENTS}"
            ).fetchone()[0]

            print("\n" + "=" * 100)
            print("BASIC COUNTS")
            print("=" * 100)
            print(f"Matched unique ingredients: {total:,}")

            print("\n" + "=" * 100)
            print("MATCHING MODE DISTRIBUTION")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        matching_mode,
                        COUNT(*) AS n,
                        ROUND(100.0 * COUNT(*) / NULLIF({total}, 0), 2) AS pct,
                        ROUND(AVG(rf_score), 2) AS avg_rf_score,
                        ROUND(AVG(bge_score), 4) AS avg_bge_score,
                        ROUND(AVG(lexical_score), 4) AS avg_lexical,
                        ROUND(AVG(branch_penalty), 4) AS avg_branch_prior,
                        SUM(CASE WHEN exact_variant_match THEN 1 ELSE 0 END) AS n_exact
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    GROUP BY matching_mode
                    ORDER BY n DESC
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("RF SCORE DISTRIBUTION (rf_used = true)")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        confidence,
                        COUNT(*) AS n,
                        ROUND(AVG(rf_score), 2) AS avg_rf_score,
                        ROUND(MIN(rf_score), 2) AS min_rf_score,
                        ROUND(MAX(rf_score), 2) AS max_rf_score,
                        ROUND(AVG(bge_score), 4) AS avg_bge,
                        ROUND(AVG(lexical_score), 4) AS avg_lexical,
                        ROUND(AVG(branch_penalty), 4) AS avg_branch_prior
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE rf_used = TRUE
                    GROUP BY confidence
                    ORDER BY n DESC
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("CONFIDENCE DISTRIBUTION")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        confidence,
                        COUNT(*) AS n,
                        ROUND(100.0 * COUNT(*) / NULLIF({total}, 0), 2) AS pct
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    GROUP BY confidence
                    ORDER BY n DESC
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("FINAL LEVEL DISTRIBUTION")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        final_level,
                        COUNT(*) AS n,
                        ROUND(100.0 * COUNT(*) / NULLIF({total}, 0), 2) AS pct
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    GROUP BY final_level
                    ORDER BY final_level
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("SCORE SUMMARY BY CONFIDENCE")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        confidence,
                        COUNT(*) AS n,
                        ROUND(AVG(level2_score), 4) AS avg_l2,
                        ROUND(AVG(level3_score), 4) AS avg_l3,
                        ROUND(AVG(specific_score), 4) AS avg_specific,
                        ROUND(AVG(path_score), 4) AS avg_path,
                        ROUND(MIN(path_score), 4) AS min_path,
                        ROUND(MAX(path_score), 4) AS max_path
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    GROUP BY confidence
                    ORDER BY n DESC
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("RANDOM SAMPLE")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        ingredient,
                        level2_name,
                        level2_score,
                        level3_name,
                        level3_score,
                        specific_name,
                        specific_score,
                        final_level,
                        confidence,
                        path_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    ORDER BY random()
                    LIMIT 50
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("HIGH CONFIDENCE SAMPLE")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        ingredient,
                        level2_name,
                        level3_name,
                        specific_name,
                        specific_score,
                        final_level,
                        path_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE confidence = 'HIGH'
                    ORDER BY path_score DESC
                    LIMIT 50
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("MEDIUM CONFIDENCE SAMPLE")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        ingredient,
                        level2_name,
                        level3_name,
                        specific_name,
                        specific_score,
                        final_level,
                        path_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE confidence = 'MEDIUM'
                    ORDER BY random()
                    LIMIT 50
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("LOW / NO_MATCH SAMPLE")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        ingredient,
                        level2_name,
                        level2_score,
                        level3_name,
                        level3_score,
                        confidence,
                        path_score,
                        top1_name,
                        top1_score,
                        top2_name,
                        top2_score,
                        top3_name,
                        top3_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE confidence IN ('LOW', 'NO_MATCH')
                    ORDER BY random()
                    LIMIT 50
                """).df().to_string(index=False)
            )

            print("\n" + "=" * 100)
            print("KEY INGREDIENT CHECKS")
            print("=" * 100)

            key_ingredients = [
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

            for ing in key_ingredients:
                print("\n" + "-" * 100)
                print(f"Ingredient: {ing}")

                df = con.execute(f"""
                    SELECT
                        ingredient,
                        level2_name,
                        level2_score,
                        level3_name,
                        level3_score,
                        specific_name,
                        specific_score,
                        final_level,
                        confidence,
                        path_score,
                        top1_name,
                        top1_score,
                        top2_name,
                        top2_score,
                        top3_name,
                        top3_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE lower(ingredient) = lower(?)
                    LIMIT 20
                """, [ing]).df()

                if df.empty:
                    print("Not found in sampled/matched ingredients.")
                else:
                    print(df.to_string(index=False))

            print("\n" + "=" * 100)
            print("POTENTIALLY SUSPICIOUS HIGH MATCHES")
            print("=" * 100)
            print(
                con.execute(f"""
                    SELECT
                        ingredient,
                        specific_name,
                        specific_score,
                        level2_name,
                        level3_name,
                        final_level,
                        path_score
                    FROM {OUTPUT_TABLE_INGREDIENTS}
                    WHERE confidence = 'HIGH'
                      AND (
                          lower(ingredient) LIKE '% milk'
                          OR lower(ingredient) LIKE '% starch'
                          OR lower(ingredient) LIKE '% oil'
                          OR lower(ingredient) LIKE '% sugar'
                          OR lower(ingredient) LIKE '% syrup'
                      )
                    ORDER BY random()
                    LIMIT 80
                """).df().to_string(index=False)
            )

    print(f"TXT report saved to: {OUTPUT_REPORT_TXT}")

_KEY_INGREDIENTS = [
    "salt", "sugar", "apple juice", "burrata cheese", "vegetable oil",
    "potato starch", "rice milk", "oat milk", "semi skimmed milk",
    "corn syrup", "glucose syrup", "wheat flour", "olive oil",
    "tomato paste", "yogurt", "yoghurt",
]


def _probe_key_ingredients(hierarchy: dict, model) -> None:
    """Encode key ingredients and run them through the full RF+beam matcher.

    Runs before the main batch loop so results are visible regardless of sampling.
    Prints a compact table showing which path RF vs. beam search took.
    """
    print("\n" + "=" * 80)
    print("KEY INGREDIENT PROBE  (RF primary + beam fallback)")
    print("=" * 80)

    probe_embs = model.encode(
        _KEY_INGREDIENTS,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    probe_idx = {ing: i for i, ing in enumerate(_KEY_INGREDIENTS)}

    matcher = StratifiedMatcherOptimized(
        hierarchy=hierarchy,
        ingredient_embeddings=probe_embs,
        ingredient_to_idx=probe_idx,
        threshold_level2=THRESHOLD_LEVEL2,
        threshold_level3=THRESHOLD_LEVEL3,
        threshold_specific=THRESHOLD_SPECIFIC,
        top_k_l2=TOP_K_L2,
        top_k_l3=TOP_K_L3,
        top_k_specific=TOP_K_SPECIFIC,
        use_flat_rapidfuzz_short_filter=True,
        rf_min_score=65.0,
        rf_top_k_flat_variants=100,
        rf_top_k_flat_terms=30,
    )

    fmt = "{:<25} {:^6} {:^5} {:<30} {:<30} {:>6}"
    print(fmt.format("ingredient", "conf", "level", "level3_name", "specific_name", "score"))
    print("-" * 80)
    for ing, result in zip(_KEY_INGREDIENTS, matcher.match_batch(_KEY_INGREDIENTS, show_progress=False)):
        print(fmt.format(
            ing[:24],
            result.confidence,
            result.final_level,
            (result.level3_name or "")[:29],
            (result.specific_name or "")[:29],
            f"{result.path_score:.3f}",
        ))
    print("=" * 80 + "\n")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    try:
        terms = con.execute(
            """
            SELECT termCode, termExtendedName, reportHierarchyCode, reportHierarchyLevel
            FROM foodex2_terms
            """
        ).fetchdf()

        n_products = con.execute("SELECT COUNT(*) FROM products_ingredients").fetchone()[0]
        n_matchable = con.execute(
            f"SELECT COUNT(*) FROM products_ingredients WHERE {MATCHABLE_WHERE}"
        ).fetchone()[0]
        print(
            f"Loaded {len(terms):,} FoodEx2 terms | "
            f"{n_products:,} product-ingredient rows | "
            f"{n_matchable:,} English non-additive matchable rows"
        )
        print(f"Using model: {MODEL_NAME}")
        print(
            f"Ingredient chunk size={INGREDIENT_CHUNK_SIZE:,}, "
            f"encode batch size={ENCODE_BATCH_SIZE}, "
            f"beam=({TOP_K_L2},{TOP_K_L3},{TOP_K_SPECIFIC})"
        )

        hierarchy = build_hierarchy_index(terms)
        model = SentenceTransformer(MODEL_NAME)
        hierarchy = generate_hierarchy_embeddings(hierarchy, model)
        build_flat_rapidfuzz_index(hierarchy)
        print(f"Built flat RapidFuzz index: {len(hierarchy['flat_rf']['variants']):,} variants")

        _probe_key_ingredients(hierarchy, model)

        unique_ingredients = con.execute(f"""
            SELECT DISTINCT ingredient
            FROM products_ingredients
            WHERE ingredient IS NOT NULL
              AND matchable = TRUE
              AND tag_lang = 'en'
              AND COALESCE(is_additive, FALSE) = FALSE
            ORDER BY ingredient
        """).df()["ingredient"].tolist()
        print(f"Found {len(unique_ingredients):,} unique ingredients to match")

        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE_INGREDIENTS}")
        con.execute(f"CREATE TABLE {OUTPUT_TABLE_INGREDIENTS} ({RESULT_COLUMNS_SQL})")

        for start, chunk in batched(unique_ingredients, INGREDIENT_CHUNK_SIZE):
            end = start + len(chunk)
            print(f"Matching ingredients {start + 1:,}-{end:,} / {len(unique_ingredients):,}")

            # Encode only this chunk. Cast to float32 to avoid accidental float64 memory use.
            ingredient_embeddings = model.encode(
                chunk,
                batch_size=ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)

            ingredient_to_idx = {ing: idx for idx, ing in enumerate(chunk)}
            matcher = StratifiedMatcherOptimized(
                hierarchy=hierarchy,
                ingredient_embeddings=ingredient_embeddings,
                ingredient_to_idx=ingredient_to_idx,
                threshold_level2=THRESHOLD_LEVEL2,
                threshold_level3=THRESHOLD_LEVEL3,
                threshold_specific=THRESHOLD_SPECIFIC,
                top_k_l2=TOP_K_L2,
                top_k_l3=TOP_K_L3,
                top_k_specific=TOP_K_SPECIFIC,
                use_flat_rapidfuzz_short_filter=True,
                rf_min_score=65.0,
                rf_top_k_flat_variants=100,
                rf_top_k_flat_terms=30,
            )

            results = matcher.match_batch(chunk, show_progress=False)
            chunk_df = pd.DataFrame(result_records(results))

            con.register("chunk_results", chunk_df)
            con.execute(f"INSERT INTO {OUTPUT_TABLE_INGREDIENTS} SELECT * FROM chunk_results")
            con.unregister("chunk_results")

            del ingredient_embeddings, ingredient_to_idx, matcher, results, chunk_df
            gc.collect()

        con.execute(f"COPY {OUTPUT_TABLE_INGREDIENTS} TO '{OUTPUT_INGREDIENT_PARQUET}' (FORMAT PARQUET)")
        write_matching_report(con)

        # Stream the product-level join directly to parquet — no intermediate table,
        # one pass over the data instead of two.
        print(f"Exporting product-level join → {OUTPUT_PRODUCTS_PARQUET} …")
        con.execute(f"""
            COPY (
                SELECT
                    p.code, p.lang_product_name, p.product_name, p.product_context,
                    p.original_tag, p.tag_lang, p.ingredient, p.nlp_text,
                    p.is_additive, p.is_noise_phrase, p.is_context_term, p.matchable,
                    i.level2_code, i.level2_name, i.level2_score,
                    i.level3_code, i.level3_name, i.level3_score,
                    i.specific_code, i.specific_name, i.specific_score,
                    i.final_level, i.confidence, i.path_score,
                    i.top1_code, i.top1_name, i.top1_score,
                    i.top2_code, i.top2_name, i.top2_score,
                    i.top3_code, i.top3_name, i.top3_score,
                    i.model_name
                FROM products_ingredients p
                LEFT JOIN {OUTPUT_TABLE_INGREDIENTS} i
                  ON p.ingredient = i.ingredient
                 AND p.matchable = TRUE
                 AND p.tag_lang = 'en'
                 AND COALESCE(p.is_additive, FALSE) = FALSE
            )
            TO '{OUTPUT_PRODUCTS_PARQUET}' (FORMAT PARQUET)
        """)
        summary = con.execute(f"""
            SELECT confidence, COUNT(*) AS n
            FROM {OUTPUT_TABLE_INGREDIENTS}
            GROUP BY confidence
            ORDER BY n DESC
        """).fetchdf()
        print("\nIngredient-level confidence summary:")
        print(summary.to_string(index=False))

    finally:
        con.close()

    print(f"Saved ingredient-level parquet → {OUTPUT_INGREDIENT_PARQUET}")
    print(f"Saved product-level parquet   → {OUTPUT_PRODUCTS_PARQUET}")
    print(f"Saved TXT report              → {OUTPUT_REPORT_TXT}")


if __name__ == "__main__":
    main()
