from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from stratified_matcher import (
    StratifiedMatcherOptimized,
    build_hierarchy_index,
    generate_hierarchy_embeddings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EMBEDDINGS = OUTPUT_DIR / "embeddings"
OUTPUT_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

# Experimental model: strong English retrieval model, suitable for short query -> term matching.
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_SLUG = MODEL_NAME.replace("/", "__")

OUTPUT_TABLE_INGREDIENTS = "matching_results_ingredients_bge"
OUTPUT_TABLE_PRODUCTS = "matching_results_bge"
OUTPUT_INGREDIENT_PARQUET = OUTPUT_DIR / f"matching_results_ingredients_{MODEL_SLUG}.parquet"
OUTPUT_PRODUCTS_PARQUET = OUTPUT_DIR / f"matching_results_{MODEL_SLUG}.parquet"

# BGE cosine score calibration differs from all-mpnet-base-v2; start slightly less strict
# and recalibrate after manual inspection of 100-200 examples.
THRESHOLD_LEVEL2 = 0.40
THRESHOLD_LEVEL3 = 0.50
THRESHOLD_SPECIFIC = 0.70
BATCH_SIZE = 128

MATCHABLE_WHERE = """
    ingredient IS NOT NULL
    AND matchable = TRUE
    AND tag_lang = 'en'
    AND COALESCE(is_additive, FALSE) = FALSE
"""


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    try:
        # Matching is intentionally based only on termExtendedName.
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
            f"{n_products:,} product-ingredient rows "
            f"({n_matchable:,} English non-additive matchable rows)"
        )

        hierarchy = build_hierarchy_index(terms)
        print(f"Using embedding model: {MODEL_NAME}")
        print("Scope: English tags only, additives excluded, context/noise excluded")
        model = SentenceTransformer(MODEL_NAME)
        hierarchy = generate_hierarchy_embeddings(hierarchy, model)

        # Only embed unique English, non-additive, matchable ingredients.
        unique_ingredients = con.execute(
            f"""
            SELECT DISTINCT ingredient
            FROM products_ingredients
            WHERE {MATCHABLE_WHERE}
            """
        ).df()["ingredient"].tolist()
        print(f"Found {len(unique_ingredients):,} unique ingredients to embed")

        ingredient_embeddings = model.encode(
            unique_ingredients,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(OUTPUT_EMBEDDINGS / f"ingredient_embeddings_{MODEL_SLUG}.npy", ingredient_embeddings)

        ingredient_to_idx = {ing: idx for idx, ing in enumerate(unique_ingredients)}
        matcher = StratifiedMatcherOptimized(
            hierarchy=hierarchy,
            ingredient_embeddings=ingredient_embeddings,
            ingredient_to_idx=ingredient_to_idx,
            threshold_level2=THRESHOLD_LEVEL2,
            threshold_level3=THRESHOLD_LEVEL3,
            threshold_specific=THRESHOLD_SPECIFIC,
        )

        results = matcher.match_batch(unique_ingredients, show_progress=False)
        records = [
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
            }
            for r in results
        ]
        ingredient_results = pd.DataFrame(records)
        ingredient_results.to_parquet(OUTPUT_INGREDIENT_PARQUET, index=False)

        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE_INGREDIENTS}")
        con.register("ingredient_results", ingredient_results)
        con.execute(
            f"CREATE TABLE {OUTPUT_TABLE_INGREDIENTS} AS SELECT * FROM ingredient_results"
        )

        # Join inside DuckDB. Non-English, additive, context, and noise rows intentionally
        # keep NULL match columns because they are outside the matching objective.
        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE_PRODUCTS}")
        con.execute(f"""
            CREATE TABLE {OUTPUT_TABLE_PRODUCTS} AS
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
        """)
        con.execute(
            f"COPY {OUTPUT_TABLE_PRODUCTS} TO '{OUTPUT_PRODUCTS_PARQUET}' (FORMAT PARQUET)"
        )
    finally:
        con.close()

    print(f"Saved ingredient-level matching to {OUTPUT_INGREDIENT_PARQUET}")
    print(f"Saved full products matching to {OUTPUT_PRODUCTS_PARQUET}")


if __name__ == "__main__":
    main()
