from pathlib import Path

import duckdb
import pandas as pd

from cleaning import build_ingredient_rows, load_ingredient_taxonomy, normalize_with_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"
TAXONOMY_PATH = PROJECT_ROOT / "nlp" / "ingredients.txt"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = OUTPUT_DIR / "products_ingredients.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "products_ingredients.parquet"
OUTPUT_TABLE = "products_ingredients"

SOURCE_QUERY = """
SELECT code, lang_product_name, product_name, ingredients_tags
FROM products_with_language1 TABLESAMPLE bernoulli(10 PERCENT)
WHERE lang_ingredients_text = 'en'
  AND ingredients_tags IS NOT NULL
  AND NOT regexp_matches(
        COALESCE(categories_tags, ''),
        'en:beauty-products|en:cosmetics|en:hygiene-products|'
        'en:pet-food|en:cat-food|en:dog-food|en:animal-food|en:bird-food|'
        'en:cleaning-products|en:detergents|en:non-food-products'
    )
  AND food_groups_tags IS NOT NULL
"""


def main(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    taxonomy = {}
    if TAXONOMY_PATH.exists():
        taxonomy = load_ingredient_taxonomy(TAXONOMY_PATH)
    else:
        print(f"Warning: taxonomy file not found at {TAXONOMY_PATH}. Skipping taxonomy normalization.")

    con = duckdb.connect(str(DB_PATH))
    try:
        raw = con.execute(SOURCE_QUERY).fetchdf()
        print(f"Loaded {len(raw):,} source products for ingredient extraction")

        expanded = build_ingredient_rows(raw)
        if taxonomy:
            expanded["nlp_text"] = expanded["ingredient"].apply(
                lambda x: normalize_with_taxonomy(x, taxonomy)
            )

        print(f"Saving expanded ingredient dataset: {len(expanded):,} rows")
        expanded.to_csv(OUTPUT_CSV, index=False)
        expanded.to_parquet(OUTPUT_PARQUET, index=False)

        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
        con.register("df_temp", expanded)
        con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df_temp")
        print(f"Created DuckDB table '{OUTPUT_TABLE}'")
    finally:
        con.close()


if __name__ == "__main__":
    main()
