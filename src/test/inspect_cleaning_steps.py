from pathlib import Path
from contextlib import redirect_stdout

import duckdb
import pandas as pd

from etl.cleaning import (
    build_ingredient_rows,
    load_ingredient_taxonomy,
    normalize_with_taxonomy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"
TAXONOMY_PATH = PROJECT_ROOT / "nlp" / "ingredients.txt"

OUTPUT_DIR = PROJECT_ROOT / "src" / "test" / "output"
OUTPUT_TXT = OUTPUT_DIR / "cleaning_test_output"

SOURCE_QUERY = """
SELECT 
    code, 
    lang_product_name, 
    product_name, 
    ingredients_tags
FROM products_with_language1
WHERE lang_ingredients_text = 'en'
  AND ingredients_tags IS NOT NULL
  AND NOT regexp_matches(
        COALESCE(categories_tags, ''),
        'en:beauty-products|en:cosmetics|en:hygiene-products|'
        'en:pet-food|en:cat-food|en:dog-food|en:animal-food|en:bird-food|'
        'en:cleaning-products|en:detergents|en:non-food-products'
    )
  AND food_groups_tags IS NOT NULL
ORDER BY random()
LIMIT 20
"""


def show_raw_products(raw: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("RAW PRODUCTS")
    print("=" * 80)

    for i, row in raw.iterrows():
        print(f"\n[{i}] code: {row['code']}")
        print(f"product_name: {row['product_name']}")
        print(f"lang_product_name: {row['lang_product_name']}")
        print(f"ingredients_tags raw type: {type(row['ingredients_tags'])}")
        print(f"ingredients_tags: {row['ingredients_tags']}")


def show_expanded_rows(expanded: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("EXPANDED INGREDIENT ROWS")
    print("=" * 80)

    print(f"\nShape: {expanded.shape}")
    print("\nColumns:")
    print(list(expanded.columns))

    print("\nFirst rows:")
    print(expanded.head(100).to_string(index=False))


def show_grouped_by_product(expanded: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("INGREDIENTS GROUPED BY PRODUCT")
    print("=" * 80)

    if expanded.empty:
        print("\nNo expanded ingredient rows generated.")
        return

    for code, group in expanded.groupby("code"):
        product_name = group["product_name"].iloc[0] if "product_name" in group else ""
        print(f"\nProduct code: {code}")
        print(f"Product name: {product_name}")
        print(f"Number of extracted ingredients: {len(group)}")

        for _, row in group.iterrows():
            ingredient = row.get("ingredient", "")
            nlp_text = row.get("nlp_text", None)

            if nlp_text is not None:
                print(f"  - {ingredient}  ->  {nlp_text}")
            else:
                print(f"  - {ingredient}")


def inspect_one_product(raw: pd.DataFrame, product_index: int = 0) -> None:
    print("\n" + "=" * 80)
    print("ONE-PRODUCT INSPECTION")
    print("=" * 80)

    if raw.empty:
        print("\nNo raw products available.")
        return

    one = raw.iloc[[product_index]].copy()

    print("\nInput row:")
    print(one.to_string(index=False))

    expanded_one = build_ingredient_rows(one)

    print("\nOutput rows from build_ingredient_rows:")
    if expanded_one.empty:
        print("No ingredients extracted.")
    else:
        print(expanded_one.to_string(index=False))


def run_inspection() -> None:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DB_PATH: {DB_PATH}")
    print(f"TAXONOMY_PATH: {TAXONOMY_PATH}")
    print(f"OUTPUT_TXT: {OUTPUT_TXT}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    taxonomy = {}
    if TAXONOMY_PATH.exists():
        taxonomy = load_ingredient_taxonomy(TAXONOMY_PATH)
        print(f"\nLoaded taxonomy entries: {len(taxonomy):,}")
    else:
        print(f"\nWarning: taxonomy file not found at {TAXONOMY_PATH}")

    con = duckdb.connect(str(DB_PATH))

    try:
        raw = con.execute(SOURCE_QUERY).fetchdf()

        print(f"\nLoaded raw products: {len(raw):,}")

        show_raw_products(raw)

        expanded = build_ingredient_rows(raw)

        if taxonomy and not expanded.empty:
            expanded["nlp_text"] = expanded["ingredient"].apply(
                lambda x: normalize_with_taxonomy(x, taxonomy)
            )

        show_expanded_rows(expanded)
        show_grouped_by_product(expanded)
        inspect_one_product(raw, product_index=0)

    finally:
        con.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_inspection()

    print(f"Inspection exported to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()