from pathlib import Path
import duckdb
import pandas as pd

from src.etl.cleaning import parse_ingredient_tags

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"

# Query to get some sample products with ingredients
SAMPLE_QUERY = """
SELECT code, product_name, ingredients_tags
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
LIMIT 10
"""

con = duckdb.connect(str(DB_PATH))
try:
    df = con.execute(SAMPLE_QUERY).fetchdf()
    print(f"Loaded {len(df)} sample products\n")

    for idx, row in df.iterrows():
        print(f"Product {idx + 1}: {row['product_name']}")
        print(f"  Raw ingredients_tags: {row['ingredients_tags'][:100]}...")

        # Show all tags
        all_tags = str(row['ingredients_tags']).split(",")
        languages = {}
        for tag in all_tags:
            tag = tag.strip()
            if ':' in tag:
                lang = tag.split(':')[0]
                languages[lang] = languages.get(lang, 0) + 1
        print(f"  Language breakdown: {languages}")

        # Show filtered result
        filtered = parse_ingredient_tags(row['ingredients_tags'])
        print(f"  Filtered ingredients (en: only): {filtered[:5]}{'...' if len(filtered) > 5 else ''}")
        print(f"  Total after filtering: {len(filtered)} ingredients\n")

finally:
    con.close()
