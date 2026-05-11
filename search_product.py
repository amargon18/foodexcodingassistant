"""
Search for a product by its code in the OpenFoodFacts database.
Usage: python search_product.py <code>
"""

import sys
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"


def search_product(code: str, verbose: bool = False) -> dict:
    """
    Search for a product by code in the database.

    Args:
        code: Product code to search for
        verbose: If True, show detailed information

    Returns:
        Dictionary with product data or None if not found
    """
    con = duckdb.connect(str(DB_PATH))

    try:
        # Search in products_with_language1 first (English products)
        product = con.execute(
            """
            SELECT *
            FROM products_with_language1
            WHERE code = ?
            LIMIT 1
            """,
            [code],
        ).fetchone()

        if product:
            columns = [desc[0] for desc in con.description]
            result = dict(zip(columns, product))
            result["_source"] = "products_with_language1"
            return result

        # If not found, search in main products table
        product = con.execute(
            """
            SELECT *
            FROM products
            WHERE code = ?
            LIMIT 1
            """,
            [code],
        ).fetchone()

        if product:
            columns = [desc[0] for desc in con.description]
            result = dict(zip(columns, product))
            result["_source"] = "products"
            return result

        return None

    finally:
        con.close()


def display_product(product: dict) -> None:
    """Display product information in a readable format."""
    if not product:
        print("❌ Product not found")
        return

    source = product.pop("_source", "unknown")
    print(f"\n{'='*70}")
    print(f"  PRODUCT FOUND (from: {source})")
    print(f"{'='*70}")

    # Key fields to display first
    key_fields = [
        "code",
        "product_name",
        "generic_name",
        "lang",
        "lang_product_name",
        "lang_ingredients_text",
    ]

    for field in key_fields:
        if field in product:
            value = product.pop(field)
            if value is not None:
                if isinstance(value, str) and len(value) > 80:
                    print(f"\n  {field}:")
                    print(f"    {value[:80]}...")
                else:
                    print(f"  {field:<30} {value}")

    # Ingredients tags
    if "ingredients_tags" in product:
        tags = product.pop("ingredients_tags")
        if tags:
            tags_list = str(tags).split(",")[:10]
            print(f"\n  ingredients_tags ({len(str(tags).split(','))} total):")
            for tag in tags_list:
                print(f"    - {tag.strip()}")
            if len(str(tags).split(",")) > 10:
                print(f"    ... and {len(str(tags).split(',')) - 10} more")

    # Other relevant fields
    other_fields = [
        "ingredients_text",
        "allergens_tags",
        "categories_tags",
        "food_groups_tags",
        "brands_tags",
        "countries_tags",
    ]

    for field in other_fields:
        if field in product and product[field]:
            value = product.pop(field)
            if isinstance(value, str) and len(value) > 80:
                print(f"\n  {field}:")
                print(f"    {value[:80]}...")
            else:
                print(f"  {field:<30} {value}")

    # Show remaining fields if verbose
    if product:
        print(f"\n  Other fields: {len(product)} more")

    print(f"\n{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python search_product.py <product_code> [--verbose]")
        print("\nExample:")
        print("  python search_product.py 5060292484616")
        print("  python search_product.py 3017620425035 --verbose")
        sys.exit(1)

    code = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Searching for product code: {code}...")

    product = search_product(code, verbose)
    display_product(product)


if __name__ == "__main__":
    main()

