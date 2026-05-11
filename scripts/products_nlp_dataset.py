"""
Build the full NLP-ready dataset from products_with_language1.

Applies the same transformations as the EDA notebook (Cell 21) to all 1.2M rows
and exports to products_nlp_full.csv (and optionally a DuckDB table).

Usage:
    python build_nlp_dataset.py [--output-table]  # --output-table writes back to DuckDB
"""

import re
import sys
import argparse
import duckdb
import pandas as pd

DB_PATH = "/home/alexl/TFM/openfoodfacts.duckdb"
OUTPUT_CSV = "/home/alexl/TFM/products_nlp_full.csv"
OUTPUT_TABLE = "products_nlp"
CHUNK_SIZE = 50_000
SAMPLE_PCT = 100  # Default: full dataset


# ---------------------------------------------------------------------------
# NLP helpers (identical logic to the notebook)
# ---------------------------------------------------------------------------

def clean_text(text, max_chars=None):
    """Lowercase-preserving normalize: whitespace, non-printable, optional truncation."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\xC0-\xFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars:
        text = text[:max_chars]
    return text


def parse_tags_to_text(tags_str):
    """
    'en:cereals,en:bread' → 'cereals, bread'
    Strips 2-letter lang prefix and replaces hyphens/underscores with spaces.
    """
    if pd.isna(tags_str) or str(tags_str).strip() == "":
        return ""
    tags = [t.strip() for t in str(tags_str).split(",")]
    cleaned = []
    for t in tags:
        t = re.sub(r"^[a-z]{2}:", "", t)
        t = t.replace("-", " ").replace("_", " ").strip()
        if t:
            cleaned.append(t)
    return ", ".join(cleaned)


def build_nlp_text(row, ingredients_max_chars=500):
    """
    Concatenates (in priority order):
      1. product_name
      2. generic_name   (if different from product_name)
      3. categories_tags (cleaned)
      4. food_groups_tags (cleaned)
      5. ingredients_text (truncated)

    Each non-empty part is appended with a period separator.
    """
    parts = []

    name = clean_text(row.get("product_name"))
    if name:
        parts.append(name)

    generic = clean_text(row.get("generic_name"))
    if generic and generic.lower() != name.lower():
        parts.append(generic)

    cats = parse_tags_to_text(row.get("categories_tags"))
    if cats:
        parts.append(cats)

    food_grp = parse_tags_to_text(row.get("food_groups_tags"))
    if food_grp:
        parts.append(food_grp)

    ingredients = clean_text(row.get("ingredients_text"), max_chars=ingredients_max_chars)
    if ingredients:
        parts.append(ingredients)

    text_parts = []
    for p in parts:
        p = p.rstrip(".")
        if p:
            text_parts.append(p + ".")

    return " ".join(text_parts)


# ---------------------------------------------------------------------------
# Per-chunk transform
# ---------------------------------------------------------------------------

COLS_NEEDED = [
    "code",
    "lang_product_name",
    "lang_ingredients_text",
    "product_name",
    "generic_name",
    "categories_tags",
    "food_groups_tags",
    "ingredients_text",
    "nutrition-score-fr_100g",
    "created_datetime",
]


def transform_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["product_name_clean"] = df["product_name"].apply(clean_text)
    df["generic_name_clean"] = df["generic_name"].apply(clean_text)
    df["ingredients_text_clean"] = df["ingredients_text"].apply(
        lambda x: clean_text(x, max_chars=500)
    )
    df["categories_clean"] = df["categories_tags"].apply(parse_tags_to_text)
    df["food_groups_clean"] = df["food_groups_tags"].apply(parse_tags_to_text)

    df["nlp_text"] = df.apply(build_nlp_text, axis=1)
    df["nlp_word_count"] = df["nlp_text"].str.split().str.len()
    df["approx_tokens"] = (df["nlp_word_count"] * 1.3).round().astype("Int64")

    return df[[
        "code",
        "lang_product_name",
        "lang_ingredients_text",
        "product_name_clean",
        "generic_name_clean",
        "categories_clean",
        "food_groups_clean",
        "ingredients_text_clean",
        "nlp_text",
        "nlp_word_count",
        "approx_tokens",
        "nutrition-score-fr_100g",
        "created_datetime",
    ]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-table",
        action="store_true",
        help="Also write results as a DuckDB table named products_nlp",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=100,
        help="Percentage of data to sample (0-100, default: 100 for full dataset)",
    )
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)

    total = con.execute("SELECT COUNT(*) FROM products_with_language1").fetchone()[0]
    print(f"Total rows in products_with_language1: {total:,}")

    cols_sql = ", ".join(f'"{c}"' for c in COLS_NEEDED)

    # Build query with optional SAMPLE clause
    query = f"SELECT {cols_sql} FROM products_with_language1"
    output_file = OUTPUT_CSV

    if args.sample < 100:
        query += f" TABLESAMPLE {args.sample} PERCENT"
        output_file = OUTPUT_CSV.replace(".csv", f"_sample_{args.sample:.0f}pct.csv")
        print(f"Using {args.sample}% sample")

    # Get actual sample size
    sample_query = f"SELECT COUNT(*) FROM ({query})"
    sample_total = con.execute(sample_query).fetchone()[0]
    print(f"Sample size: {sample_total:,} rows")

    first_chunk = True
    processed = 0

    if args.output_table:
        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")

    cursor = con.execute(query)
    with open(output_file, "w") as f_out:
        while True:
            chunk_df = cursor.fetch_df_chunk(CHUNK_SIZE)
            if chunk_df.empty:
                break

            result = transform_chunk(chunk_df)

            result.to_csv(f_out, index=False, header=first_chunk)
            first_chunk = False

            if args.output_table:
                if processed == 0:
                    con.execute(
                        f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM result LIMIT 0"
                    )
                con.execute(f"INSERT INTO {OUTPUT_TABLE} SELECT * FROM result")

            processed += len(result)
            pct = processed / sample_total * 100
            print(f"  {processed:>10,} / {sample_total:,}  ({pct:.1f}%)", end="\r", flush=True)

    print(f"\nDone. {processed:,} rows written to {output_file}")

    if args.output_table:
        print(f"Table '{OUTPUT_TABLE}' written to {DB_PATH}")

    # Quick quality check
    sample = pd.read_csv(output_file, nrows=5)
    print("\nSample output:")
    print(sample[["code", "lang_product_name", "nlp_text", "approx_tokens"]].to_string())

    empty = pd.read_csv(output_file).query("nlp_text.str.strip() == ''", engine="python")
    print(f"\nRows with empty nlp_text: {len(empty):,} ({len(empty)/processed*100:.2f}%)")


if __name__ == "__main__":
    main()
