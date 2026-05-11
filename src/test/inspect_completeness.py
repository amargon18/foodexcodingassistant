"""
Inspect the `completeness` column in the products table and evaluate whether
it is a representative measure of actual data quality.

Analysis plan:
  1. Basic stats and null coverage
  2. Value distribution (histogram buckets)
  3. Cross-validation: actual fill-rate of key columns per completeness bucket
  4. Correlation between completeness and real fill-rate
  5. Summary verdict
"""

from pathlib import Path
from contextlib import redirect_stdout

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"

OUTPUT_DIR = PROJECT_ROOT / "src" / "test" / "output"
OUTPUT_TXT = OUTPUT_DIR / "completeness_inspection_output"

# Key quality columns whose fill-rate we compare against `completeness`
KEY_COLUMNS = [
    "product_name",
    "generic_name",
    "brands",
    "categories",
    "countries",
    "ingredients_text",
    "allergens",
    "traces",
    "quantity",
    "packaging",
    "labels",
    "origins",
    "manufacturing_places",
    "emb_codes",
    "serving_size",
    "energy_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. Basic stats
# ---------------------------------------------------------------------------

def show_basic_stats(con: duckdb.DuckDBPyConnection) -> None:
    header("1. BASIC STATISTICS — completeness column")

    stats = con.execute("""
        SELECT
            COUNT(*)                                    AS total_rows,
            COUNT(completeness)                         AS non_null,
            COUNT(*) - COUNT(completeness)              AS null_count,
            ROUND(100.0 * COUNT(completeness) / COUNT(*), 2) AS pct_non_null,
            ROUND(MIN(completeness), 4)                 AS min_val,
            ROUND(MAX(completeness), 4)                 AS max_val,
            ROUND(AVG(completeness), 4)                 AS mean_val,
            ROUND(MEDIAN(completeness), 4)              AS median_val,
            ROUND(STDDEV(completeness), 4)              AS std_val,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY completeness), 4) AS p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY completeness), 4) AS p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY completeness), 4) AS p90,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY completeness), 4) AS p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY completeness), 4) AS p99
        FROM products
    """).fetchdf()

    for col in stats.columns:
        print(f"  {col:<20}: {stats[col].iloc[0]}")


# ---------------------------------------------------------------------------
# 2. Value distribution
# ---------------------------------------------------------------------------

def show_distribution(con: duckdb.DuckDBPyConnection) -> None:
    header("2. DISTRIBUTION — histogram in 0.1-wide buckets")

    dist = con.execute("""
        SELECT
            FLOOR(completeness * 10) / 10   AS bucket_start,
            COUNT(*)                         AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM products
        WHERE completeness IS NOT NULL
        GROUP BY bucket_start
        ORDER BY bucket_start
    """).fetchdf()

    print(f"\n  {'Bucket':>10}  {'Count':>10}  {'%':>6}  Bar")
    for _, row in dist.iterrows():
        bar = "#" * int(row["pct"] / 0.5)   # 1 char per 0.5%
        print(f"  [{row['bucket_start']:.1f}-{row['bucket_start']+0.1:.1f})  "
              f"{int(row['count']):>10,}  {row['pct']:>5.1f}%  {bar}")

    # Values above 1.0 (anomalous range)
    over_one = con.execute(
        "SELECT COUNT(*) AS cnt FROM products WHERE completeness > 1.0"
    ).fetchone()[0]
    print(f"\n  Products with completeness > 1.0: {over_one:,}")


# ---------------------------------------------------------------------------
# 3. Fill-rate of key columns per completeness bucket
# ---------------------------------------------------------------------------

def _available_key_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return only those KEY_COLUMNS that actually exist in the table."""
    existing = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'products'"
        ).fetchall()
    }
    return [c for c in KEY_COLUMNS if c in existing]


def show_fillrate_by_bucket(con: duckdb.DuckDBPyConnection) -> None:
    header("3. ACTUAL FILL-RATE OF KEY COLUMNS PER COMPLETENESS BUCKET")

    cols = _available_key_columns(con)
    if not cols:
        print("  No key columns found in products table.")
        return

    # Build a query that counts non-null values per bucket for each key column
    non_null_exprs = ",\n            ".join(
        f"ROUND(100.0 * COUNT({c}) / COUNT(*), 1) AS \"{c}\""
        for c in cols
    )

    query = f"""
        SELECT
            FLOOR(completeness * 10) / 10  AS bucket,
            COUNT(*)                        AS n_products,
            {non_null_exprs}
        FROM products
        WHERE completeness IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """

    df = con.execute(query).fetchdf()

    # Print wide table in two passes to keep line width manageable
    chunk_size = 8
    col_chunks = [cols[i:i+chunk_size] for i in range(0, len(cols), chunk_size)]

    for chunk in col_chunks:
        display_cols = ["bucket", "n_products"] + chunk
        print()
        print(df[display_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Correlation between completeness and real fill-rate
# ---------------------------------------------------------------------------

def show_correlation(con: duckdb.DuckDBPyConnection) -> None:
    header("4. CORRELATION — completeness vs. actual field fill-rate")

    cols = _available_key_columns(con)
    if not cols:
        print("  No key columns found.")
        return

    # Compute per-row fill_rate = fraction of key cols that are non-null
    case_exprs = " + ".join(
        f"(CASE WHEN {c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR)) <> '' THEN 1 ELSE 0 END)"
        for c in cols
    )
    n = len(cols)

    query = f"""
        SELECT
            ROUND(CORR(completeness, ({case_exprs}) * 1.0 / {n}), 4) AS pearson_r,
            ROUND(AVG(completeness), 4)                                AS mean_completeness,
            ROUND(AVG(({case_exprs}) * 1.0 / {n}), 4)                AS mean_fill_rate
        FROM products
        WHERE completeness IS NOT NULL
    """

    row = con.execute(query).fetchone()
    print(f"\n  Pearson r (completeness vs fill_rate) : {row[0]}")
    print(f"  Mean completeness value               : {row[1]}")
    print(f"  Mean actual fill-rate ({n} key cols)  : {row[2]}")
    print(f"\n  Key columns used ({n}):")
    for c in cols:
        print(f"    - {c}")


# ---------------------------------------------------------------------------
# 5. Sample products at extreme completeness values
# ---------------------------------------------------------------------------

def show_extreme_samples(con: duckdb.DuckDBPyConnection) -> None:
    header("5. SAMPLE PRODUCTS AT EXTREME COMPLETENESS VALUES")

    display_cols = ["code", "product_name", "completeness",
                    "ingredients_text", "allergens", "categories"]
    existing = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'products'"
        ).fetchall()
    }
    select_cols = ", ".join(c for c in display_cols if c in existing)

    for label, order in [("LOW (≤ 0.2)", "ASC"), ("HIGH (≥ 0.9)", "DESC")]:
        print(f"\n  --- {label} completeness ---")
        df = con.execute(f"""
            SELECT {select_cols}
            FROM products
            WHERE completeness IS NOT NULL
            ORDER BY completeness {order}
            LIMIT 5
        """).fetchdf()
        pd.set_option("display.max_colwidth", 60)
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Verdict
# ---------------------------------------------------------------------------

def print_verdict(con: duckdb.DuckDBPyConnection) -> None:
    header("6. SUMMARY VERDICT")

    cols = _available_key_columns(con)
    n = len(cols)
    case_exprs = " + ".join(
        f"(CASE WHEN {c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR)) <> '' THEN 1 ELSE 0 END)"
        for c in cols
    )

    row = con.execute(f"""
        SELECT
            ROUND(CORR(completeness, ({case_exprs}) * 1.0 / {n}), 4) AS r,
            ROUND(AVG(completeness), 4)                                AS mean_c,
            ROUND(AVG(({case_exprs}) * 1.0 / {n}), 4)                AS mean_fr
        FROM products WHERE completeness IS NOT NULL
    """).fetchone()

    r, mean_c, mean_fr = row

    print(f"""
  Pearson correlation between `completeness` and actual field fill-rate: {r}

  Interpretation:
    - r >= 0.80  →  strong agreement; completeness is representative
    - 0.50-0.79  →  moderate agreement; useful but imperfect proxy
    - < 0.50     →  weak agreement; completeness alone is not sufficient

  Mean `completeness` score : {mean_c}
  Mean actual fill-rate     : {mean_fr}
  Gap (completeness - fill) : {round(mean_c - mean_fr, 4)}

  Conclusion:
  {"✔  completeness IS a representative proxy for product data quality." if r is not None and r >= 0.7
   else "△  completeness is a PARTIAL proxy — supplement with per-field fill-rate checks."
   if r is not None and r >= 0.5
   else "✘  completeness is NOT a reliable proxy for data quality in this dataset."}
""")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_inspection() -> None:
    print(f"DB_PATH      : {DB_PATH}")
    print(f"OUTPUT_FILE  : {OUTPUT_TXT}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        show_basic_stats(con)
        show_distribution(con)
        show_fillrate_by_bucket(con)
        show_correlation(con)
        show_extreme_samples(con)
        print_verdict(con)
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
