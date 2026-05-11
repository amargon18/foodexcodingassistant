"""
Inspection / evaluation script for the FoodEx2 matching pipeline.

Reads directly from DuckDB — does NOT re-run the heavy model.

Sections:
  1. Ingredient quality breakdown (matchable vs context / noise / additive / non-English)
  2. Confidence distribution of matching results
  3. Spot-check of key test cases (apple juice, yogurt, sugar, vegetable oil, …)
  4. Sample of HIGH / MEDIUM / NO_MATCH results for manual review
  5. CSV export for manual annotation
"""

from contextlib import redirect_stdout
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"

OUTPUT_DIR = PROJECT_ROOT / "src" / "test" / "output"
OUTPUT_TXT = OUTPUT_DIR / "matching_inspection_output"
OUTPUT_CSV = OUTPUT_DIR / "matching_review_sample.csv"

# Ingredients that exercise known tricky cases mentioned in the improvement plan
SPOT_CHECK_INGREDIENTS = [
    "apple juice",
    "yogurt",
    "yoghurt",
    "sugar",
    "vegetable oil",
    "potato starch",
    "salt",
    "potato",
    "icing sugar",
    "burrata cheese",
    "semi skimmed milk",
    "corn syrup",
    "vitamin c",
    "e330",
    "flavouring",
    "vegetable",
    "root vegetable",
    "added sugar",
]


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _table_cols(con: duckdb.DuckDBPyConnection, table: str) -> set:
    """Return the set of column names for a table, or empty set if it doesn't exist."""
    try:
        return {
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [table],
            ).fetchall()
        }
    except Exception:
        return set()


# ── 1. Ingredient quality breakdown ──────────────────────────────────────────

def show_ingredient_quality(con: duckdb.DuckDBPyConnection) -> None:
    header("1. INGREDIENT QUALITY BREAKDOWN — products_ingredients")

    try:
        total = con.execute("SELECT COUNT(*) FROM products_ingredients").fetchone()[0]
    except Exception as e:
        print(f"  products_ingredients table not found: {e}")
        return

    rows = con.execute("""
        SELECT
            SUM(CASE WHEN matchable THEN 1 ELSE 0 END)          AS matchable,
            SUM(CASE WHEN is_context_term THEN 1 ELSE 0 END)    AS context_term,
            SUM(CASE WHEN is_noise_phrase THEN 1 ELSE 0 END)    AS noise_phrase,
            SUM(CASE WHEN is_additive THEN 1 ELSE 0 END)        AS additive,
            SUM(CASE WHEN tag_lang != 'en' THEN 1 ELSE 0 END)   AS non_english,
            COUNT(DISTINCT ingredient)                           AS unique_ingredients,
            COUNT(DISTINCT CASE WHEN matchable THEN ingredient END) AS unique_matchable
        FROM products_ingredients
    """).fetchone()

    matchable, context, noise, additive, non_en, unique, unique_m = rows

    print(f"\n  Total rows                : {total:>12,}")
    print(f"  Matchable                 : {matchable:>12,}  ({100*matchable/total:.1f}%)")
    print(f"  Context terms (excluded)  : {context:>12,}  ({100*context/total:.1f}%)")
    print(f"  Noise phrases (excluded)  : {noise:>12,}  ({100*noise/total:.1f}%)")
    print(f"  Additives (E-codes)       : {additive:>12,}  ({100*additive/total:.1f}%)")
    print(f"  Non-English tags          : {non_en:>12,}  ({100*non_en/total:.1f}%)")
    print(f"  Unique ingredients        : {unique:>12,}")
    print(f"  Unique matchable          : {unique_m:>12,}")

    # Sample context terms
    ctx_sample = con.execute("""
        SELECT DISTINCT ingredient FROM products_ingredients
        WHERE is_context_term ORDER BY ingredient LIMIT 20
    """).df()
    print(f"\n  Sample context terms: {ctx_sample['ingredient'].tolist()}")

    # Sample noise phrases
    noise_sample = con.execute("""
        SELECT DISTINCT ingredient FROM products_ingredients
        WHERE is_noise_phrase ORDER BY ingredient LIMIT 15
    """).df()
    print(f"\n  Sample noise phrases: {noise_sample['ingredient'].tolist()}")


# ── 2. Confidence distribution ────────────────────────────────────────────────

def show_confidence_distribution(con: duckdb.DuckDBPyConnection) -> None:
    header("2. CONFIDENCE DISTRIBUTION — matching_results_ingredients")

    cols = _table_cols(con, "matching_results_ingredients")
    if not cols:
        print("  matching_results_ingredients not found — run src/matching/run_matching.py first")
        return

    path_expr = "ROUND(AVG(path_score), 4) AS avg_path_score," if "path_score" in cols else ""
    try:
        dist = con.execute(f"""
            SELECT confidence, COUNT(*) AS n,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
                   {path_expr}
                   ROUND(AVG(final_level), 2) AS avg_final_level
            FROM matching_results_ingredients
            GROUP BY confidence
            ORDER BY n DESC
        """).fetchdf()
    except Exception as e:
        print(f"  Error: {e}")
        return

    print()
    print(dist.to_string(index=False))

    # final_level distribution
    level_dist = con.execute("""
        SELECT final_level, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM matching_results_ingredients
        GROUP BY final_level ORDER BY final_level
    """).fetchdf()
    print("\n  final_level distribution:")
    print(level_dist.to_string(index=False))


# ── 3. Spot-checks ────────────────────────────────────────────────────────────

def show_spot_checks(con: duckdb.DuckDBPyConnection) -> None:
    header("3. SPOT-CHECK — key test ingredients")

    cols = _table_cols(con, "matching_results_ingredients")
    if not cols:
        print("  matching_results_ingredients not found — run src/matching/run_matching.py first")
        return

    has_path = "path_score" in cols

    for ing in SPOT_CHECK_INGREDIENTS:
        rows = con.execute(
            "SELECT * FROM matching_results_ingredients WHERE ingredient = ?", [ing]
        ).fetchdf()

        if rows.empty:
            print(f"\n  [{ing}]  — NOT IN matching_results_ingredients")
            continue

        r = rows.iloc[0]
        print(f"\n  [{ing}]")
        print(f"    confidence   : {r['confidence']}")
        print(f"    final_level  : {r['final_level']}")
        if has_path:
            print(f"    path_score   : {r['path_score']:.4f}")
        print(f"    L2  : {r.get('level2_name', '?')}  ({r.get('level2_score', 0):.4f})")
        print(f"    L3  : {r.get('level3_name', '?')}  ({r.get('level3_score', 0):.4f})")
        print(f"    spec: {r.get('specific_name', '?')}  ({r.get('specific_score', 0):.4f})")
        print(f"    top1: {r.get('top1_name', '?')}  ({r.get('top1_score', 0):.4f})")
        print(f"    top2: {r.get('top2_name', '?')}  ({r.get('top2_score', 0):.4f})")
        print(f"    top3: {r.get('top3_name', '?')}  ({r.get('top3_score', 0):.4f})")


# ── 4. Sample for manual review ───────────────────────────────────────────────

def show_review_sample(con: duckdb.DuckDBPyConnection) -> None:
    header("4. REVIEW SAMPLE — 30 rows per confidence level")

    avail = _table_cols(con, "matching_results_ingredients")
    if not avail:
        print("  matching_results_ingredients not found")
        return

    order_col = "path_score" if "path_score" in avail else "top1_score"
    try:
        sample = con.execute(f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY confidence ORDER BY RANDOM()) AS rn
                FROM matching_results_ingredients
            )
            SELECT * EXCLUDE rn FROM ranked WHERE rn <= 30
            ORDER BY confidence, {order_col} DESC
        """).fetchdf()
    except Exception as e:
        print(f"  Error: {e}")
        return

    pd.set_option("display.max_colwidth", 45)
    want = ["ingredient", "confidence", "final_level", "path_score",
            "level2_name", "level3_name", "specific_name"]
    print_cols = [c for c in want if c in sample.columns]
    print(sample[print_cols].to_string(index=False))


# ── 5. CSV export ─────────────────────────────────────────────────────────────

def export_review_csv(con: duckdb.DuckDBPyConnection) -> None:
    header("5. EXPORTING REVIEW CSV")

    avail = _table_cols(con, "matching_results_ingredients")
    if not avail:
        print("  matching_results_ingredients not found")
        return

    wanted = [
        "ingredient", "confidence", "final_level", "path_score",
        "level2_code", "level2_name", "level2_score",
        "level3_code", "level3_name", "level3_score",
        "specific_code", "specific_name", "specific_score",
        "top1_code", "top1_name", "top1_score",
        "top2_code", "top2_name", "top2_score",
        "top3_code", "top3_name", "top3_score",
    ]
    select_cols = ", ".join(c for c in wanted if c in avail)

    try:
        df = con.execute(f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY confidence ORDER BY RANDOM()) AS rn
                FROM matching_results_ingredients
            )
            SELECT {select_cols}
            FROM ranked WHERE rn <= 50
            ORDER BY confidence, ingredient
        """).fetchdf()
    except Exception as e:
        print(f"  Error: {e}")
        return

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Exported {len(df):,} rows to {OUTPUT_CSV}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_inspection() -> None:
    print(f"DB_PATH      : {DB_PATH}")
    print(f"OUTPUT_TXT   : {OUTPUT_TXT}")
    print(f"OUTPUT_CSV   : {OUTPUT_CSV}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        show_ingredient_quality(con)
        show_confidence_distribution(con)
        show_spot_checks(con)
        show_review_sample(con)
        export_review_csv(con)
    finally:
        con.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_inspection()

    print(f"Inspection exported to: {OUTPUT_TXT}")
    print(f"Review CSV exported to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
