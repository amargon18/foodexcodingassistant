from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIERARCHY_PATH = PROJECT_ROOT / "matching" / "foodHierarchy_nlp_extended.csv"
BACKUP_NLP_PATH = PROJECT_ROOT / "eda" / "foodex2_nlp.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_PARQUET = OUTPUT_DIR / "foodex2_terms.parquet"
OUTPUT_TABLE = "foodex2_terms"


def _build_nlp_text(df: pd.DataFrame) -> pd.DataFrame:
    if "nlp_text" in df.columns:
        return df

    texts = []
    for _, row in df.iterrows():
        parts = [str(row.get("termExtendedName", ""))]
        if pd.notna(row.get("commonNames")):
            parts.append(str(row["commonNames"]))
        if pd.notna(row.get("scientificNames")):
            parts.append(str(row["scientificNames"]))
        texts.append(". ".join([p.strip() for p in parts if p]).strip())
    df["nlp_text"] = texts
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HIERARCHY_PATH.exists():
        raise FileNotFoundError(f"FoodEx2 hierarchy file not found at {HIERARCHY_PATH}")

    terms = pd.read_csv(HIERARCHY_PATH)
    print(f"Loaded FoodEx2 hierarchy terms: {len(terms):,}")

    if "reportHierarchyLevel" not in terms.columns:
        terms["reportHierarchyLevel"] = terms["reportHierarchyCode"].astype(str).apply(lambda v: len(v.split(".")))

    if BACKUP_NLP_PATH.exists():
        nlp_df = pd.read_csv(BACKUP_NLP_PATH, usecols=["termCode", "nlp_text"] )
        terms = terms.merge(nlp_df, on="termCode", how="left")
        missing = terms["nlp_text"].isna().sum()
        if missing:
            print(f"Warning: {missing:,} FoodEx2 terms are missing nlp_text in backup file")
    else:
        terms = _build_nlp_text(terms)

    terms["nlp_text"] = terms["nlp_text"].fillna(terms["termExtendedName"]).astype(str)
    base_cols = ["termCode", "termExtendedName", "reportHierarchyCode", "reportHierarchyLevel", "nlp_text"]
    extra_cols = ["allFacets", "facets_resolved"]
    keep = base_cols + [c for c in extra_cols if c in terms.columns]
    selected = terms[keep].copy()

    selected.to_parquet(OUTPUT_PARQUET, index=False)

    con = duckdb.connect(str(PROJECT_ROOT / "openfoodfacts.duckdb"))
    try:
        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
        con.register("df_temp", selected)
        con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df_temp")
        print(f"Created DuckDB table '{OUTPUT_TABLE}' with {len(selected):,} rows")
    finally:
        con.close()


if __name__ == "__main__":
    main()
