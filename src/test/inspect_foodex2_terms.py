from pathlib import Path
from contextlib import redirect_stdout

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HIERARCHY_PATH = PROJECT_ROOT / "matching" / "foodHierarchy_nlp_extended.csv"

OUTPUT_DIR = PROJECT_ROOT / "src" / "test" / "output"
OUTPUT_TXT = OUTPUT_DIR / "foodex2_test_output"


DB_PATH = PROJECT_ROOT / "openfoodfacts.duckdb"
OUTPUT_PARQUET = OUTPUT_DIR / "foodex2_terms.parquet"
OUTPUT_TABLE = "foodex2_terms"


def show_basic_info(terms: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("BASIC INFO")
    print("=" * 100)

    print(f"\nShape: {terms.shape}")

    print("\nColumns:")
    for col in terms.columns:
        print(f"  - {col}")

    print("\nDtypes:")
    print(terms.dtypes.to_string())

    print("\nMissing values:")
    missing = terms.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())


def show_level_distribution(terms: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("REPORT HIERARCHY LEVEL DISTRIBUTION")
    print("=" * 100)

    print(
        terms["reportHierarchyLevel"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


def show_hierarchy_examples(terms: pd.DataFrame, n: int = 30) -> None:
    print("\n" + "=" * 100)
    print("HIGH-LEVEL HIERARCHY EXAMPLES")
    print("=" * 100)

    high_level = terms[terms["reportHierarchyLevel"] <= 2].copy()

    if high_level.empty:
        print("\nNo high-level hierarchy terms found.")
        return

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
    ]

    print(high_level.head(n)[cols].to_string(index=False))


def show_sample_terms(terms: pd.DataFrame, n: int = 30) -> None:
    print("\n" + "=" * 100)
    print(f"RANDOM SAMPLE OF {n} TERMS")
    print("=" * 100)

    if terms.empty:
        print("\nNo terms available.")
        return

    sample = terms.sample(min(n, len(terms)), random_state=None)

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
    ]

    print(sample[cols].to_string(index=False))


def show_short_term_names(terms: pd.DataFrame, n: int = 30) -> None:
    print("\n" + "=" * 100)
    print("SHORTEST TERM EXTENDED NAMES")
    print("=" * 100)

    temp = terms.copy()
    temp["name_len"] = temp["termExtendedName"].fillna("").astype(str).str.len()

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
        "name_len",
    ]

    print(temp.sort_values("name_len").head(n)[cols].to_string(index=False))


def show_long_term_names(terms: pd.DataFrame, n: int = 20) -> None:
    print("\n" + "=" * 100)
    print("LONGEST TERM EXTENDED NAMES")
    print("=" * 100)

    temp = terms.copy()
    temp["name_len"] = temp["termExtendedName"].fillna("").astype(str).str.len()

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
        "name_len",
    ]

    print(
        temp.sort_values("name_len", ascending=False)
        .head(n)[cols]
        .to_string(index=False)
    )


def show_duplicate_term_names(terms: pd.DataFrame, n: int = 30) -> None:
    print("\n" + "=" * 100)
    print("DUPLICATED TERM EXTENDED NAMES")
    print("=" * 100)

    duplicated = terms[
        terms["termExtendedName"].fillna("").astype(str).duplicated(keep=False)
    ].copy()

    if duplicated.empty:
        print("\nNo duplicated termExtendedName values found.")
        return

    duplicated["dup_count"] = duplicated.groupby("termExtendedName")[
        "termExtendedName"
    ].transform("count")

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
        "dup_count",
    ]

    print(
        duplicated.sort_values(
            ["dup_count", "termExtendedName"],
            ascending=[False, True],
        )
        .head(n)[cols]
        .to_string(index=False)
    )


def show_terms_by_keyword(terms: pd.DataFrame, keywords: list[str]) -> None:
    print("\n" + "=" * 100)
    print("KEYWORD CHECKS IN termExtendedName")
    print("=" * 100)

    searchable = terms["termExtendedName"].fillna("").astype(str).str.lower()

    cols = [
        "termCode",
        "termExtendedName",
        "reportHierarchyCode",
        "reportHierarchyLevel",
    ]

    for keyword in keywords:
        mask = searchable.str.contains(keyword.lower(), regex=False)
        subset = terms[mask]

        print("\n" + "-" * 100)
        print(f"Keyword: {keyword}")
        print(f"Matches: {len(subset):,}")

        if subset.empty:
            continue

        print(subset.head(20)[cols].to_string(index=False))


def prepare_terms() -> pd.DataFrame:
    if not HIERARCHY_PATH.exists():
        raise FileNotFoundError(f"FoodEx2 hierarchy file not found at {HIERARCHY_PATH}")

    terms = pd.read_csv(HIERARCHY_PATH)
    print(f"Loaded FoodEx2 hierarchy terms: {len(terms):,}")

    if "termExtendedName" not in terms.columns:
        raise ValueError("Column 'termExtendedName' not found in FoodEx2 hierarchy file.")

    if "termCode" not in terms.columns:
        raise ValueError("Column 'termCode' not found in FoodEx2 hierarchy file.")

    if "reportHierarchyCode" not in terms.columns:
        raise ValueError("Column 'reportHierarchyCode' not found in FoodEx2 hierarchy file.")

    if "reportHierarchyLevel" not in terms.columns:
        terms["reportHierarchyLevel"] = (
            terms["reportHierarchyCode"]
            .astype(str)
            .apply(lambda v: len(v.split(".")))
        )

    terms["termExtendedName"] = terms["termExtendedName"].fillna("").astype(str)

    return terms


def export_terms(terms: pd.DataFrame) -> None:
    selected = terms[
        [
            "termCode",
            "termExtendedName",
            "reportHierarchyCode",
            "reportHierarchyLevel",
        ]
    ].copy()

    selected.to_parquet(OUTPUT_PARQUET, index=False)

    con = duckdb.connect(str(DB_PATH))

    try:
        con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
        con.register("df_temp", selected)
        con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM df_temp")

        print(f"\nCreated DuckDB table '{OUTPUT_TABLE}' with {len(selected):,} rows")
        print(f"Saved parquet to: {OUTPUT_PARQUET}")

    finally:
        con.close()


def run_inspection() -> None:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"HIERARCHY_PATH: {HIERARCHY_PATH}")
    print(f"OUTPUT_TXT: {OUTPUT_TXT}")
    print(f"OUTPUT_PARQUET: {OUTPUT_PARQUET}")
    print(f"DB_PATH: {DB_PATH}")

    terms = prepare_terms()

    show_basic_info(terms)
    show_level_distribution(terms)
    show_hierarchy_examples(terms, n=30)
    show_sample_terms(terms, n=30)
    show_short_term_names(terms, n=30)
    show_long_term_names(terms, n=20)
    show_duplicate_term_names(terms, n=30)

    show_terms_by_keyword(
        terms,
        keywords=[
            "milk",
            "cheese",
            "yogurt",
            "sugar",
            "syrup",
            "oil",
            "potato",
            "tomato",
            "chicken",
            "beef",
            "fish",
            "flour",
            "additive",
            "flavour",
            "color",
            "colour",
        ],
    )

    export_terms(terms)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_inspection()

    print(f"FoodEx2 inspection exported to: {OUTPUT_TXT}")
    print(f"FoodEx2 parquet exported to: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()