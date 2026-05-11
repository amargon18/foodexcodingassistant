import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.etl.build_foodex2_terms import main as build_foodex2_terms
from src.etl.build_products_ingredients import main as build_products_ingredients
from src.matching.run_matching import main as run_matching
from src.allergenscan.run_allergenscan import main as run_allergenscan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full OpenFoodFacts -> FoodEx2 matching pipeline."
    )
    parser.add_argument("--skip-products", action="store_true", help="Skip building products_ingredients")
    parser.add_argument("--skip-terms", action="store_true", help="Skip building foodex2_terms")
    parser.add_argument("--skip-matching", action="store_true", help="Skip running the matching step")
    parser.add_argument("--skip-allergenscan", action="store_true", help="Skip allergen detection step")
    args = parser.parse_args()

    if not args.skip_products:
        print("\n=== STEP 1: Build products_ingredients ===")
        build_products_ingredients()
    else:
        print("\n=== STEP 1 skipped ===")

    if not args.skip_terms:
        print("\n=== STEP 2: Build foodex2_terms ===")
        build_foodex2_terms()
    else:
        print("\n=== STEP 2 skipped ===")

    if not args.skip_matching:
        print("\n=== STEP 3: Run stratified matching ===")
        run_matching()
    else:
        print("\n=== STEP 3 skipped ===")

    if not args.skip_allergenscan:
        print("\n=== STEP 4: AllergenScan — Undeclared Allergen Detection ===")
        run_allergenscan()
    else:
        print("\n=== STEP 4 skipped ===")

    print("\nPipeline finished. Results are available in data/, allergenscan/data/allergenscan_results/, and openfoodfacts.duckdb.")


if __name__ == "__main__":
    main()
