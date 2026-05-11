from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Set

# Inject project root so allergenscan package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from allergenscan.allergenscan.detector import AllergenDetector

from .confidence_scorer import score_product
from .ingredient_matcher import IngredientMatcher
from .models import (
    Confidence,
    IngredientMatch,
    ProductAnalysis,
    ReviewFlag,
    ReviewStatus,
)
from .review_classifier import classify

_ALLERGEN_MAP_PATH = _PROJECT_ROOT / "allergenscan" / "data" / "foodex2_to_allergen.csv"
_AMBIGUOUS_PATH    = _PROJECT_ROOT / "allergenscan" / "data" / "ambiguous_ingredients.csv"


class ProductAnalyzer:
    """Orchestrates FoodEx2 coding, allergen enrichment, scoring, and review classification."""

    def __init__(self, matcher: IngredientMatcher) -> None:
        self.matcher = matcher
        self.allergen_detector = AllergenDetector(
            allergen_mapping_path=str(_ALLERGEN_MAP_PATH),
            ambiguous_ingredients_path=str(_AMBIGUOUS_PATH),
            min_confidence="MEDIUM",
        )

    # ── Ingredient parsing ────────────────────────────────────────────────────

    def parse_ingredient_list(self, raw_text: str) -> List[str]:
        """Parse raw ingredients_text using the same ETL cleaning pipeline."""
        from src.etl.cleaning import parse_ingredients_text_raw
        return parse_ingredients_text_raw(raw_text)

    # ── Allergen enrichment ───────────────────────────────────────────────────

    def _enrich_allergens(self, match: IngredientMatch) -> IngredientMatch:
        """
        Attach allergen data to an IngredientMatch from two sources:
        1. FoodEx2 code → allergen mapping (for matched ingredients).
        2. Ambiguous pattern check (for all ingredients regardless of match).
        """
        allergens: list = []

        if match.final_code and match.confidence not in (Confidence.NO_MATCH,):
            allergens.extend(
                self.allergen_detector.get_allergens_from_foodex2(
                    match.final_code, match.final_name or ""
                )
            )

        # Ambiguous pattern check on original text
        ambiguous = self.allergen_detector.check_ambiguous_ingredient(match.original_text)
        if ambiguous:
            for allergen, certainty, _pattern in ambiguous:
                if (allergen, certainty) not in allergens:
                    allergens.append((allergen, certainty))
            if ReviewFlag.AMBIGUOUS_INGREDIENT not in match.review_flags:
                match.review_flags.append(ReviewFlag.AMBIGUOUS_INGREDIENT)

        match.allergens = allergens
        return match

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze(
        self,
        product_code: str,
        product_name: str,
        ingredients_text: str,
        declared_allergens: Optional[Set[str]] = None,
        ingredient_tokens: Optional[List[str]] = None,
    ) -> ProductAnalysis:
        """
        Full analysis pipeline for a single product.

        Args:
            product_code: Barcode or identifier string.
            product_name: Human-readable product name.
            ingredients_text: Raw ingredient list (comma or newline separated).
                Used only when ingredient_tokens is None.
            declared_allergens: Set of allergen codes already declared on the label.
            ingredient_tokens: Pre-cleaned matchable ingredient strings from the
                products_ingredients parquet. When provided, skips parse_ingredient_list
                so that live analysis uses the same token set as the precomputed table.
        """
        declared_allergens = declared_allergens or set()
        ingredient_texts = (
            ingredient_tokens
            if ingredient_tokens is not None
            else self.parse_ingredient_list(ingredients_text)
        )

        if not ingredient_texts:
            return ProductAnalysis(
                product_code=product_code,
                product_name=product_name,
                ingredients=[],
                confidence_score=0.0,
                review_status=ReviewStatus.MANUAL_REQUIRED,
                review_flags=[ReviewFlag.UNKNOWN_INGREDIENT],
                review_reason="No se han proporcionado ingredientes.",
            )

        # 1. Match all ingredients against FoodEx2 hierarchy
        matches: List[IngredientMatch] = self.matcher.match_batch(ingredient_texts)

        # 2. Enrich each match with allergen data
        matches = [self._enrich_allergens(m) for m in matches]

        # 3. Check allergen discrepancies vs declared list
        if declared_allergens:
            for m in matches:
                detected_codes = {a[0] for a in m.allergens}
                undeclared = detected_codes - declared_allergens
                if undeclared and ReviewFlag.ALLERGEN_DISCREPANCY not in m.review_flags:
                    m.review_flags.append(ReviewFlag.ALLERGEN_DISCREPANCY)

        # 4. Product-level confidence score
        confidence_score = score_product(matches)

        # 5. Review classification
        status, product_flags, reason = classify(confidence_score, matches)

        # Propagate ALLERGEN_DISCREPANCY to product level if any ingredient has it
        if any(ReviewFlag.ALLERGEN_DISCREPANCY in m.review_flags for m in matches):
            if ReviewFlag.ALLERGEN_DISCREPANCY not in product_flags:
                product_flags.append(ReviewFlag.ALLERGEN_DISCREPANCY)

        return ProductAnalysis(
            product_code=product_code,
            product_name=product_name,
            ingredients=matches,
            confidence_score=confidence_score,
            review_status=status,
            review_flags=product_flags,
            review_reason=reason,
        )
