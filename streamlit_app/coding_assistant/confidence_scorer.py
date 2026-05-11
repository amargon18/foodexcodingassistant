from __future__ import annotations

from typing import List

from .models import Confidence, IngredientMatch


def score_product(ingredients: List[IngredientMatch]) -> float:
    """
    Compute product-level confidence score (0.0–1.0).

    Formula:
        0.4 * avg_match_score
      + 0.3 * high_confidence_ratio
      + 0.2 * coverage_ratio
      + 0.1 * medium_confidence_ratio
    """
    if not ingredients:
        return 0.0

    n = len(ingredients)
    avg_score    = sum(m.final_score for m in ingredients) / n
    high_ratio   = sum(1 for m in ingredients if m.confidence == Confidence.HIGH) / n
    medium_ratio = sum(1 for m in ingredients if m.confidence == Confidence.MEDIUM) / n
    coverage     = sum(1 for m in ingredients if m.confidence != Confidence.NO_MATCH) / n

    return 0.4 * avg_score + 0.3 * high_ratio + 0.2 * coverage + 0.1 * medium_ratio


def score_breakdown(ingredients: List[IngredientMatch]) -> str:
    """Return human-readable breakdown of the confidence score."""
    if not ingredients:
        return "Sin ingredientes."
    n = len(ingredients)
    avg_score    = sum(m.final_score for m in ingredients) / n
    high_ratio   = sum(1 for m in ingredients if m.confidence == Confidence.HIGH) / n
    medium_ratio = sum(1 for m in ingredients if m.confidence == Confidence.MEDIUM) / n
    coverage     = sum(1 for m in ingredients if m.confidence != Confidence.NO_MATCH) / n
    total = 0.4 * avg_score + 0.3 * high_ratio + 0.2 * coverage + 0.1 * medium_ratio
    return (
        f"Score {total:.2f} = "
        f"0.4×score_medio({avg_score:.2f}) + "
        f"0.3×ratio_HIGH({high_ratio:.2f}) + "
        f"0.2×cobertura({coverage:.2f}) + "
        f"0.1×ratio_MEDIUM({medium_ratio:.2f})"
    )
