from __future__ import annotations

from typing import List, Tuple

from .models import Confidence, IngredientMatch, ReviewFlag, ReviewStatus


def classify(
    confidence_score: float,
    ingredients: List[IngredientMatch],
) -> Tuple[ReviewStatus, List[ReviewFlag], str]:
    """
    Decide review status for a product.

    Returns (ReviewStatus, product_level_flags, human_readable_reason).
    """
    n = len(ingredients)
    if n == 0:
        return (
            ReviewStatus.MANUAL_REQUIRED,
            [ReviewFlag.UNKNOWN_INGREDIENT],
            "No se han proporcionado ingredientes para analizar.",
        )

    no_match_count = sum(1 for m in ingredients if m.confidence == Confidence.NO_MATCH)
    high_count     = sum(1 for m in ingredients if m.confidence == Confidence.HIGH)
    no_match_ratio = no_match_count / n
    high_ratio     = high_count / n

    # Aggregate ingredient-level flags (dedup, order-preserving)
    seen: set = set()
    product_flags: List[ReviewFlag] = []
    for m in ingredients:
        for f in m.review_flags:
            if f not in seen:
                seen.add(f)
                product_flags.append(f)

    # ── MANUAL_REQUIRED ───────────────────────────────────────────────────────
    if confidence_score < 0.50 or no_match_ratio >= 0.30:
        parts = []
        if confidence_score < 0.50:
            parts.append(f"confianza {confidence_score:.2f} inferior a 0.50")
        if no_match_ratio >= 0.30:
            parts.append(
                f"{no_match_count}/{n} ingredientes sin match "
                f"({no_match_ratio:.0%} ≥ 30%)"
            )
        reason = "Revisión manual requerida: " + "; ".join(parts) + "."
        return ReviewStatus.MANUAL_REQUIRED, product_flags, reason

    # ── AUTO_APPROVED ─────────────────────────────────────────────────────────
    critical_flags = {
        ReviewFlag.ALLERGEN_DISCREPANCY,
        ReviewFlag.AMBIGUOUS_INGREDIENT,
    }
    has_critical = bool(critical_flags & set(product_flags))

    if (
        confidence_score >= 0.80
        and no_match_count == 0
        and high_ratio >= 0.70
        and not has_critical
    ):
        reason = (
            f"Auto-aprobado: confianza {confidence_score:.2f}, "
            f"{high_count}/{n} ingredientes con match HIGH."
        )
        return ReviewStatus.AUTO_APPROVED, product_flags, reason

    # ── NEEDS_REVIEW (everything else) ────────────────────────────────────────
    parts = []
    if confidence_score < 0.80:
        parts.append(f"confianza {confidence_score:.2f} inferior a 0.80")
    if ReviewFlag.MULTIPLE_CANDIDATES in product_flags:
        parts.append("candidatos ambiguos (diferencia de score < 0.05)")
    if ReviewFlag.ALLERGEN_DISCREPANCY in product_flags:
        parts.append("alérgenos detectados no declarados")
    if ReviewFlag.AMBIGUOUS_INGREDIENT in product_flags:
        parts.append("ingrediente(s) ambiguo(s) detectado(s)")
    if ReviewFlag.LOW_CONFIDENCE in product_flags:
        parts.append("match(es) de baja confianza")
    if not parts:
        parts.append("no cumple criterios de auto-aprobación")
    reason = "Necesita revisión: " + "; ".join(parts) + "."
    return ReviewStatus.NEEDS_REVIEW, product_flags, reason
