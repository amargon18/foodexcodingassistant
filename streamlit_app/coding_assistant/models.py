from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NO_MATCH = "NO_MATCH"


class ReviewStatus(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    MANUAL_REQUIRED = "MANUAL_REQUIRED"


class ReviewFlag(str, Enum):
    AMBIGUOUS_INGREDIENT = "AMBIGUOUS_INGREDIENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    ALLERGEN_DISCREPANCY = "ALLERGEN_DISCREPANCY"
    UNKNOWN_INGREDIENT = "UNKNOWN_INGREDIENT"


@dataclass
class FoodEx2Candidate:
    code: str
    name: str
    score: float
    level: int  # 2, 3, or 5 (specific)


@dataclass
class IngredientMatch:
    original_text: str
    normalized: str
    confidence: Confidence
    final_code: Optional[str]
    final_name: Optional[str]
    final_score: float
    final_level: int                      # 0=no match, 2, 3, 5
    alternatives: List[FoodEx2Candidate]  # top-2 and top-3 candidates
    allergens: List[Tuple[str, str]]      # [(allergen_code, certainty_str)]
    reasoning: str
    needs_review: bool
    review_flags: List[ReviewFlag]
    # Diagnostic fields from the precomputed parquet.
    # path_score present since first bge-small run.
    # All others populated once the parquet is regenerated with RF diagnostics.
    path_score: Optional[float] = None
    matching_mode: Optional[str] = None       # "rf_exact" | "rf_reranked" | "beam"
    rf_score: Optional[float] = None          # RapidFuzz score 0-100
    rf_matched_variant: Optional[str] = None  # FoodEx2 variant string that triggered the RF hit
    branch_penalty: Optional[float] = None    # L2 BGE prior applied during RF reranking
    exact_variant_match: Optional[bool] = None


@dataclass
class ProductAnalysis:
    product_code: str
    product_name: str
    ingredients: List[IngredientMatch]
    confidence_score: float
    review_status: ReviewStatus
    review_flags: List[ReviewFlag]
    review_reason: str
    processed_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
