from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Inject project root so src.matching is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.matching.stratified_matcher import (
    build_hierarchy_index,
    generate_hierarchy_embeddings,
)

from .models import Confidence, FoodEx2Candidate, IngredientMatch, ReviewFlag

# ── Paths ─────────────────────────────────────────────────────────────────────
_FOODEX2_TERMS_PATH = _PROJECT_ROOT / "data" / "foodex2_terms.parquet"
_RESULTS_PATH = _PROJECT_ROOT / "data" / "matching_results_ingredients_baai__bge_small_en_v1_5.parquet"
# Thresholds from run_matching.py
_THR_L2 = 0.55
_THR_L3 = 0.7
_THR_SP = 0.85

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_ingredient(text: str) -> str:
    """Normalise ingredient text to maximise pre-computed cache hits."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("-", " ").replace("_", " ")
    text = text.lower().strip()
    return text


# ── Helpers to convert match data into IngredientMatch ───────────────────────

def _confidence_from_str(conf_str: str) -> Confidence:
    try:
        return Confidence(conf_str)
    except ValueError:
        return Confidence.NO_MATCH


def _build_flags(
    confidence: Confidence,
    top1_score: float,
    top2_score: Optional[float],
) -> List[ReviewFlag]:
    flags: List[ReviewFlag] = []
    if confidence == Confidence.NO_MATCH:
        flags.append(ReviewFlag.UNKNOWN_INGREDIENT)
    elif confidence == Confidence.LOW:
        flags.append(ReviewFlag.LOW_CONFIDENCE)
    if (
        top2_score is not None
        and top2_score > 0.0
        and (top1_score - top2_score) < 0.05
    ):
        flags.append(ReviewFlag.MULTIPLE_CANDIDATES)
    return flags


def _reasoning(confidence: Confidence, original: str, name: Optional[str], score: float, level: int) -> str:
    if confidence == Confidence.NO_MATCH:
        return f"Sin match FoodEx2 para '{original}' (score máximo: {score:.3f})."
    level_label = {2: "L2 (categoría)", 3: "L3 (subcategoría)", 5: "L5 (específico)"}.get(level, f"L{level}")
    return (
        f"{confidence.value} ({level_label}): "
        f"'{original}' → {name} (score: {score:.3f})"
    )


# ── Main class ────────────────────────────────────────────────────────────────

class IngredientMatcher:
    """
    Matches ingredient text to FoodEx2 codes using stratified semantic matching.

    Strategy:
    1. Normalise the ingredient text.
    2. Check the pre-computed results cache (exact, then lowercase-fallback).
    3. On cache miss: encode with sentence-transformers and run the stratified
       L2→L3→specific cascade against pre-built hierarchy embeddings.
    """

    def __init__(self) -> None:
        self._load_precomputed_cache()
        self._load_hierarchy_and_model()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _load_precomputed_cache(self) -> None:
        df = pd.read_parquet(_RESULTS_PATH)
        self._cache: Dict[str, dict] = {}
        self._cache_lower: Dict[str, str] = {}
        for row in df.itertuples(index=False):
            key = row.ingredient
            self._cache[key] = row._asdict()
            lk = key.lower()
            if lk not in self._cache_lower:
                self._cache_lower[lk] = key

    def _load_hierarchy_and_model(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(MODEL_NAME)
        terms = pd.read_parquet(_FOODEX2_TERMS_PATH)
        self.hierarchy = build_hierarchy_index(terms)
        self.hierarchy = generate_hierarchy_embeddings(self.hierarchy, self.model)

    # ── Cache lookup ──────────────────────────────────────────────────────────

    def _cache_lookup(self, normalized: str) -> Optional[dict]:
        if normalized in self._cache:
            return self._cache[normalized]
        original_key = self._cache_lower.get(normalized.lower())
        if original_key:
            return self._cache[original_key]
        return None

    # ── Convert a cache row dict → IngredientMatch ────────────────────────────

    def _row_to_match(self, row: dict, original: str, normalized: str) -> IngredientMatch:
        confidence  = _confidence_from_str(row.get("confidence", "NO_MATCH"))
        top1_code   = row.get("top1_code")
        top1_name   = row.get("top1_name")
        top1_score  = float(row.get("top1_score") or 0.0)
        top2_code   = row.get("top2_code")
        top2_name   = row.get("top2_name")
        top2_score  = float(row.get("top2_score") or 0.0) if row.get("top2_code") else None
        top3_code   = row.get("top3_code")
        top3_name   = row.get("top3_name")
        top3_score  = float(row.get("top3_score") or 0.0) if row.get("top3_code") else None
        final_level = int(row.get("final_level") or 0)

        # path_score: combined RF+BGE+lexical quality signal; fall back to top1_score
        # for parquets generated before this field was added.
        raw_path   = row.get("path_score")
        path_score = float(raw_path) if raw_path is not None else top1_score

        # Diagnostic fields present only in parquets produced by the latest script.
        raw_mode    = row.get("matching_mode")
        raw_rf      = row.get("rf_score")
        raw_variant = row.get("rf_matched_variant")
        raw_pen     = row.get("branch_penalty")
        raw_exact   = row.get("exact_variant_match")
        matching_mode       = str(raw_mode)    if raw_mode    is not None else None
        rf_score            = float(raw_rf)    if raw_rf      is not None else None
        rf_matched_variant  = str(raw_variant) if raw_variant is not None else None
        branch_penalty      = float(raw_pen)   if raw_pen     is not None else None
        exact_variant_match = bool(raw_exact)  if raw_exact   is not None else None

        alternatives: List[FoodEx2Candidate] = []
        if top2_code is not None:
            alternatives.append(FoodEx2Candidate(
                code=top2_code, name=top2_name or "", score=top2_score or 0.0, level=final_level
            ))
        if top3_code is not None:
            alternatives.append(FoodEx2Candidate(
                code=top3_code, name=top3_name or "", score=top3_score or 0.0, level=final_level
            ))

        flags = _build_flags(confidence, top1_score, top2_score)

        return IngredientMatch(
            original_text=original,
            normalized=normalized,
            confidence=confidence,
            final_code=top1_code if top1_code else None,
            final_name=top1_name if top1_name else None,
            final_score=path_score,
            final_level=final_level,
            alternatives=alternatives,
            allergens=[],
            reasoning=_reasoning(confidence, original, top1_name, path_score, final_level),
            needs_review=confidence in (Confidence.LOW, Confidence.NO_MATCH),
            review_flags=flags,
            path_score=path_score,
            matching_mode=matching_mode,
            rf_score=rf_score,
            rf_matched_variant=rf_matched_variant,
            branch_penalty=branch_penalty,
            exact_variant_match=exact_variant_match,
        )

    # ── On-the-fly matching ────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # shape (1, 768)

    def _match_vector(self, vec: np.ndarray, original: str, normalized: str) -> IngredientMatch:
        """
        Run the stratified L2 → L3 → specific cascade with a pre-encoded vector.
        Mirrors StratifiedMatcherOptimized.match_single but takes vec directly.
        """
        h = self.hierarchy
        l2 = h["level_2"]
        l3 = h["level_3"]
        lsp = h["level_specific"]
        children_map = h["children_map"]

        # ── Level 2 ───────────────────────────────────────────────────────────
        sim_l2 = (vec @ l2["embeddings"].T)[0]  # (N_l2,)
        top3_l2_idx = np.argsort(sim_l2)[-3:][::-1]
        t2_list = [
            (l2["codes"][i], l2["names"][i], float(sim_l2[i])) for i in top3_l2_idx
        ]
        top_code_l2, top_name_l2, top_score_l2 = t2_list[0]

        if top_score_l2 < _THR_L2:
            return self._build_no_match(original, normalized, top_score_l2, t2_list, level=2)

        # ── Level 3 ───────────────────────────────────────────────────────────
        children_l3 = children_map.get(top_code_l2, [])
        children_l3_idx = [
            l3["code_to_idx"][c] for c in children_l3 if c in l3["code_to_idx"]
        ]
        if not children_l3_idx:
            return self._build_match(
                original, normalized, Confidence.LOW, top_code_l2, top_name_l2,
                top_score_l2, level=2, t_list=t2_list,
            )

        sim_l3 = (vec @ l3["embeddings"][children_l3_idx].T)[0]
        top3_l3_local = np.argsort(sim_l3)[-3:][::-1]
        t3_list = [
            (
                l3["codes"][children_l3_idx[i]],
                l3["names"][children_l3_idx[i]],
                float(sim_l3[i]),
            )
            for i in top3_l3_local
        ]
        top_code_l3, top_name_l3, top_score_l3 = t3_list[0]

        if top_score_l3 < _THR_L3:
            return self._build_match(
                original, normalized, Confidence.LOW, top_code_l2, top_name_l2,
                top_score_l2, level=2, t_list=t2_list,
            )

        # ── Specific level ────────────────────────────────────────────────────
        children_sp = children_map.get(top_code_l3, [])
        children_sp_idx = [
            lsp["code_to_idx"][c] for c in children_sp if c in lsp["code_to_idx"]
        ]
        if not children_sp_idx:
            return self._build_match(
                original, normalized, Confidence.MEDIUM, top_code_l3, top_name_l3,
                top_score_l3, level=3, t_list=t3_list,
            )

        sim_sp = (vec @ lsp["embeddings"][children_sp_idx].T)[0]
        top3_sp_local = np.argsort(sim_sp)[-3:][::-1]
        tsp_list = [
            (
                lsp["codes"][children_sp_idx[i]],
                lsp["names"][children_sp_idx[i]],
                float(sim_sp[i]),
            )
            for i in top3_sp_local
        ]
        top_code_sp, top_name_sp, top_score_sp = tsp_list[0]

        if top_score_sp >= _THR_SP:
            return self._build_match(
                original, normalized, Confidence.HIGH, top_code_sp, top_name_sp,
                top_score_sp, level=5, t_list=tsp_list,
            )

        return self._build_match(
            original, normalized, Confidence.MEDIUM, top_code_l3, top_name_l3,
            top_score_l3, level=3, t_list=t3_list,
        )

    def _build_no_match(
        self,
        original: str,
        normalized: str,
        top_score: float,
        t_list: list,
        level: int,
    ) -> IngredientMatch:
        alternatives = [
            FoodEx2Candidate(code=c, name=n, score=s, level=level)
            for c, n, s in t_list[1:]
            if c is not None
        ]
        return IngredientMatch(
            original_text=original,
            normalized=normalized,
            confidence=Confidence.NO_MATCH,
            final_code=None,
            final_name=None,
            final_score=top_score,
            final_level=0,
            alternatives=alternatives,
            allergens=[],
            reasoning=_reasoning(Confidence.NO_MATCH, original, None, top_score, level),
            needs_review=True,
            review_flags=[ReviewFlag.UNKNOWN_INGREDIENT],
            path_score=top_score,
            matching_mode="beam",
        )

    def _build_match(
        self,
        original: str,
        normalized: str,
        confidence: Confidence,
        code: str,
        name: str,
        score: float,
        level: int,
        t_list: list,
    ) -> IngredientMatch:
        alternatives = [
            FoodEx2Candidate(code=c, name=n, score=s, level=level)
            for c, n, s in t_list[1:]
            if c is not None
        ]
        top2_score = t_list[1][2] if len(t_list) > 1 and t_list[1][0] else None
        flags = _build_flags(confidence, score, top2_score)
        return IngredientMatch(
            original_text=original,
            normalized=normalized,
            confidence=confidence,
            final_code=code,
            final_name=name,
            final_score=score,
            final_level=level,
            alternatives=alternatives,
            allergens=[],
            reasoning=_reasoning(confidence, original, name, score, level),
            needs_review=confidence in (Confidence.LOW, Confidence.NO_MATCH),
            review_flags=flags,
            path_score=score,
            matching_mode="beam",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def match(self, original_text: str) -> IngredientMatch:
        normalized = normalize_ingredient(original_text)
        cached = self._cache_lookup(normalized)
        if cached is not None:
            result = self._row_to_match(cached, original_text, normalized)
            # RF-mode parquet rows (rf_exact / rf_reranked) only store the
            # top-1 winner; top2/top3 are null so alternatives is empty.
            # Run a beam search to populate alternatives whenever the match is
            # not HIGH-confidence — that is exactly when alternatives matter.
            if not result.alternatives and result.confidence != Confidence.HIGH:
                vec = self._encode(normalized)
                live = self._match_vector(vec, original_text, normalized)
                if live.alternatives:
                    result.alternatives = live.alternatives
                else:
                    # Thin FoodEx2 node: no siblings at the matched level.
                    # Fall back to the top-3 L2 category candidates so the
                    # expander always has something to show for non-HIGH matches.
                    l2 = self.hierarchy["level_2"]
                    sim_l2 = (vec @ l2["embeddings"].T)[0]
                    top_idx = np.argsort(sim_l2)[-4:][::-1]
                    result.alternatives = [
                        FoodEx2Candidate(
                            code=l2["codes"][i],
                            name=l2["names"][i],
                            score=float(sim_l2[i]),
                            level=2,
                        )
                        for i in top_idx
                        if l2["codes"][i] != result.final_code
                    ][:3]
            return result
        vec = self._encode(normalized)
        return self._match_vector(vec, original_text, normalized)

    def match_batch(self, texts: List[str]) -> List[IngredientMatch]:
        return [self.match(t) for t in texts]
