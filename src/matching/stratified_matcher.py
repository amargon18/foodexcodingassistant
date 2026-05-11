"""
Stratified FoodEx2 matcher.

Changes vs. the original greedy implementation:
  - FoodEx2 termExtendedName variants: comma-order inversion + British/American spelling
    are generated before embedding so "Juice, apple" matches "apple juice".
  - Variant embeddings are averaged per term (encode_name_variants).
  - token_jaccard provides a lexical secondary signal combined with embedding similarity.
  - Beam search (top_k_l2 / top_k_l3 / top_k_specific) explores multiple hierarchy paths
    per ingredient instead of committing greedily to the top-1 branch at each level.
  - Path scoring (0.20 * L2 + 0.30 * L3 + 0.50 * spec_combined) selects the overall
    best path rather than falling through independently at each threshold.
  - final_level now reflects the actual reportHierarchyLevel of the matched term instead
    of the hardcoded value 5.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from rapidfuzz import fuzz, process as rf_process


# ── FoodEx2 name normalization and variant generation ─────────────────────────

_SPELLING_PAIRS = [
    ("yoghurt", "yogurt"),
    ("flavour", "flavor"),
    ("colour", "color"),
    ("skimmed", "skim"),
    ("sulphur", "sulfur"),
    ("fibre", "fiber"),
    ("litre", "liter"),
    ("centre", "center"),
]


def _nlk(text: str) -> str:
    """Inline normalize_lookup_key without importing cleaning to avoid circular deps."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_foodex_name(name: str) -> str:
    """Normalise a FoodEx2 termExtendedName for embedding and lookup."""
    name = _nlk(name)
    name = name.replace("(", " ").replace(")", " ")
    name = name.replace("/", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def spelling_variants(text: str) -> list:
    """Return all British/American spelling variants of a normalised name."""
    variants = {text}
    for brit, amer in _SPELLING_PAIRS:
        for v in list(variants):
            if brit in v:
                variants.add(v.replace(brit, amer))
            if amer in v:
                variants.add(v.replace(amer, brit))
    return list(variants)


def comma_order_variants(name: str) -> list:
    """Generate name variants by inverting comma-separated parts and adding spellings.

    'Juice, apple'  ->  {'juice apple', 'apple juice'}
    'Yoghurt, cow milk, flavoured'  ->  {'yoghurt cow milk flavoured', 'yogurt ...', ...}
    """
    base = normalize_foodex_name(name)
    variants = {base}

    if "," in name:
        parts = [normalize_foodex_name(p) for p in name.split(",")]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            # Natural English order: reverse all parts
            variants.add(" ".join(reversed(parts)))
            # Also try first two parts swapped (common case: "X, Y extra")
            variants.add(" ".join([parts[1], parts[0]] + parts[2:]))

    expanded: set = set()
    for v in variants:
        expanded.update(spelling_variants(v))

    return list(expanded)


def encode_name_variants(names: list, model) -> np.ndarray:
    """Encode a list of FoodEx2 names using variant-averaged embeddings.

    For each name, comma_order_variants() generates 1–N text strings.
    All variants are encoded at once, then averaged per term and re-normalised.
    This biases the term embedding toward the midpoint of all natural phrasings.
    """
    # Collect variants and the slice boundaries for each term
    all_texts: list = []
    slices: list = []
    for name in names:
        variants = comma_order_variants(name)
        start = len(all_texts)
        all_texts.extend(variants)
        slices.append((start, len(all_texts)))

    raw_embs = model.encode(
        all_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    final_embs = np.empty((len(names), raw_embs.shape[1]), dtype=np.float32)
    for idx, (s, e) in enumerate(slices):
        mean_emb = raw_embs[s:e].mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        final_embs[idx] = mean_emb / norm if norm > 0 else mean_emb

    return final_embs


# ── Short-ingredient detection ────────────────────────────────────────────────

def is_short_ingredient(ingredient: str) -> bool:
    """Return True if the ingredient has <= 2 normalised tokens."""
    return len(_nlk(ingredient).split()) <= 2


# ── Lexical similarity ────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two ingredient/term strings.

    Punctuation is stripped before tokenising so 'Juice, apple' and 'apple juice'
    share both tokens.
    """
    def _tokens(s: str) -> set:
        return set(re.sub(r"[^\w\s]", " ", _nlk(s)).split())

    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Hierarchy index ───────────────────────────────────────────────────────────

def build_hierarchy_index(foodex2_df) -> Dict[str, dict]:
    hierarchy: Dict[str, dict] = {}

    food_df = foodex2_df[
        foodex2_df["reportHierarchyCode"].astype(str).str.startswith("Z0001")
    ].copy()

    level_2 = food_df[food_df["reportHierarchyLevel"] == 2].copy()
    hierarchy["level_2"] = {
        "df": level_2,
        "codes": level_2["termCode"].tolist(),
        "names": level_2["termExtendedName"].tolist(),
        "hierarchy_codes": level_2["reportHierarchyCode"].tolist(),
        "levels": level_2["reportHierarchyLevel"].tolist(),
    }

    level_3 = food_df[food_df["reportHierarchyLevel"] == 3].copy()
    hierarchy["level_3"] = {
        "df": level_3,
        "codes": level_3["termCode"].tolist(),
        "names": level_3["termExtendedName"].tolist(),
        "hierarchy_codes": level_3["reportHierarchyCode"].tolist(),
        "levels": level_3["reportHierarchyLevel"].tolist(),
    }

    level_specific = food_df[food_df["reportHierarchyLevel"] >= 4].copy()
    hierarchy["level_specific"] = {
        "df": level_specific,
        "codes": level_specific["termCode"].tolist(),
        "names": level_specific["termExtendedName"].tolist(),
        "hierarchy_codes": level_specific["reportHierarchyCode"].tolist(),
        "levels": level_specific["reportHierarchyLevel"].tolist(),
    }

    # Vectorised parent map: O(N) dict lookups instead of O(N*M) DataFrame filters
    l2_hier_to_code = dict(zip(
        level_2["reportHierarchyCode"].astype(str), level_2["termCode"]
    ))
    l3_hier_to_code = dict(zip(
        level_3["reportHierarchyCode"].astype(str), level_3["termCode"]
    ))

    parent_map: dict = {}

    l3_parent_hier = level_3["reportHierarchyCode"].astype(str).str.split(".").str[:2].str.join(".")
    for child_code, parent_hier in zip(level_3["termCode"], l3_parent_hier):
        parent = l2_hier_to_code.get(parent_hier)
        if parent:
            parent_map[child_code] = parent

    spec_parent_hier = level_specific["reportHierarchyCode"].astype(str).str.split(".").str[:3].str.join(".")
    for child_code, parent_hier in zip(level_specific["termCode"], spec_parent_hier):
        parent = l3_hier_to_code.get(parent_hier)
        if parent:
            parent_map[child_code] = parent

    children_map: dict = {}
    for child, parent in parent_map.items():
        children_map.setdefault(parent, []).append(child)

    hierarchy["parent_map"] = parent_map
    hierarchy["children_map"] = children_map

    return hierarchy


def _resolve_path(
    level_name: str,
    code: str,
    name: str,
    hier_code: str,
    hier_level: int,
    l2: dict,
    l3: dict,
    l2_hier_to_idx: dict,
    l3_hier_to_idx: dict,
) -> dict:
    """Resolve the full hierarchy path (L2 → L3 → specific) for one term."""
    hier_parts = hier_code.split(".")

    l2_code = l2_name = None
    l3_code = l3_name = None
    specific_code = specific_name = None
    specific_level = None

    if level_name == "level_2":
        l2_code, l2_name = code, name
    else:
        if len(hier_parts) >= 2:
            l2_idx = l2_hier_to_idx.get(".".join(hier_parts[:2]))
            if l2_idx is not None:
                l2_code = l2["codes"][l2_idx]
                l2_name = l2["names"][l2_idx]

    if level_name == "level_3":
        l3_code, l3_name = code, name
    elif level_name == "level_specific":
        if len(hier_parts) >= 3:
            l3_idx = l3_hier_to_idx.get(".".join(hier_parts[:3]))
            if l3_idx is not None:
                l3_code = l3["codes"][l3_idx]
                l3_name = l3["names"][l3_idx]
        specific_code, specific_name, specific_level = code, name, hier_level

    return {
        "level2_code": l2_code,
        "level2_name": l2_name,
        "level3_code": l3_code,
        "level3_name": l3_name,
        "specific_code": specific_code,
        "specific_name": specific_name,
        "specific_level": specific_level,
        "final_level": hier_level,
    }


def build_flat_rapidfuzz_index(hierarchy: Dict[str, dict]) -> None:
    """Build a flat RapidFuzz variant index over all FoodEx2 food terms.

    Populates hierarchy["flat_rf"] with:
      variants           – list[str] of all comma-order + spelling variants
      variant_term_refs  – list[dict] parallel to variants; each entry carries
                           level_name, term_idx, termCode, termExtendedName,
                           reportHierarchyCode, reportHierarchyLevel, and the
                           precomputed hierarchy path dict.
    """
    l2 = hierarchy["level_2"]
    l3 = hierarchy["level_3"]

    l2_hier_to_idx = {str(hc): i for i, hc in enumerate(l2["hierarchy_codes"])}
    l3_hier_to_idx = {str(hc): i for i, hc in enumerate(l3["hierarchy_codes"])}

    variants: List[str] = []
    variant_term_refs: List[dict] = []

    for level_name in ("level_2", "level_3", "level_specific"):
        level = hierarchy[level_name]
        for term_idx, (code, name, hier_code, hier_level) in enumerate(zip(
            level["codes"], level["names"], level["hierarchy_codes"], level["levels"]
        )):
            path = _resolve_path(
                level_name, code, name, str(hier_code), int(hier_level),
                l2, l3, l2_hier_to_idx, l3_hier_to_idx,
            )
            for variant in comma_order_variants(name):
                variants.append(variant)
                variant_term_refs.append({
                    "level_name": level_name,
                    "term_idx": term_idx,
                    "termCode": code,
                    "termExtendedName": name,
                    "reportHierarchyCode": str(hier_code),
                    "reportHierarchyLevel": int(hier_level),
                    "path": path,
                })

    hierarchy["flat_rf"] = {
        "variants": variants,
        "variant_term_refs": variant_term_refs,
    }


def get_hierarchy_path_for_term(
    hierarchy: Dict[str, dict], level_name: str, term_code: str
) -> Optional[dict]:
    """Return the precomputed path dict for *term_code* at *level_name*.

    Requires build_flat_rapidfuzz_index() to have been called first.
    Returns None if the term is not found.
    """
    flat_rf = hierarchy.get("flat_rf")
    if flat_rf is None:
        return None
    for ref in flat_rf["variant_term_refs"]:
        if ref["termCode"] == term_code and ref["level_name"] == level_name:
            return ref["path"]
    return None


def generate_hierarchy_embeddings(hierarchy: Dict[str, dict], model) -> Dict[str, dict]:
    """Embed FoodEx2 terms using variant-averaged embeddings (see encode_name_variants)."""
    for level_name in ["level_2", "level_3", "level_specific"]:
        names = hierarchy[level_name]["names"]
        embeddings = encode_name_variants(names, model)
        hierarchy[level_name]["embeddings"] = embeddings
        hierarchy[level_name]["code_to_idx"] = {
            code: idx for idx, code in enumerate(hierarchy[level_name]["codes"])
        }
    return hierarchy


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StratifiedMatchResult:
    ingredient: str
    # Best match at each hierarchy level
    level2_code: Optional[str]
    level2_name: Optional[str]
    level2_score: float
    level3_code: Optional[str]
    level3_name: Optional[str]
    level3_score: float
    specific_code: Optional[str]
    specific_name: Optional[str]
    specific_score: float
    # Which level was chosen as final and the overall path score
    final_level: int
    confidence: str
    path_score: float          # beam-search path score used to select this result
    # Top-3 candidates at the final level (for diagnostics)
    top1_code: Optional[str]
    top1_name: Optional[str]
    top1_score: float
    top2_code: Optional[str]
    top2_name: Optional[str]
    top2_score: float
    top3_code: Optional[str]
    top3_name: Optional[str]
    top3_score: float
    # Matching-mode diagnostics (default to beam-search values)
    matching_mode: str = "beam"           # "rf_exact" | "rf_reranked" | "beam"
    rf_used: bool = False
    rf_score: float = 0.0
    rf_matched_variant: Optional[str] = field(default=None)
    bge_score: float = 0.0
    lexical_score: float = 0.0
    branch_penalty: float = 0.0          # L2 BGE prior used in RF reranking
    exact_variant_match: bool = False


# ── Matcher ───────────────────────────────────────────────────────────────────

class StratifiedMatcherOptimized:
    def __init__(
        self,
        hierarchy: Dict[str, dict],
        ingredient_embeddings: np.ndarray,
        ingredient_to_idx: dict,
        threshold_level2: float = 0.45,
        threshold_level3: float = 0.65,
        threshold_specific: float = 0.75,
        top_k_l2: int = 3,
        top_k_l3: int = 5,
        top_k_specific: int = 10,
        use_flat_rapidfuzz_short_filter: bool = True,
        rf_min_score: float = 65.0,
        rf_top_k_flat_variants: int = 100,
        rf_top_k_flat_terms: int = 30,
    ):
        self.hierarchy = hierarchy
        self.ingredient_embeddings = ingredient_embeddings
        self.ingredient_to_idx = ingredient_to_idx
        self.threshold_level2 = threshold_level2
        self.threshold_level3 = threshold_level3
        self.threshold_specific = threshold_specific
        self.top_k_l2 = top_k_l2
        self.top_k_l3 = top_k_l3
        self.top_k_specific = top_k_specific
        self.use_flat_rapidfuzz_short_filter = use_flat_rapidfuzz_short_filter
        self.rf_min_score = rf_min_score
        self.rf_top_k_flat_variants = rf_top_k_flat_variants
        self.rf_top_k_flat_terms = rf_top_k_flat_terms

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _top3(
        sims: np.ndarray,
        codes: list,
        names: list,
        global_idx: list = None,
    ) -> list:
        """Return top-3 (code, name, score) tuples from a 1-D similarity row.

        global_idx: when sims covers a children subset, maps local → global list index.
        Pads with (None, None, 0.0) when fewer than 3 candidates exist.
        """
        k = min(3, len(sims))
        top_local = np.argsort(sims)[-k:][::-1]
        result = []
        for li in top_local:
            gi = global_idx[li] if global_idx is not None else int(li)
            result.append((codes[gi], names[gi], float(sims[li])))
        while len(result) < 3:
            result.append((None, None, 0.0))
        return result

    def _get_ingredient_embedding(self, ingredient: str) -> Optional[np.ndarray]:
        idx = self.ingredient_to_idx.get(ingredient)
        if idx is None:
            return None
        return self.ingredient_embeddings[idx : idx + 1]

    def _no_embed_result(self, ingredient: str) -> StratifiedMatchResult:
        return StratifiedMatchResult(
            ingredient=ingredient,
            level2_code=None, level2_name=None, level2_score=0.0,
            level3_code=None, level3_name=None, level3_score=0.0,
            specific_code=None, specific_name=None, specific_score=0.0,
            final_level=0, confidence="NO_MATCH", path_score=0.0,
            top1_code=None, top1_name=None, top1_score=0.0,
            top2_code=None, top2_name=None, top2_score=0.0,
            top3_code=None, top3_name=None, top3_score=0.0,
        )

    @staticmethod
    def _combined_spec_score(ingredient: str, spec_name: str, emb_score: float) -> float:
        """Combine embedding score with token Jaccard; boost exact normalised matches."""
        lexical = token_jaccard(ingredient, spec_name)
        combined = 0.75 * emb_score + 0.25 * lexical
        if _nlk(ingredient) == _nlk(spec_name):
            combined = max(combined, 0.98)
        return combined

    # ── RapidFuzz flat-index retrieval ───────────────────────────────────────

    def _branch_prior(self, ing_emb: np.ndarray) -> dict:
        """Return {l2_code: bge_similarity} for every L2 node (one matmul)."""
        l2 = self.hierarchy["level_2"]
        sims = (ing_emb @ l2["embeddings"].T)[0]
        return {code: float(sims[i]) for i, code in enumerate(l2["codes"])}

    def _build_rf_result(
        self,
        ingredient: str,
        ref: dict,
        final_score: float,
        bge_score: float,
        lexical: float,
        confidence: Optional[str] = None,
        matching_mode: str = "rf_reranked",
        rf_score: float = 0.0,
        rf_matched_variant: Optional[str] = None,
        branch_penalty: float = 0.0,
        exact_variant_match: bool = False,
    ) -> StratifiedMatchResult:
        """Assemble a StratifiedMatchResult from a flat-RF candidate."""
        path = ref["path"]
        if confidence is None:
            if final_score >= self.threshold_specific and lexical >= 0.30:
                confidence = "HIGH"
            elif final_score >= self.threshold_level3:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        l2_score = bge_score if path["level2_code"] is not None else 0.0
        l3_score = bge_score if path["level3_code"] is not None else 0.0
        spec_score = final_score if path["specific_code"] is not None else 0.0
        return StratifiedMatchResult(
            ingredient=ingredient,
            level2_code=path["level2_code"], level2_name=path["level2_name"], level2_score=l2_score,
            level3_code=path["level3_code"], level3_name=path["level3_name"], level3_score=l3_score,
            specific_code=path["specific_code"], specific_name=path["specific_name"], specific_score=spec_score,
            final_level=path["final_level"], confidence=confidence, path_score=final_score,
            top1_code=ref["termCode"], top1_name=ref["termExtendedName"], top1_score=final_score,
            top2_code=None, top2_name=None, top2_score=0.0,
            top3_code=None, top3_name=None, top3_score=0.0,
            matching_mode=matching_mode,
            rf_used=True,
            rf_score=rf_score,
            rf_matched_variant=rf_matched_variant,
            bge_score=bge_score,
            lexical_score=lexical,
            branch_penalty=branch_penalty,
            exact_variant_match=exact_variant_match,
        )

    def _get_flat_rf_candidates(self, ingredient: str) -> list:
        """Search the flat FoodEx2 variant index with RapidFuzz.

        Returns a list of (ref, rf_score, matched_variant) tuples, deduplicated by
        termCode, sorted by rf_score descending, capped at rf_top_k_flat_terms.
        """
        query = _nlk(ingredient)
        flat_rf = self.hierarchy["flat_rf"]

        matches = rf_process.extract(
            query,
            flat_rf["variants"],
            scorer=fuzz.WRatio,
            limit=self.rf_top_k_flat_variants,
            score_cutoff=self.rf_min_score,
        )

        best_by_code: dict = {}
        for match_text, score, idx in matches:
            ref = flat_rf["variant_term_refs"][idx]
            code = ref["termCode"]
            if code not in best_by_code or score > best_by_code[code][1]:
                best_by_code[code] = (ref, float(score), match_text)

        sorted_cands = sorted(best_by_code.values(), key=lambda x: x[1], reverse=True)
        return sorted_cands[: self.rf_top_k_flat_terms]

    def match_with_flat_rf(self, ingredient: str) -> Optional[StratifiedMatchResult]:
        """Primary matching via flat RF with BGE reranking and hierarchy branch priors.

        Flow:
          1. Flat RF retrieves candidates from all FoodEx2 variants (all levels).
          2. Exact variant match (rf_score == 100) is accepted directly as HIGH.
          3. Non-exact candidates are reranked:
               0.50 * bge_score + 0.28 * (rf_score/100) + 0.10 * lexical + 0.12 * l2_prior
             where l2_prior is the BGE similarity of the ingredient to the candidate's
             L2 parent (one matmul computed once, shared across all candidates).
          Returns None if no RF candidates → caller falls back to beam search.
        """
        ing_emb = self._get_ingredient_embedding(ingredient)
        if ing_emb is None:
            return None

        candidates = self._get_flat_rf_candidates(ingredient)
        if not candidates:
            return None

        # Fast-path: exact variant match — accept directly, no BGE reranking needed
        for ref, rf_score, matched_variant in candidates:
            if rf_score == 100.0:
                term_emb = self.hierarchy[ref["level_name"]]["embeddings"][ref["term_idx"]]
                bge_score = float(ing_emb[0] @ term_emb)
                lexical = token_jaccard(ingredient, ref["termExtendedName"])
                final_score = 0.50 * bge_score + 0.28 + 0.10 * lexical
                return self._build_rf_result(
                    ingredient, ref, final_score, bge_score, lexical,
                    confidence="HIGH",
                    matching_mode="rf_exact",
                    rf_score=rf_score,
                    rf_matched_variant=matched_variant,
                    branch_penalty=0.0,
                    exact_variant_match=True,
                )

        # Non-exact: compute L2 branch priors once, then rerank all candidates
        branch_priors = self._branch_prior(ing_emb)

        best_final_score = -1.0
        best_ref: Optional[dict] = None
        best_bge = 0.0
        best_lexical = 0.0
        best_rf_score = 0.0
        best_matched_variant: Optional[str] = None
        best_prior = 0.0

        for ref, rf_score, matched_variant in candidates:
            term_emb = self.hierarchy[ref["level_name"]]["embeddings"][ref["term_idx"]]
            bge_score = float(ing_emb[0] @ term_emb)
            lexical = token_jaccard(ingredient, ref["termExtendedName"])
            prior = branch_priors.get(ref["path"]["level2_code"], 0.0)
            combined = 0.50 * bge_score + 0.28 * (rf_score / 100.0) + 0.10 * lexical + 0.12 * prior
            if combined > best_final_score:
                best_final_score = combined
                best_ref = ref
                best_bge = bge_score
                best_lexical = lexical
                best_rf_score = rf_score
                best_matched_variant = matched_variant
                best_prior = prior

        if best_ref is None:
            return None

        return self._build_rf_result(
            ingredient, best_ref, best_final_score, best_bge, best_lexical,
            matching_mode="rf_reranked",
            rf_score=best_rf_score,
            rf_matched_variant=best_matched_variant,
            branch_penalty=best_prior,
            exact_variant_match=False,
        )

    # ── match_single (beam search) ────────────────────────────────────────────

    def match_single(self, ingredient: str) -> StratifiedMatchResult:
        if self.use_flat_rapidfuzz_short_filter:
            rf_result = self.match_with_flat_rf(ingredient)
            if rf_result is not None:
                return rf_result

        ing_emb = self._get_ingredient_embedding(ingredient)
        if ing_emb is None:
            return self._no_embed_result(ingredient)

        l2 = self.hierarchy["level_2"]
        l3 = self.hierarchy["level_3"]
        lspec = self.hierarchy["level_specific"]

        sim_l2 = (ing_emb @ l2["embeddings"].T)[0]
        t2 = self._top3(sim_l2, l2["codes"], l2["names"])

        # top_k_l2 candidates above threshold
        k2 = min(self.top_k_l2, len(sim_l2))
        l2_top = np.argsort(sim_l2)[-k2:][::-1]
        l2_cands = [(int(li), float(sim_l2[li])) for li in l2_top if sim_l2[li] >= self.threshold_level2]

        if not l2_cands:
            return StratifiedMatchResult(
                ingredient=ingredient,
                level2_code=t2[0][0], level2_name=t2[0][1], level2_score=t2[0][2],
                level3_code=None, level3_name=None, level3_score=0.0,
                specific_code=None, specific_name=None, specific_score=0.0,
                final_level=2, confidence="NO_MATCH", path_score=0.0,
                top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
            )

        best: Optional[StratifiedMatchResult] = None

        def _update(candidate: StratifiedMatchResult) -> None:
            nonlocal best
            if best is None or candidate.path_score > best.path_score:
                best = candidate

        for l2_local, l2_score in l2_cands:
            l2_code = l2["codes"][l2_local]
            l2_name = l2["names"][l2_local]

            children_l3 = self.hierarchy["children_map"].get(l2_code, [])
            children_l3_idx = [l3["code_to_idx"][c] for c in children_l3 if c in l3["code_to_idx"]]

            if not children_l3_idx:
                ps = 0.40 * l2_score
                _update(StratifiedMatchResult(
                    ingredient=ingredient,
                    level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                    level3_code=None, level3_name=None, level3_score=0.0,
                    specific_code=None, specific_name=None, specific_score=0.0,
                    final_level=2, confidence="LOW", path_score=ps,
                    top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                    top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                    top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
                ))
                continue

            sim_l3_sub = (ing_emb @ l3["embeddings"][children_l3_idx].T)[0]
            t3_sub = self._top3(sim_l3_sub, l3["codes"], l3["names"], children_l3_idx)

            k3 = min(self.top_k_l3, len(children_l3_idx))
            l3_top = np.argsort(sim_l3_sub)[-k3:][::-1]
            l3_cands = [(int(li), float(sim_l3_sub[li])) for li in l3_top if sim_l3_sub[li] >= self.threshold_level3]

            if not l3_cands:
                ps = 0.40 * l2_score
                _update(StratifiedMatchResult(
                    ingredient=ingredient,
                    level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                    level3_code=t3_sub[0][0], level3_name=t3_sub[0][1],
                    level3_score=float(sim_l3_sub.max()),
                    specific_code=None, specific_name=None, specific_score=0.0,
                    final_level=2, confidence="LOW", path_score=ps,
                    top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                    top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                    top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
                ))
                continue

            for l3_local, l3_score in l3_cands:
                l3_global = children_l3_idx[l3_local]
                l3_code = l3["codes"][l3_global]
                l3_name = l3["names"][l3_global]

                children_spec = self.hierarchy["children_map"].get(l3_code, [])
                children_spec_idx = [lspec["code_to_idx"][c] for c in children_spec if c in lspec["code_to_idx"]]

                if not children_spec_idx:
                    ps = 0.40 * l2_score + 0.60 * l3_score
                    _update(StratifiedMatchResult(
                        ingredient=ingredient,
                        level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                        level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                        specific_code=None, specific_name=None, specific_score=0.0,
                        final_level=3, confidence="MEDIUM", path_score=ps,
                        top1_code=t3_sub[0][0], top1_name=t3_sub[0][1], top1_score=t3_sub[0][2],
                        top2_code=t3_sub[1][0], top2_name=t3_sub[1][1], top2_score=t3_sub[1][2],
                        top3_code=t3_sub[2][0], top3_name=t3_sub[2][1], top3_score=t3_sub[2][2],
                    ))
                    continue

                sim_spec = (ing_emb @ lspec["embeddings"][children_spec_idx].T)[0]
                tspec = self._top3(sim_spec, lspec["codes"], lspec["names"], children_spec_idx)

                k_spec = min(self.top_k_specific, len(children_spec_idx))
                spec_top = np.argsort(sim_spec)[-k_spec:][::-1]

                best_combined = -1.0
                best_spec_code = best_spec_name = None
                best_spec_level = 3

                for spec_local in spec_top:
                    emb_score = float(sim_spec[spec_local])
                    if emb_score < self.threshold_specific:
                        break  # sorted descending; no need to continue
                    spec_global = children_spec_idx[spec_local]
                    combined = self._combined_spec_score(
                        ingredient,
                        lspec["names"][spec_global],
                        emb_score,
                    )
                    if combined > best_combined:
                        best_combined = combined
                        best_spec_code = lspec["codes"][spec_global]
                        best_spec_name = lspec["names"][spec_global]
                        best_spec_level = lspec["levels"][spec_global]

                if best_spec_code is not None:
                    ps = 0.20 * l2_score + 0.30 * l3_score + 0.50 * best_combined
                    _update(StratifiedMatchResult(
                        ingredient=ingredient,
                        level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                        level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                        specific_code=best_spec_code, specific_name=best_spec_name,
                        specific_score=best_combined,
                        final_level=best_spec_level, confidence="HIGH", path_score=ps,
                        top1_code=tspec[0][0], top1_name=tspec[0][1], top1_score=tspec[0][2],
                        top2_code=tspec[1][0], top2_name=tspec[1][1], top2_score=tspec[1][2],
                        top3_code=tspec[2][0], top3_name=tspec[2][1], top3_score=tspec[2][2],
                    ))
                else:
                    # spec below threshold — L3 fallback
                    ps = 0.40 * l2_score + 0.60 * l3_score
                    _update(StratifiedMatchResult(
                        ingredient=ingredient,
                        level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                        level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                        specific_code=tspec[0][0], specific_name=tspec[0][1],
                        specific_score=tspec[0][2],
                        final_level=3, confidence="MEDIUM", path_score=ps,
                        top1_code=t3_sub[0][0], top1_name=t3_sub[0][1], top1_score=t3_sub[0][2],
                        top2_code=t3_sub[1][0], top2_name=t3_sub[1][1], top2_score=t3_sub[1][2],
                        top3_code=t3_sub[2][0], top3_name=t3_sub[2][1], top3_score=t3_sub[2][2],
                    ))

        return best if best is not None else self._no_embed_result(ingredient)

    # ── match_batch (vectorised beam search) ─────────────────────────────────

    def match_batch(self, ingredients: List[str], show_progress: bool = True) -> List[StratifiedMatchResult]:
        """Batch matching with beam search.

        Vectorisation strategy:
          - One L2 matmul for all M ingredients.
          - One L3 matmul per unique winning L2 node (same as before, but now
            each ingredient contributes to up to top_k_l2 L2 groups).
          - One specific matmul per unique winning L3 node.
          - Per-ingredient best_path accumulator is updated whenever a complete
            path scores better than the current best.
        """
        n = len(ingredients)
        results: list = [None] * n

        # RF primary stage: run flat RF on all ingredients before beam search
        if self.use_flat_rapidfuzz_short_filter:
            for k, ing in enumerate(ingredients):
                if ing in self.ingredient_to_idx:
                    rf_result = self.match_with_flat_rf(ing)
                    if rf_result is not None:
                        results[k] = rf_result

        # Build valid_ks for beam search, skipping ingredients already handled by RF
        valid_ks: list = []
        for k, ing in enumerate(ingredients):
            if results[k] is not None:
                continue
            if ing in self.ingredient_to_idx:
                valid_ks.append(k)
            else:
                results[k] = self._no_embed_result(ing)

        if not valid_ks:
            return results  # type: ignore

        valid_ings = [ingredients[k] for k in valid_ks]
        M = len(valid_ks)
        batch_emb = self.ingredient_embeddings[
            [self.ingredient_to_idx[ing] for ing in valid_ings]
        ]  # (M, D)

        l2 = self.hierarchy["level_2"]
        l3 = self.hierarchy["level_3"]
        lspec = self.hierarchy["level_specific"]

        # ── Level 2: one matmul for all M ingredients ─────────────────────────
        sims_l2 = batch_emb @ l2["embeddings"].T  # (M, L2_count)

        # Precompute L2 top-3 for diagnostics (used in fallback results)
        t2_all = [self._top3(sims_l2[i], l2["codes"], l2["names"]) for i in range(M)]

        # Per-ingredient: top_k_l2 candidates above threshold
        l2_cands: list = []
        for i in range(M):
            row = sims_l2[i]
            k = min(self.top_k_l2, len(row))
            top_k = np.argsort(row)[-k:][::-1]
            cands = [(int(li), float(row[li])) for li in top_k if row[li] >= self.threshold_level2]
            l2_cands.append(cands)

        # Handle NO_MATCH immediately
        for i, cands in enumerate(l2_cands):
            if not cands:
                t2 = t2_all[i]
                results[valid_ks[i]] = StratifiedMatchResult(
                    ingredient=valid_ings[i],
                    level2_code=t2[0][0], level2_name=t2[0][1], level2_score=t2[0][2],
                    level3_code=None, level3_name=None, level3_score=0.0,
                    specific_code=None, specific_name=None, specific_score=0.0,
                    final_level=2, confidence="NO_MATCH", path_score=0.0,
                    top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                    top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                    top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
                )

        # Per-ingredient best-path accumulator
        best_path: list = [None] * M

        def _update(i: int, candidate: StratifiedMatchResult) -> None:
            if best_path[i] is None or candidate.path_score > best_path[i].path_score:
                best_path[i] = candidate

        # Group by L2 local index: one ingredient may appear in multiple groups
        l2_groups: dict = defaultdict(list)   # l2_local_idx -> [(ingr_i, l2_score)]
        for i, cands in enumerate(l2_cands):
            for l2_local, l2_score in cands:
                l2_groups[l2_local].append((i, l2_score))

        # ── Level 3: one matmul per unique L2 group ───────────────────────────
        for l2_local, ingr_scores in l2_groups.items():
            l2_code = l2["codes"][l2_local]
            l2_name = l2["names"][l2_local]
            ingr_is = [x[0] for x in ingr_scores]
            l2_score_map = {x[0]: x[1] for x in ingr_scores}

            children_l3_codes = self.hierarchy["children_map"].get(l2_code, [])
            children_l3_idx = [
                l3["code_to_idx"][c] for c in children_l3_codes if c in l3["code_to_idx"]
            ]

            if not children_l3_idx:
                for i, l2_score in ingr_scores:
                    ps = 0.40 * l2_score
                    t2 = t2_all[i]
                    _update(i, StratifiedMatchResult(
                        ingredient=valid_ings[i],
                        level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                        level3_code=None, level3_name=None, level3_score=0.0,
                        specific_code=None, specific_name=None, specific_score=0.0,
                        final_level=2, confidence="LOW", path_score=ps,
                        top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                        top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                        top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
                    ))
                continue

            group_emb = batch_emb[ingr_is]                          # (G, D)
            children_l3_emb = l3["embeddings"][children_l3_idx]     # (C3, D)
            sims_l3 = group_emb @ children_l3_emb.T                 # (G, C3)
            G = len(ingr_is)

            # Per-ingredient L3 candidates within this L2 group
            l3_cands_per_ingr: list = []
            for gi in range(G):
                row_l3 = sims_l3[gi]
                k3 = min(self.top_k_l3, len(children_l3_idx))
                top_k3 = np.argsort(row_l3)[-k3:][::-1]
                cands_l3 = [(int(li), float(row_l3[li])) for li in top_k3 if row_l3[li] >= self.threshold_level3]
                l3_cands_per_ingr.append(cands_l3)

            # L2-only fallback for ingredients whose best L3 is below threshold
            for gi, (i, _) in enumerate(ingr_scores):
                if not l3_cands_per_ingr[gi]:
                    l2_score = l2_score_map[i]
                    ps = 0.40 * l2_score
                    best_l3_local = int(np.argmax(sims_l3[gi]))
                    l3_global = children_l3_idx[best_l3_local]
                    t2 = t2_all[i]
                    _update(i, StratifiedMatchResult(
                        ingredient=valid_ings[i],
                        level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                        level3_code=l3["codes"][l3_global], level3_name=l3["names"][l3_global],
                        level3_score=float(sims_l3[gi, best_l3_local]),
                        specific_code=None, specific_name=None, specific_score=0.0,
                        final_level=2, confidence="LOW", path_score=ps,
                        top1_code=t2[0][0], top1_name=t2[0][1], top1_score=t2[0][2],
                        top2_code=t2[1][0], top2_name=t2[1][1], top2_score=t2[1][2],
                        top3_code=t2[2][0], top3_name=t2[2][1], top3_score=t2[2][2],
                    ))

            # Group by L3 local index for spec matmuls
            # l3_local -> [(gi, ingr_i, l2_score, l3_score)]
            l3_groups: dict = defaultdict(list)
            for gi, (i, _) in enumerate(ingr_scores):
                for l3_local, l3_score in l3_cands_per_ingr[gi]:
                    l3_groups[l3_local].append((gi, i, l2_score_map[i], l3_score))

            # ── Specific: one matmul per unique L3 group ──────────────────────
            for l3_local, gi_ingr_scores in l3_groups.items():
                l3_global_idx = children_l3_idx[l3_local]
                l3_code = l3["codes"][l3_global_idx]
                l3_name = l3["names"][l3_global_idx]

                children_spec_codes = self.hierarchy["children_map"].get(l3_code, [])
                children_spec_idx = [
                    lspec["code_to_idx"][c]
                    for c in children_spec_codes
                    if c in lspec["code_to_idx"]
                ]

                if not children_spec_idx:
                    for gi, i, l2_score, l3_score in gi_ingr_scores:
                        ps = 0.40 * l2_score + 0.60 * l3_score
                        t3 = self._top3(sims_l3[gi], l3["codes"], l3["names"], children_l3_idx)
                        _update(i, StratifiedMatchResult(
                            ingredient=valid_ings[i],
                            level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                            level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                            specific_code=None, specific_name=None, specific_score=0.0,
                            final_level=3, confidence="MEDIUM", path_score=ps,
                            top1_code=t3[0][0], top1_name=t3[0][1], top1_score=t3[0][2],
                            top2_code=t3[1][0], top2_name=t3[1][1], top2_score=t3[1][2],
                            top3_code=t3[2][0], top3_name=t3[2][1], top3_score=t3[2][2],
                        ))
                    continue

                # Deduplicate ingredients within this L3 group (same ingredient
                # may appear via multiple L2 paths or multiple L3 candidates)
                unique_spec_is = list(dict.fromkeys(x[1] for x in gi_ingr_scores))
                spec_emb = batch_emb[unique_spec_is]                              # (U, D)
                children_spec_emb = lspec["embeddings"][children_spec_idx]        # (CS, D)
                sims_spec = spec_emb @ children_spec_emb.T                        # (U, CS)
                spec_i_to_si = {i: si for si, i in enumerate(unique_spec_is)}

                for gi, i, l2_score, l3_score in gi_ingr_scores:
                    si = spec_i_to_si[i]
                    row_spec = sims_spec[si]
                    k_spec = min(self.top_k_specific, len(children_spec_idx))
                    spec_top = np.argsort(row_spec)[-k_spec:][::-1]
                    tspec = self._top3(row_spec, lspec["codes"], lspec["names"], children_spec_idx)
                    t3 = self._top3(sims_l3[gi], l3["codes"], l3["names"], children_l3_idx)

                    best_combined = -1.0
                    best_spec_code = best_spec_name = None
                    best_spec_level = 3

                    for spec_local in spec_top:
                        emb_score = float(row_spec[spec_local])
                        if emb_score < self.threshold_specific:
                            break  # sorted descending
                        spec_global = children_spec_idx[spec_local]
                        combined = self._combined_spec_score(
                            valid_ings[i],
                            lspec["names"][spec_global],
                            emb_score,
                        )
                        if combined > best_combined:
                            best_combined = combined
                            best_spec_code = lspec["codes"][spec_global]
                            best_spec_name = lspec["names"][spec_global]
                            best_spec_level = lspec["levels"][spec_global]

                    if best_spec_code is not None:
                        ps = 0.20 * l2_score + 0.30 * l3_score + 0.50 * best_combined
                        _update(i, StratifiedMatchResult(
                            ingredient=valid_ings[i],
                            level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                            level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                            specific_code=best_spec_code, specific_name=best_spec_name,
                            specific_score=best_combined,
                            final_level=best_spec_level, confidence="HIGH", path_score=ps,
                            top1_code=tspec[0][0], top1_name=tspec[0][1], top1_score=tspec[0][2],
                            top2_code=tspec[1][0], top2_name=tspec[1][1], top2_score=tspec[1][2],
                            top3_code=tspec[2][0], top3_name=tspec[2][1], top3_score=tspec[2][2],
                        ))
                    else:
                        # spec below threshold — L3 fallback
                        ps = 0.40 * l2_score + 0.60 * l3_score
                        _update(i, StratifiedMatchResult(
                            ingredient=valid_ings[i],
                            level2_code=l2_code, level2_name=l2_name, level2_score=l2_score,
                            level3_code=l3_code, level3_name=l3_name, level3_score=l3_score,
                            specific_code=tspec[0][0], specific_name=tspec[0][1],
                            specific_score=tspec[0][2],
                            final_level=3, confidence="MEDIUM", path_score=ps,
                            top1_code=t3[0][0], top1_name=t3[0][1], top1_score=t3[0][2],
                            top2_code=t3[1][0], top2_name=t3[1][1], top2_score=t3[1][2],
                            top3_code=t3[2][0], top3_name=t3[2][1], top3_score=t3[2][2],
                        ))

        # Fill results from best_path accumulator
        for i in range(M):
            if results[valid_ks[i]] is None:
                results[valid_ks[i]] = (
                    best_path[i]
                    if best_path[i] is not None
                    else self._no_embed_result(valid_ings[i])
                )

        return results  # type: ignore
