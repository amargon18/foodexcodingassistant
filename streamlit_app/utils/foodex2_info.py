from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_TERMS_PATH    = _PROJECT_ROOT / "data" / "foodex2_terms.parquet"
_ALLERGEN_PATH = _PROJECT_ROOT / "allergenscan" / "data" / "foodex2_to_allergen.csv"


# ── Data loading (cached once per server process) ─────────────────────────────

@st.cache_data(show_spinner=False)
def _load_terms() -> Tuple[pd.DataFrame, Dict, Dict]:
    df = pd.read_parquet(_TERMS_PATH)
    code_to_row: Dict[str, dict] = df.set_index("termCode").to_dict("index")
    path_to_code: Dict[str, str] = {
        r["reportHierarchyCode"]: code for code, r in code_to_row.items()
    }
    return df, code_to_row, path_to_code


@st.cache_data(show_spinner=False)
def _load_allergen_map() -> Dict[str, List[Tuple[str, str]]]:
    df = pd.read_csv(_ALLERGEN_PATH)
    df = df.dropna(subset=["foodex2_code"])
    df = df[~df["foodex2_code"].astype(str).str.startswith("#")]
    result: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        code     = str(row["foodex2_code"]).strip()
        allergen = str(row["allergen"]).strip()
        certainty = str(row["certainty"]).strip()
        result.setdefault(code, []).append((allergen, certainty))
    return result


# ── Public helpers ────────────────────────────────────────────────────────────

def get_hierarchy_path(code: str) -> List[Tuple[int, str, str]]:
    """Return [(level, code, name), ...] from root to the given code."""
    _, code_to_row, path_to_code = _load_terms()
    row = code_to_row.get(code)
    if not row:
        return []
    parts = row["reportHierarchyCode"].split(".")
    ancestors = []
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        ancestor_code = path_to_code.get(prefix)
        if ancestor_code:
            r = code_to_row[ancestor_code]
            ancestors.append((r["reportHierarchyLevel"], ancestor_code, r["termExtendedName"]))
    return ancestors


def get_direct_children(code: str) -> List[Tuple[str, str]]:
    """Return [(code, name)] of the direct children of *code* in the hierarchy."""
    df, code_to_row, _ = _load_terms()
    row = code_to_row.get(code)
    if not row:
        return []
    parent_path  = row["reportHierarchyCode"]
    parent_level = row["reportHierarchyLevel"]
    mask = (
        df["reportHierarchyCode"].str.startswith(parent_path + ".")
        & (df["reportHierarchyLevel"] == parent_level + 1)
    )
    children = df[mask]
    return list(zip(children["termCode"], children["termExtendedName"]))


def get_allergen_mapping(code: str) -> List[Tuple[str, str]]:
    """Return [(allergen, certainty)] from the allergen CSV for *code*."""
    return _load_allergen_map().get(code, [])


def get_term_name(code: str) -> Optional[str]:
    """Return the extended name for *code*, or None if not found."""
    _, code_to_row, _ = _load_terms()
    row = code_to_row.get(code)
    return row["termExtendedName"] if row else None


# ── Facet helpers ─────────────────────────────────────────────────────────────

_FACET_RE = re.compile(r"^(F\d+)\.([A-Z0-9]+)$")


def parse_allFacets(facets_str: str) -> List[Dict[str, str]]:
    """Parse raw FoodEx2 allFacets notation into structured records.

    Format: ``{base_code}#{facet_cat}.{facet_code}${facet_cat}.{facet_code}…``

    Returns a list of ``{"facet_cat": "F01", "facet_code": "A059P"}`` dicts.
    Returns ``[]`` when the string contains no facets (base code only, or null).
    """
    if not facets_str or pd.isna(facets_str):
        return []
    s = str(facets_str)
    if "#" not in s:
        return []
    result = []
    for segment in s.split("#", 1)[1].split("$"):
        segment = segment.strip()
        m = _FACET_RE.match(segment)
        if m:
            result.append({"facet_cat": m.group(1), "facet_code": m.group(2)})
    return result


def get_facets_resolved(code: str) -> List[Dict[str, str]]:
    """Return pre-resolved facets for *code* from the parquet's facets_resolved column.

    Each dict has keys: ``facet_cat``, ``facet_label``, ``code``, ``name``.
    Falls back to ``parse_allFacets`` (no label or name) when the resolved
    column is absent or unparseable.
    """
    _, code_to_row, _ = _load_terms()
    row = code_to_row.get(code)
    if not row:
        return []

    raw = row.get("facets_resolved")
    if raw and not (isinstance(raw, float) and pd.isna(raw)):
        try:
            facets = json.loads(str(raw))
            if isinstance(facets, list) and facets:
                return [f for f in facets if isinstance(f, dict)]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse raw notation (no label/name available)
    raw_all = row.get("allFacets")
    return parse_allFacets(raw_all) if raw_all else []
