import sys
from pathlib import Path

# Make streamlit_app/ importable (utils, components, coding_assistant)
_PAGE_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_PAGE_DIR))

_PROJECT_ROOT = _PAGE_DIR.parent

import streamlit as st

st.set_page_config(
    page_title="Codificador FoodEx2 · AllergenScan",
    page_icon="🏷️",
    layout="wide",
)

from utils.styles import inject_css, section_header
from utils.data_loader import ALLERGEN_NAMES_ES, ALLERGEN_EMOJIS
from utils.foodex2_info import (
    get_allergen_mapping, get_direct_children, get_facets_resolved, get_hierarchy_path,
)
from coding_assistant.models import Confidence, ReviewFlag, ReviewStatus

inject_css()

# ── Cached resources (loaded once per server process) ─────────────────────────

@st.cache_resource(show_spinner="Cargando modelo FoodEx2 (primera carga ~20s)...")
def _get_matcher():
    from coding_assistant.ingredient_matcher import IngredientMatcher
    return IngredientMatcher()


@st.cache_resource
def _get_analyzer():
    from coding_assistant.product_analyzer import ProductAnalyzer
    return ProductAnalyzer(matcher=_get_matcher())


# ── Product search (DuckDB) ───────────────────────────────────────────────────

_DB_PATH          = str(_PROJECT_ROOT / "openfoodfacts.duckdb")
_PARQUET_PRODUCTS = str(_PROJECT_ROOT / "data" / "products_ingredients.parquet")
_PARQUET_MATCHES  = str(_PROJECT_ROOT / "data" / "matching_results_ingredients_baai__bge_small_en_v1_5.parquet")

# OFF allergen tag → internal allergen code (same mapping as detector.py)
_OFF_TAG_TO_ALLERGEN: dict = {
    "en:gluten": "gluten", "en:wheat": "gluten", "en:barley": "gluten",
    "en:rye": "gluten", "en:oats": "gluten", "en:spelt": "gluten",
    "en:crustaceans": "crustaceans",
    "en:eggs": "eggs",
    "en:fish": "fish",
    "en:peanuts": "peanuts",
    "en:soybeans": "soy", "en:soy": "soy",
    "en:milk": "milk",
    "en:nuts": "tree_nuts", "en:almonds": "tree_nuts", "en:hazelnuts": "tree_nuts",
    "en:walnuts": "tree_nuts", "en:cashew-nuts": "tree_nuts",
    "en:pecan-nuts": "tree_nuts", "en:brazil-nuts": "tree_nuts",
    "en:pistachio-nuts": "tree_nuts", "en:macadamia-nuts": "tree_nuts",
    "en:celery": "celery",
    "en:mustard": "mustard",
    "en:sesame-seeds": "sesame", "en:sesame": "sesame",
    "en:sulphur-dioxide-and-sulphites": "sulphites", "en:sulphites": "sulphites",
    "en:lupin": "lupin",
    "en:molluscs": "molluscs",
}


def _parse_allergen_tags(tags_str: str) -> set:
    """Parse a comma-separated OFF allergen tags string → set of internal codes."""
    if not tags_str or str(tags_str) == "nan":
        return set()
    result = set()
    for tag in str(tags_str).split(","):
        code = _OFF_TAG_TO_ALLERGEN.get(tag.strip().lower())
        if code:
            result.add(code)
    return result


@st.cache_data(show_spinner="Cargando productos precomputados...")
def _load_precomputed_products(limit: int = 5000):
    """Join parquet files to compute product-level confidence scores."""
    import duckdb
    return duckdb.execute(f"""
        WITH
        -- One row per unique (product, ingredient): eliminates duplicate tags in
        -- products_ingredients that share the same ingredient text.
        pi AS (
            SELECT code, MAX(product_name) AS product_name, LOWER(ingredient) AS ing_key
            FROM '{_PARQUET_PRODUCTS}'
            WHERE matchable = true
            GROUP BY code, LOWER(ingredient)
        ),
        -- One row per unique ingredient: the ingredient-level parquet should be
        -- deduplicated already, but QUALIFY guards against any future duplicates.
        m AS (
            SELECT LOWER(ingredient) AS ing_key, confidence, path_score
            FROM '{_PARQUET_MATCHES}'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY LOWER(ingredient)
                ORDER BY path_score DESC NULLS LAST
            ) = 1
        )
        SELECT
            pi.code,
            COALESCE(MAX(pi.product_name), 'Sin nombre')                          AS product_name,
            COUNT(*)                                                               AS total_ingredients,
            COUNT(CASE WHEN m.confidence = 'HIGH'     THEN 1 END)                AS high_count,
            COUNT(CASE WHEN m.confidence = 'MEDIUM'   THEN 1 END)                AS medium_count,
            COUNT(CASE WHEN m.confidence NOT IN ('NO_MATCH')
                            AND m.confidence IS NOT NULL THEN 1 END)              AS matched_count,
            COUNT(*) - COUNT(m.ing_key)                                           AS no_match_count,
            ROUND(
                0.4 * COALESCE(AVG(m.path_score), 0)
                + 0.3 * COUNT(CASE WHEN m.confidence = 'HIGH'   THEN 1 END)::DOUBLE / COUNT(*)
                + 0.2 * COUNT(CASE WHEN m.confidence NOT IN ('NO_MATCH')
                                        AND m.confidence IS NOT NULL THEN 1 END)::DOUBLE / COUNT(*)
                + 0.1 * COUNT(CASE WHEN m.confidence = 'MEDIUM' THEN 1 END)::DOUBLE / COUNT(*),
            4) AS confidence_score
        FROM pi
        LEFT JOIN m ON pi.ing_key = m.ing_key
        GROUP BY pi.code
        ORDER BY confidence_score DESC
        LIMIT {limit}
    """).df()


@st.cache_data(show_spinner=False)
def _load_product_ingredient_tags(code: str):
    """Return matchable pre-cleaned ingredient strings for a product from the parquet.

    Returns an empty list when the product is not present so the caller can
    fall back to parsing raw ingredients_text.
    """
    import duckdb
    try:
        rows = duckdb.execute(
            f"SELECT ingredient FROM '{_PARQUET_PRODUCTS}' WHERE code = ? AND matchable = true",
            [str(code)],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _search_by_barcode(barcode: str):
    """Query DuckDB for a product by exact barcode."""
    import duckdb
    con = duckdb.connect(_DB_PATH, read_only=True)
    df = con.execute(
        """
        SELECT code, product_name, ingredients_text, allergens, traces
        FROM products
        WHERE code = ?
          AND ingredients_text IS NOT NULL
        LIMIT 1
        """,
        [barcode.strip()],
    ).fetchdf()
    con.close()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _search_by_name(name: str):
    """Query DuckDB for products matching a name (case-insensitive LIKE)."""
    import duckdb
    con = duckdb.connect(_DB_PATH, read_only=True)
    df = con.execute(
        """
        SELECT code, product_name, ingredients_text, allergens, traces
        FROM products
        WHERE LOWER(product_name) LIKE LOWER(?)
          AND ingredients_text IS NOT NULL
          AND product_name IS NOT NULL
        ORDER BY LENGTH(product_name)
        LIMIT 15
        """,
        [f"%{name.strip()}%"],
    ).fetchdf()
    con.close()
    return df


def _load_product_into_state(row) -> None:
    """Copy a search result row into session_state for form pre-population."""
    allergens = _parse_allergen_tags(row.get("allergens", ""))
    traces    = _parse_allergen_tags(row.get("traces", ""))
    st.session_state["prefill_code"]        = str(row.get("code", ""))
    st.session_state["prefill_name"]        = str(row.get("product_name", ""))
    st.session_state["prefill_ingredients"] = str(row.get("ingredients_text", ""))
    st.session_state["prefill_allergens"]   = allergens | traces


# ── Rendering helpers ─────────────────────────────────────────────────────────

_STATUS_LABEL = {
    ReviewStatus.AUTO_APPROVED:   "AUTO APROBADO ✓",
    ReviewStatus.NEEDS_REVIEW:    "REVISIÓN NECESARIA ⚠️",
    ReviewStatus.MANUAL_REQUIRED: "REVISIÓN MANUAL ✗",
}

_FLAG_DESC = {
    ReviewFlag.UNKNOWN_INGREDIENT:   "Ingrediente sin match en la jerarquía FoodEx2.",
    ReviewFlag.LOW_CONFIDENCE:       "Match de baja confianza (sólo categoría L2).",
    ReviewFlag.MULTIPLE_CANDIDATES:  "Múltiples candidatos similares (diferencia de score < 0.05).",
    ReviewFlag.AMBIGUOUS_INGREDIENT: "Ingrediente ambiguo — puede contener múltiples alérgenos.",
    ReviewFlag.ALLERGEN_DISCREPANCY: "Alérgeno detectado no declarado en el etiquetado.",
}

ALLERGEN_KEYS = list(ALLERGEN_NAMES_ES.keys())


def _render_header(result, declared_set: set = None) -> None:
    from components.coding_cards import product_summary_html
    st.markdown(product_summary_html(result, declared_set or set()), unsafe_allow_html=True)


def _render_ingredients(result) -> None:
    from components.coding_cards import ingredient_card_html, foodex2_tree_html, facets_html
    section_header(f"🧪 Ingredientes ({len(result.ingredients)})")

    for m in result.ingredients:
        st.markdown(ingredient_card_html(m), unsafe_allow_html=True)

        if m.alternatives:
            with st.expander(f"Candidatos · «{m.original_text}»"):
                for alt in m.alternatives:
                    st.markdown(
                        f"- `{alt.code}` **{alt.name}** "
                        f"(score: {alt.score:.3f} · L{alt.level})"
                    )

        if m.final_code:
            with st.expander(f"FoodEx2 · {m.final_code} — {m.final_name}"):
                # ── Hierarchy path as tree ────────────────────────────────────
                path = get_hierarchy_path(m.final_code)
                if path:
                    st.markdown(
                        '<div class="section-label-sm">Ruta jerárquica</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(foodex2_tree_html(path, m.final_code), unsafe_allow_html=True)

                # ── Direct children for broad L2/L3 matches ───────────────────
                if m.final_level in (2, 3):
                    children = get_direct_children(m.final_code)
                    if children:
                        st.markdown(
                            f'<div class="section-label-sm">Subcategorías directas ({len(children)})</div>',
                            unsafe_allow_html=True,
                        )
                        for cc, cn in children[:12]:
                            st.markdown(f"- `{cc}` {cn}")
                        if len(children) > 12:
                            st.caption(f"… y {len(children) - 12} más")

                # ── Facets as inline tags ─────────────────────────────────────
                resolved_facets = get_facets_resolved(m.final_code)
                if resolved_facets:
                    st.markdown(
                        '<div class="section-label-sm">Facetas FoodEx2</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(facets_html(resolved_facets), unsafe_allow_html=True)

                # ── Allergen mapping from CSV ──────────────────────────────────
                mapped_allergens = get_allergen_mapping(m.final_code)
                if mapped_allergens:
                    st.markdown(
                        '<div class="section-label-sm">Alérgenos vinculados al código</div>',
                        unsafe_allow_html=True,
                    )
                    chips = "".join(
                        f'<span class="{"chip-detected-certain" if cert == "CERTAIN" else "chip-detected-possible"}">'
                        f'{ALLERGEN_EMOJIS.get(allergen, "")} {ALLERGEN_NAMES_ES.get(allergen, allergen)}'
                        f'</span>'
                        for allergen, cert in mapped_allergens
                    )
                    st.markdown(
                        f'<div class="allergen-grid">{chips}</div>',
                        unsafe_allow_html=True,
                    )
                elif m.allergens:
                    st.caption("Alérgenos detectados vía patrón ambiguo (no vinculados al código FoodEx2).")

                # ── Matching rationale ────────────────────────────────────────
                st.markdown(
                    '<div class="section-label-sm">Explicación del match</div>',
                    unsafe_allow_html=True,
                )
                mode_explain = {
                    "rf_exact":    "Coincidencia exacta por RapidFuzz con un término FoodEx2.",
                    "rf_reranked": "Candidato RapidFuzz reordenado por similitud BGE.",
                    "beam":        "Búsqueda en cascada L2→L3→específico por embeddings BGE.",
                }.get(m.matching_mode or "", "Modo de matching desconocido.")
                st.caption(mode_explain)

                _why_parts = [
                    f"path_score **{m.final_score:.3f}**",
                    f"nivel **L{m.final_level}**",
                ]
                if m.rf_score is not None:
                    _why_parts.append(f"RF **{m.rf_score:.0f}/100**")
                if m.branch_penalty is not None and abs(m.branch_penalty) > 1e-4:
                    _warn = " ⚠" if m.branch_penalty > 0.5 else ""
                    _why_parts.append(f"penalty **{m.branch_penalty:.3f}**{_warn}")
                if m.rf_matched_variant:
                    _why_parts.append(f"variante «{m.rf_matched_variant}»")
                st.markdown(" · ".join(_why_parts))


def _render_allergen_profile(result, declared_set: set) -> None:
    from components.coding_cards import allergen_grid_html
    section_header("🔬 Perfil de alérgenos")
    st.markdown(allergen_grid_html(result, declared_set), unsafe_allow_html=True)


def _render_review_decision(result) -> None:
    section_header("📋 Decisión de revisión")
    if result.review_status == ReviewStatus.AUTO_APPROVED:
        st.success(f"**{_STATUS_LABEL[result.review_status]}**\n\n{result.review_reason}")
    elif result.review_status == ReviewStatus.NEEDS_REVIEW:
        st.warning(f"**{_STATUS_LABEL[result.review_status]}**\n\n{result.review_reason}")
    else:
        st.error(f"**{_STATUS_LABEL[result.review_status]}**\n\n{result.review_reason}")

    if result.review_flags:
        st.markdown("**Flags activos:**")
        for flag in result.review_flags:
            st.markdown(f"- `{flag.value}` — {_FLAG_DESC.get(flag, '')}")

    from coding_assistant.confidence_scorer import score_breakdown
    with st.expander("Desglose del score de confianza"):
        st.code(score_breakdown(result.ingredients))


def _render_result(result, declared_set: set) -> None:
    _render_header(result, declared_set)
    st.divider()
    _render_ingredients(result)
    st.divider()
    _render_allergen_profile(result, declared_set)
    st.divider()
    _render_review_decision(result)


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("🏷️ Codificador FoodEx2")
st.caption(
    "Asigna códigos FoodEx2 a ingredientes · evalúa confianza · "
    "decide si el producto puede aprobarse automáticamente."
)

# ── Primary view: precomputed products ───────────────────────────────────────

section_header("📊 Productos con codificación precomputada")

_precomp_df = _load_precomputed_products()

# ── Suspicious-match filters ──────────────────────────────────────────────────
with st.expander("🔍 Filtros", expanded=False):
    _f1, _f2, _f3 = st.columns([2, 1, 2])
    with _f1:
        _min_conf = st.slider(
            "Confianza mínima", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="filter_min_conf",
        )
    with _f2:
        _min_ing = st.number_input(
            "Mín. ingredientes", min_value=1, value=1, step=1,
            key="filter_min_ing",
        )
    with _f3:
        _show_suspicious = st.checkbox(
            "Solo sospechosos (ingredientes sin match en parquet)",
            value=False,
            key="filter_suspicious",
            help=(
                "Muestra productos con al menos un ingrediente que no tiene "
                "resultado precomputado (no_match_count > 0). "
                "Esto puede indicar cobertura parcial del modelo."
            ),
        )

    _s1, _s2, _s3 = st.columns([2, 2, 1])
    with _s1:
        _sort_by = st.selectbox(
            "Ordenar por",
            options=["confidence_score", "total_ingredients"],
            format_func=lambda x: {
                "confidence_score":  "Confianza",
                "total_ingredients": "Nº ingredientes",
            }[x],
            key="sort_by",
        )
    with _s2:
        _sort_dir = st.radio(
            "Dirección",
            options=["Descendente", "Ascendente"],
            horizontal=True,
            key="sort_dir",
        )

# Reset to page 0 when sort parameters change so the user isn't left on a
# page that no longer makes sense for the new ordering.
_sort_key = f"{_sort_by}_{_sort_dir}"
if st.session_state.get("_prev_sort_key") != _sort_key:
    st.session_state["precomp_page"] = 0
    st.session_state["_prev_sort_key"] = _sort_key

_filtered_df = _precomp_df[
    (_precomp_df["confidence_score"] >= _min_conf) &
    (_precomp_df["total_ingredients"] >= _min_ing)
]
if _show_suspicious:
    _filtered_df = _filtered_df[_filtered_df["no_match_count"] > 0]

_filtered_df = _filtered_df.sort_values(
    _sort_by, ascending=(_sort_dir == "Ascendente")
)

# ── Pagination controls ───────────────────────────────────────────────────────
_ctrl_left, _ctrl_right = st.columns([3, 1])
with _ctrl_left:
    _page_size = st.slider(
        "Productos por página", min_value=10, max_value=20, value=10, step=1,
        key="precomp_page_size",
    )
with _ctrl_right:
    _total_pages = max(1, (len(_filtered_df) - 1) // _page_size + 1)
    st.markdown(
        f'<p style="text-align:right;margin-top:28px;color:#64748B">'
        f'{len(_filtered_df):,} productos · {_total_pages:,} páginas</p>',
        unsafe_allow_html=True,
    )

if "precomp_page" not in st.session_state:
    st.session_state["precomp_page"] = 0
_page_num = min(st.session_state["precomp_page"], _total_pages - 1)
st.session_state["precomp_page"] = _page_num

_start = _page_num * _page_size
_page_df = _filtered_df.iloc[_start : _start + _page_size]

# Table header
_hcols = st.columns([4, 2, 1, 2, 2])
for _col, _label in zip(_hcols, ["Producto", "Código", "Ingredientes", "Confianza", "Acciones"]):
    _col.markdown(f"**{_label}**")
st.divider()

# Product rows
for _, _row in _page_df.iterrows():
    _rc = st.columns([4, 2, 1, 2, 2])
    _score    = float(_row["confidence_score"])
    _no_match = int(_row.get("no_match_count", 0))
    _color = "#16a34a" if _score >= 0.70 else ("#ea580c" if _score >= 0.50 else "#dc2626")
    _suspicious_badge = (
        ' <span style="color:#dc2626;font-size:0.75em">⚠ parcial</span>'
        if _no_match > 0 else ""
    )

    _rc[0].markdown(
        f'{_row["product_name"]}{_suspicious_badge}',
        unsafe_allow_html=True,
    )
    _conf_cls = "dash-conf-high" if _score >= 0.70 else ("dash-conf-medium" if _score >= 0.50 else "dash-conf-low")
    _rc[1].markdown(f'<span class="code-mono">{_row["code"]}</span>', unsafe_allow_html=True)
    _rc[2].markdown(str(int(_row["total_ingredients"])))
    _rc[3].markdown(
        f'<span class="{_conf_cls}">{_score:.1%}</span> '
        f'<span class="dash-counts">'
        f'{int(_row["high_count"])}H·{int(_row["medium_count"])}M'
        f'{f"·{_no_match}?" if _no_match > 0 else ""}</span>',
        unsafe_allow_html=True,
    )

    _bc1, _bc2 = _rc[4].columns(2)
    if _bc1.button("Cargar ↓", key=f"pre_load_{_row['code']}", use_container_width=True):
        _db_rows = _search_by_barcode(str(_row["code"]))
        if not _db_rows.empty:
            _load_product_into_state(_db_rows.iloc[0].to_dict())
        st.rerun()
    if _bc2.button("Analizar 🔬", key=f"pre_analyze_{_row['code']}", use_container_width=True, type="primary"):
        _db_rows = _search_by_barcode(str(_row["code"]))
        if not _db_rows.empty:
            _load_product_into_state(_db_rows.iloc[0].to_dict())
            st.session_state["analyze_now"] = True
        st.rerun()

# Pagination controls
_nav_prev, _nav_info, _nav_next = st.columns([1, 4, 1])
if _nav_prev.button("← Anterior", disabled=(_page_num == 0), use_container_width=True):
    st.session_state["precomp_page"] -= 1
    st.rerun()
_nav_info.markdown(
    f'<p style="text-align:center;margin-top:8px">Página {_page_num + 1} de {_total_pages}</p>',
    unsafe_allow_html=True,
)
if _nav_next.button("Siguiente →", disabled=(_page_num >= _total_pages - 1), use_container_width=True):
    st.session_state["precomp_page"] += 1
    st.rerun()

st.divider()

# ── Product search section ────────────────────────────────────────────────────

with st.expander("🔍 Buscar producto por código de barras o nombre", expanded=True):
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        search_query = st.text_input(
            "Buscar",
            placeholder="Código de barras (ej: 3017620422003) o nombre (ej: Nutella)",
            label_visibility="collapsed",
        )
    with btn_col:
        do_search = st.button("🔍 Buscar", use_container_width=True)

    if do_search and search_query.strip():
        query = search_query.strip()
        with st.spinner("Buscando..."):
            # Detect barcode (all digits) vs name search
            if query.isdigit():
                results_df = _search_by_barcode(query)
                search_type = "barcode"
            else:
                results_df = _search_by_name(query)
                search_type = "name"

        if results_df.empty:
            st.warning(
                f"No se encontraron productos con «{query}». "
                "Prueba con otro código o nombre."
            )
        else:
            st.success(f"Se encontraron **{len(results_df)}** producto(s).")
            for _, row in results_df.iterrows():
                r_code = row.get("code", "")
                r_name = row.get("product_name", "Sin nombre")
                r_ing  = str(row.get("ingredients_text", ""))
                ing_preview = r_ing[:80] + "…" if len(r_ing) > 80 else r_ing

                c_info, c_load, c_analyze = st.columns([5, 1, 1])
                with c_info:
                    st.markdown(
                        f"**{r_name}** &nbsp; "
                        f'<span style="color:#64748B;font-size:0.85em">`{r_code}`</span>'
                        f'<br/><span style="color:#94A3B8;font-size:0.82em">{ing_preview}</span>',
                        unsafe_allow_html=True,
                    )
                with c_load:
                    if st.button("Cargar ↓", key=f"load_{r_code}", use_container_width=True):
                        _load_product_into_state(row.to_dict())
                        st.rerun()
                with c_analyze:
                    if st.button("Analizar 🔬", key=f"analyze_{r_code}", use_container_width=True, type="primary"):
                        _load_product_into_state(row.to_dict())
                        st.session_state["analyze_now"] = True
                        st.rerun()

st.divider()

# ── Read pre-filled values from session state (set by search loader) ──────────

prefill_code        = st.session_state.get("prefill_code", "")
prefill_name        = st.session_state.get("prefill_name", "")
prefill_ingredients = st.session_state.get("prefill_ingredients", "")
prefill_allergens   = st.session_state.get("prefill_allergens", set())

# ── Split-screen: form (left) | results (right) ───────────────────────────────

_form_col, _result_col = st.columns([2, 3], gap="large")

with _form_col:
    section_header("🔬 Análisis de producto")

    if prefill_code or prefill_name:
        st.info(
            f"**{prefill_name}** · `{prefill_code}`\n\n"
            "Edita los campos antes de analizar si lo necesitas.",
            icon="📦",
        )

    with st.form("coding_form"):
        col1, col2 = st.columns(2)
        with col1:
            product_code = st.text_input(
                "Código de barras",
                value=prefill_code,
                placeholder="Ej: 3017620422003",
            )
        with col2:
            product_name = st.text_input(
                "Nombre del producto",
                value=prefill_name,
                placeholder="Ej: Nutella",
            )

        ingredients_text = st.text_area(
            "Lista de ingredientes",
            value=prefill_ingredients,
            placeholder=(
                "Ej: sugar, palm oil, hazelnuts, skimmed milk powder, "
                "fat-reduced cocoa, emulsifier: soy lecithin, vanillin"
            ),
            height=130,
            help="Separa los ingredientes por comas o saltos de línea.",
        )

        st.markdown(
            '<div class="section-label-sm">Alérgenos declarados (opcional)</div>',
            unsafe_allow_html=True,
        )
        allergen_cols = st.columns(2)
        declared_checkboxes: dict = {}
        for i, key in enumerate(ALLERGEN_KEYS):
            with allergen_cols[i % 2]:
                declared_checkboxes[key] = st.checkbox(
                    f"{ALLERGEN_EMOJIS.get(key, '')} {ALLERGEN_NAMES_ES[key]}",
                    value=(key in prefill_allergens),
                    key=f"allergen_{key}",
                )

        submitted = st.form_submit_button(
            "🔬 Analizar producto", type="primary", use_container_width=True
        )


def _run_analysis(code: str, name: str, ingredients: str, allergens: set) -> None:
    """Run analysis and render results. Shared by form submit and direct search button."""
    if not ingredients.strip():
        st.warning("Este producto no tiene lista de ingredientes disponible.")
        return

    # Primary path: use pre-cleaned matchable tokens from the parquet so that
    # confidence scores are computed on the same ingredient set as the table.
    # Fallback: parse raw ingredients_text through the ETL cleaning pipeline.
    ingredient_tokens = _load_product_ingredient_tags(code) if code and code != "N/A" else []

    with st.spinner("Analizando ingredientes..."):
        analyzer = _get_analyzer()
        result = analyzer.analyze(
            product_code=code or "N/A",
            product_name=name or "Producto sin nombre",
            ingredients_text=ingredients,
            declared_allergens=allergens,
            ingredient_tokens=ingredient_tokens or None,
        )
    _render_result(result, allergens)


# ── Trigger: form submit ──────────────────────────────────────────────────────
_trigger_analyze_now = st.session_state.pop("analyze_now", False)

if submitted:
    declared_set = {k for k, v in declared_checkboxes.items() if v}
    with _result_col:
        _run_analysis(product_code.strip(), product_name.strip(), ingredients_text, declared_set)

# ── Trigger: "Analizar 🔬" button in search results ──────────────────────────
elif _trigger_analyze_now:
    declared_set = st.session_state.get("prefill_allergens", set())
    with _result_col:
        _run_analysis(
            st.session_state.get("prefill_code", ""),
            st.session_state.get("prefill_name", ""),
            st.session_state.get("prefill_ingredients", ""),
            declared_set,
        )

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    with _result_col:
        st.markdown(
            '<div class="empty-state-lab">'
            '<div class="es-icon">🏷️</div>'
            '<div class="es-title">Selecciona un producto o introduce los datos</div>'
            '<div class="es-hint">Usa la tabla superior o el buscador para cargar un producto,<br>'
            'o introduce manualmente los ingredientes en el formulario.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
