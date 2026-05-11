"""HTML generators for the FoodEx2 Coding Assistant UI (v2 lab theme)."""
import re

from utils.data_loader import ALLERGEN_EMOJIS, ALLERGEN_NAMES_ES
from coding_assistant.models import Confidence, ReviewStatus

_STATUS_CSS = {
    ReviewStatus.AUTO_APPROVED:   "auto-approved",
    ReviewStatus.NEEDS_REVIEW:    "needs-review",
    ReviewStatus.MANUAL_REQUIRED: "manual-required",
}
_STATUS_LABEL = {
    ReviewStatus.AUTO_APPROVED:   "AUTO APROBADO ✓",
    ReviewStatus.NEEDS_REVIEW:    "REVISIÓN NECESARIA ⚠",
    ReviewStatus.MANUAL_REQUIRED: "REVISIÓN MANUAL ✗",
}
_CONF_CSS = {
    Confidence.HIGH:     "high",
    Confidence.MEDIUM:   "medium",
    Confidence.LOW:      "low",
    Confidence.NO_MATCH: "no-match",
}
_CONF_COLOR = {
    Confidence.HIGH:     "#16A34A",
    Confidence.MEDIUM:   "#D97706",
    Confidence.LOW:      "#94A3B8",
    Confidence.NO_MATCH: "#DC2626",
}


def highlight_variant(original: str, variant: str, exact: bool = False) -> str:
    """Wrap first occurrence of variant inside original with a highlight mark."""
    if not variant or not original:
        return original
    css_class = "exact-hl" if exact else "variant-hl"
    pattern = re.compile(re.escape(variant.strip()), re.IGNORECASE)
    highlighted, n = pattern.subn(
        lambda m: f'<mark class="{css_class}">{m.group()}</mark>',
        original,
        count=1,
    )
    return highlighted if n else original


def product_summary_html(result, declared_set: set) -> str:
    """Full product summary card: title, status badge, confidence bar, KPI chips, allergen chips."""
    from collections import Counter
    counts = Counter(m.confidence for m in result.ingredients)
    n_high = counts[Confidence.HIGH]
    n_med  = counts[Confidence.MEDIUM]
    n_low  = counts[Confidence.LOW]
    n_none = counts[Confidence.NO_MATCH]

    status_css   = _STATUS_CSS[result.review_status]
    status_label = _STATUS_LABEL[result.review_status]
    score        = result.confidence_score

    if score >= 0.70:
        bar_color   = "#16A34A"
        score_color = "#16A34A"
    elif score >= 0.50:
        bar_color   = "#D97706"
        score_color = "#D97706"
    else:
        bar_color   = "#DC2626"
        score_color = "#DC2626"

    # Collect unique detected allergens (preserve first-seen order)
    seen: set = set()
    detected: list = []
    for m in result.ingredients:
        for a, cert in m.allergens:
            if a not in seen:
                seen.add(a)
                detected.append((a, cert))

    allergen_chips = ""
    for a, cert in detected:
        emoji = ALLERGEN_EMOJIS.get(a, "")
        name  = ALLERGEN_NAMES_ES.get(a, a)
        if a not in declared_set:
            cls = "chip-discrepancy"
        elif cert == "CERTAIN":
            cls = "chip-detected-certain"
        else:
            cls = "chip-detected-possible"
        allergen_chips += f'<span class="{cls}">{emoji} {name}</span>'

    allergen_section = ""
    if allergen_chips:
        allergen_section = (
            '<div class="chip-section-label">Alérgenos detectados</div>'
            f'<div class="allergen-grid">{allergen_chips}</div>'
        )

    conf_bar = (
        '<div class="conf-bar-wrap">'
        '<div class="conf-bar-track">'
        f'<div class="conf-bar-fill" style="width:{score*100:.1f}%;background:{bar_color}"></div>'
        '</div>'
        f'<span class="conf-bar-label" style="color:{score_color}">{score:.1%}</span>'
        '</div>'
    )

    kpis = (
        '<div class="product-summary-kpis">'
        f'<span class="kpi-chip kpi-chip-high">✓ {n_high} HIGH</span>'
        f'<span class="kpi-chip kpi-chip-medium">~ {n_med} MEDIUM</span>'
        f'<span class="kpi-chip kpi-chip-low">↓ {n_low} LOW</span>'
        f'<span class="kpi-chip kpi-chip-nomatch">✗ {n_none} NO_MATCH</span>'
        '</div>'
    )

    return (
        '<div class="product-summary">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">'
        f'<span class="product-summary-title">{result.product_name}</span>'
        f'<span class="badge-{status_css}" style="flex-shrink:0;font-size:0.78em">{status_label}</span>'
        '</div>'
        f'<div class="product-summary-meta">{result.product_code} · {result.processed_at[:19]}</div>'
        f'{conf_bar}{kpis}{allergen_section}'
        '</div>'
    )


def allergen_grid_html(result, declared_set: set) -> str:
    """Allergen profile section: declared+detected chips and discrepancy chips."""
    seen: set = set()
    detected: list = []
    for m in result.ingredients:
        for a, cert in m.allergens:
            if a not in seen:
                seen.add(a)
                detected.append((a, cert))

    detected_keys = {a for a, _ in detected}

    # Declared allergens (shown as detected if also found, else declared-only)
    declared_chips = ""
    for a in declared_set:
        emoji = ALLERGEN_EMOJIS.get(a, "")
        name  = ALLERGEN_NAMES_ES.get(a, a)
        certs = [cert for da, cert in detected if da == a]
        if certs:
            cls = "chip-detected-certain" if certs[0] == "CERTAIN" else "chip-detected-possible"
        else:
            cls = "chip-declared"
        declared_chips += f'<span class="{cls}">{emoji} {name}</span>'

    # Detected but NOT declared (discrepancy)
    undeclared = [(a, cert) for a, cert in detected if a not in declared_set]
    undeclared_chips = "".join(
        f'<span class="chip-discrepancy">⚠ {ALLERGEN_EMOJIS.get(a,"")} {ALLERGEN_NAMES_ES.get(a,a)}</span>'
        for a, cert in undeclared
    )

    parts: list[str] = []

    if declared_chips:
        parts.append(
            '<div class="chip-section-label">Declarados / Detectados</div>'
            f'<div class="allergen-grid">{declared_chips}</div>'
        )

    if undeclared_chips:
        parts.append(
            '<div class="chip-section-label-warn">⚠ Detectados sin declarar</div>'
            f'<div class="allergen-grid">{undeclared_chips}</div>'
        )
    elif declared_set and detected:
        parts.append(
            '<p style="color:#16A34A;font-size:0.88em;margin:6px 0">'
            '✓ Todos los alérgenos detectados están declarados.</p>'
        )

    if not parts:
        if not detected:
            return (
                '<p style="color:#64748B;font-size:0.88em;margin:4px 0">'
                'No se detectaron alérgenos en los ingredientes codificados.</p>'
            )
        # detected but no declared_set provided
        no_declared_chips = "".join(
            f'<span class="chip-detected-certain">{ALLERGEN_EMOJIS.get(a,"")} {ALLERGEN_NAMES_ES.get(a,a)}</span>'
            if cert == "CERTAIN"
            else f'<span class="chip-detected-possible">{ALLERGEN_EMOJIS.get(a,"")} {ALLERGEN_NAMES_ES.get(a,a)}</span>'
            for a, cert in detected
        )
        parts.append(
            '<div class="chip-section-label">Detectados</div>'
            f'<div class="allergen-grid">{no_declared_chips}</div>'
        )

    return "".join(parts)


def ingredient_card_html(m) -> str:
    """Compact ingredient card with variant highlight, code badge, allergen chips, flags."""
    conf_css = _CONF_CSS[m.confidence]

    # Name with optional RF variant highlight
    name_html = m.original_text
    if m.rf_matched_variant:
        norm = (m.normalized or m.original_text).lower().strip()
        if m.rf_matched_variant.lower().strip() != norm:
            name_html = highlight_variant(m.original_text, m.rf_matched_variant, m.exact_variant_match)
        elif m.exact_variant_match:
            name_html = f'<mark class="exact-hl">{m.original_text}</mark>'

    # Code + diagnostic line
    diag_parts = [f"score {m.final_score:.3f}", f"L{m.final_level}"]
    if m.matching_mode:
        diag_parts.append(m.matching_mode)
    if m.rf_score is not None and m.rf_score > 0:
        diag_parts.append(f"rf {m.rf_score:.0f}")
    if m.branch_penalty is not None and abs(m.branch_penalty) > 1e-4:
        diag_parts.append(f"pen {m.branch_penalty:.3f}")
    diag_str = " · ".join(diag_parts)

    if m.final_code:
        code_line = (
            '<div class="ic-meta">'
            f'→ <span class="ic-code">{m.final_code}</span> '
            f'{m.final_name} '
            f'<span style="opacity:.55;font-size:.9em">({diag_str})</span>'
            '</div>'
        )
    else:
        code_line = (
            '<div class="ic-meta" style="color:#94A3B8">'
            f'→ Sin match FoodEx2'
            f'<span style="opacity:.6;font-size:.9em"> ({diag_str})</span>'
            '</div>'
        )

    # Allergen chips
    allergen_html = ""
    if m.allergens:
        chips = "".join(
            f'<span class="allergen-chip">'
            f'{ALLERGEN_EMOJIS.get(a,"")} {ALLERGEN_NAMES_ES.get(a,a)}'
            f'{"?" if cert == "POSSIBLE" else ""}'
            f'</span>'
            for a, cert in m.allergens
        )
        allergen_html = f'<div class="ic-meta">Alérgenos: {chips}</div>'

    # Review flags
    flags_html = ""
    if m.review_flags:
        flag_str = " · ".join(f.value for f in m.review_flags)
        flags_html = f'<div class="ic-meta" style="color:#92400E">⚑ {flag_str}</div>'

    return (
        f'<div class="ingredient-card-v2 ic-{conf_css}">'
        '<div class="ic-header">'
        f'<span class="ic-name">{name_html}</span>'
        f'<span class="badge-{conf_css}" style="flex-shrink:0;font-size:0.75em">{m.confidence.value}</span>'
        '</div>'
        f'{code_line}{allergen_html}{flags_html}'
        '</div>'
    )


def foodex2_tree_html(path: list, current_code: str) -> str:
    """Vertical hierarchy tree with indented rows and highlighted current node."""
    if not path:
        return ""
    rows = []
    for lvl, code, name in path:
        is_current = (code == current_code)
        cls = "foodex2-tree-row current" if is_current else "foodex2-tree-row"
        indent = (lvl - 1) * 18
        rows.append(
            f'<div class="{cls}" style="padding-left:{8 + indent}px">'
            f'<span class="tree-level">L{lvl}</span>'
            f'<span class="tree-code">{code}</span>'
            f'<span>{name}</span>'
            '</div>'
        )
    return f'<div class="foodex2-tree">{"".join(rows)}</div>'


def facets_html(facets: list) -> str:
    """Inline facet tags (category label + name + code)."""
    if not facets:
        return ""
    tags = []
    for f in facets:
        label = f.get("facet_label") or f.get("facet_cat", "")
        name  = f.get("name", "")
        fcode = f.get("code") or f.get("facet_code", "")
        cat   = f.get("facet_cat", "")
        display_label = label or cat
        tags.append(
            '<span class="facet-tag">'
            f'<span class="facet-tag-label">{display_label}:</span> '
            f'{name} '
            f'<span class="facet-tag-code">{fcode}</span>'
            '</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:3px;margin:4px 0">'
        + "".join(tags)
        + '</div>'
    )
