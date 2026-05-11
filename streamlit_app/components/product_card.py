import streamlit as st
import pandas as pd
from utils.data_loader import (
    ALLERGEN_EMOJIS, ALLERGEN_NAMES_ES, STATUS_ICON, STATUS_LABEL,
    product_status,
)
from utils.styles import allergen_chip, badge


def render_product_card(row: pd.Series, on_detail_click=None, key: str = "") -> None:
    status = product_status(row)
    icon = STATUS_ICON[status]
    border_class = f"product-card product-card-{status}"

    code = str(row['product_code'])
    name = str(row['product_name'])[:80]
    n_alerts = int(row['undeclared_count'])
    n_certain = int(row['undeclared_certain'])
    n_possible = int(row['undeclared_possible'])

    undeclared_str = str(row.get('undeclared_allergens', ''))
    undeclared_list = [a.strip() for a in undeclared_str.split(',') if a.strip()] if undeclared_str else []

    declared_str = str(row.get('declared_allergens', ''))
    declared_list = [a.strip() for a in declared_str.split(',') if a.strip()] if declared_str else []

    # Build alert summary text
    if n_alerts > 0:
        alert_parts = []
        if n_certain:
            alert_parts.append(f"{n_certain} CERTAIN")
        if n_possible:
            alert_parts.append(f"{n_possible} POSSIBLE")
        alert_summary = f"⚠️ {n_alerts} alerta(s): {', '.join(alert_parts)}"
        undeclared_chips = " ".join(
            allergen_chip(f"{ALLERGEN_EMOJIS.get(a,'')} {ALLERGEN_NAMES_ES.get(a,a)}")
            for a in undeclared_list
        )
    else:
        alert_summary = "✅ Sin alertas de alérgenos no declarados"
        undeclared_chips = ""

    declared_chips = " ".join(
        allergen_chip(f"{ALLERGEN_EMOJIS.get(a,'')} {ALLERGEN_NAMES_ES.get(a,a)}")
        for a in declared_list if a
    ) if declared_list else '<span style="color:#94A3B8">Ninguno</span>'

    html = (
        f'<div class="{border_class}">'
        f'<b>{icon} {name}</b> '
        f'<span style="color:#94A3B8;font-size:0.85rem">{code}</span><br/>'
        f'<span>{alert_summary}</span>'
    )
    if undeclared_chips:
        html += f'<br/><span style="font-size:0.82rem">No declarados: {undeclared_chips}</span>'
    html += (
        f'<br/><span style="font-size:0.82rem;color:#64748B">'
        f'Declarados: {declared_chips}</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if on_detail_click is not None:
        if st.button("Ver detalles →", key=f"detail_{key}_{code}"):
            on_detail_click(code)


def render_product_detail(
    product_row: pd.Series,
    alerts_df: pd.DataFrame,
    matching_snippet: pd.DataFrame | None = None,
) -> None:
    code = str(product_row['product_code'])
    name = str(product_row['product_name'])
    status = product_status(product_row)
    icon = STATUS_ICON[status]

    st.markdown(f"## {icon} {name}")
    st.caption(f"Código: `{code}` · Idioma: {product_row.get('lang', 'en')}")
    st.divider()

    # Declared allergens
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏷️ Alérgenos declarados**")
        declared = str(product_row.get('declared_allergens', ''))
        items = [a.strip() for a in declared.split(',') if a.strip()]
        if items:
            chips = " ".join(
                allergen_chip(f"{ALLERGEN_EMOJIS.get(a,'')} {ALLERGEN_NAMES_ES.get(a,a)}")
                for a in items
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Ninguno")

    with col2:
        st.markdown("**🔍 Trazas declaradas**")
        traces = str(product_row.get('declared_traces', ''))
        items = [a.strip() for a in traces.split(',') if a.strip()]
        if items:
            chips = " ".join(
                allergen_chip(f"{ALLERGEN_EMOJIS.get(a,'')} {ALLERGEN_NAMES_ES.get(a,a)}")
                for a in items
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Ninguna")

    st.divider()

    # Alerts
    product_alerts = alerts_df[alerts_df['product_code'].astype(str) == code]
    if product_alerts.empty:
        st.success("✅ No se detectaron alérgenos no declarados en este producto.")
    else:
        from components.alert_card import render_alert_card
        st.markdown(f"### ⚠️ Alertas detectadas ({len(product_alerts)})")
        for i, (_, row) in enumerate(product_alerts.iterrows()):
            render_alert_card(row, key=f"detail_{code}_{i}")

    # Matching stats
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Ingredientes totales", int(product_row.get('total_ingredients', 0)))
    col2.metric("Con match", int(product_row.get('matched_ingredients', 0)))
    col3.metric("Matches HIGH", int(product_row.get('high_confidence_matches', 0)))
