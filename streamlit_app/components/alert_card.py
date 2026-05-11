import streamlit as st
import pandas as pd
from utils.data_loader import ALLERGEN_EMOJIS, ALLERGEN_NAMES_ES
from utils.styles import badge


def render_alert_card(alert: pd.Series | dict, key: str = "") -> None:
    allergen = alert['allergen']
    certainty = alert['certainty']
    css_class = "alert-certain" if certainty == 'CERTAIN' else "alert-possible"
    icon = ALLERGEN_EMOJIS.get(allergen, '⚠️')
    name = ALLERGEN_NAMES_ES.get(allergen, allergen)
    score = float(alert.get('match_score', 0))
    confidence = alert.get('match_confidence', '')

    header_html = (
        f'<div class="{css_class}">'
        f'<b>{icon} {name.upper()}</b>&nbsp;&nbsp;'
        f'{badge(certainty, certainty.lower())}'
        f'<br/><small style="color:#64748B">'
        f'Ingrediente: <b>{alert.get("source_ingredient", "")}</b> → '
        f'{alert.get("foodex2_name", "")} '
        f'(confianza {confidence}, score {score:.2f})'
        f'</small>'
        f'<br/><div class="explanation-box">'
        f'💬 {alert.get("explanation", "")}'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_alert_list(alerts_df: pd.DataFrame, product_code: str) -> None:
    product_alerts = alerts_df[alerts_df['product_code'].astype(str) == str(product_code)]
    if product_alerts.empty:
        st.success("✅ Sin alertas de alérgenos no declarados")
        return
    for i, (_, row) in enumerate(product_alerts.iterrows()):
        render_alert_card(row, key=f"alert_{product_code}_{i}")
