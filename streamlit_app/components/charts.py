import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import ALLERGEN_NAMES_ES, ALLERGEN_EMOJIS, ALLERGEN_COLORS

CERTAIN_COLOR = "#EF4444"
POSSIBLE_COLOR = "#F59E0B"
CLEAN_COLOR = "#10B981"
PRIMARY_COLOR = "#2563EB"


def allergen_bar_chart(alerts_df: pd.DataFrame) -> go.Figure:
    counts = alerts_df['allergen'].value_counts().reset_index()
    counts.columns = ['allergen', 'count']
    counts['label'] = counts['allergen'].map(
        lambda a: f"{ALLERGEN_EMOJIS.get(a, '')} {ALLERGEN_NAMES_ES.get(a, a)}"
    )
    counts['color'] = counts['allergen'].map(lambda a: ALLERGEN_COLORS.get(a, PRIMARY_COLOR))
    counts = counts.sort_values('count', ascending=True)

    fig = go.Figure(go.Bar(
        x=counts['count'],
        y=counts['label'],
        orientation='h',
        marker_color=counts['color'].tolist(),
        text=counts['count'],
        textposition='outside',
        hovertemplate='%{y}: %{x} alertas<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(300, len(counts) * 36),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
    return fig


def certainty_donut(alerts_df: pd.DataFrame) -> go.Figure:
    counts = alerts_df['certainty'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [CERTAIN_COLOR if l == 'CERTAIN' else POSSIBLE_COLOR for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} alertas (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=260,
        paper_bgcolor='white',
    )
    return fig


def product_status_donut(products_df: pd.DataFrame) -> go.Figure:
    certain = int((products_df['undeclared_certain'] > 0).sum())
    possible = int(((products_df['undeclared_certain'] == 0) & (products_df['undeclared_possible'] > 0)).sum())
    clean = int(len(products_df) - certain - possible)

    fig = go.Figure(go.Pie(
        labels=['Con CERTAIN', 'Solo POSSIBLE', 'Sin alertas'],
        values=[certain, possible, clean],
        hole=0.55,
        marker_colors=[CERTAIN_COLOR, POSSIBLE_COLOR, CLEAN_COLOR],
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} productos<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=260,
        paper_bgcolor='white',
    )
    return fig


def score_histogram(alerts_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        alerts_df,
        x='match_score',
        color='certainty',
        color_discrete_map={'CERTAIN': CERTAIN_COLOR, 'POSSIBLE': POSSIBLE_COLOR},
        nbins=30,
        barmode='overlay',
        labels={'match_score': 'Score de matching', 'count': 'Frecuencia', 'certainty': 'Certeza'},
        opacity=0.8,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=280,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
    return fig


def per_allergen_certainty_bars(alerts_df: pd.DataFrame) -> go.Figure:
    grouped = alerts_df.groupby(['allergen', 'certainty']).size().unstack(fill_value=0).reset_index()
    grouped['label'] = grouped['allergen'].map(
        lambda a: f"{ALLERGEN_EMOJIS.get(a, '')} {ALLERGEN_NAMES_ES.get(a, a)}"
    )
    grouped = grouped.sort_values('CERTAIN' if 'CERTAIN' in grouped.columns else grouped.columns[1], ascending=False)

    fig = go.Figure()
    if 'CERTAIN' in grouped.columns:
        fig.add_trace(go.Bar(
            name='CERTAIN',
            x=grouped['label'],
            y=grouped['CERTAIN'],
            marker_color=CERTAIN_COLOR,
        ))
    if 'POSSIBLE' in grouped.columns:
        fig.add_trace(go.Bar(
            name='POSSIBLE',
            x=grouped['label'],
            y=grouped['POSSIBLE'],
            marker_color=POSSIBLE_COLOR,
        ))
    fig.update_layout(
        barmode='stack',
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=340,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_tickangle=-30,
    )
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
    return fig


def confidence_pie(alerts_df: pd.DataFrame) -> go.Figure:
    counts = alerts_df['match_confidence'].value_counts()
    color_map = {'HIGH': '#10B981', 'MEDIUM': '#F59E0B', 'LOW': '#EF4444', 'NO_MATCH': '#94A3B8'}
    colors = [color_map.get(l, PRIMARY_COLOR) for l in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=colors,
        textinfo='percent+label',
        hovertemplate='%{label}: %{value}<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        paper_bgcolor='white',
        showlegend=False,
    )
    return fig
