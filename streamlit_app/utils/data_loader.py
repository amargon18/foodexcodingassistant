import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "allergenscan" / "data" / "allergenscan_results"

ALLERGEN_EMOJIS = {
    'gluten': '🌾', 'milk': '🥛', 'eggs': '🥚', 'fish': '🐟',
    'crustaceans': '🦐', 'molluscs': '🦪', 'peanuts': '🥜',
    'tree_nuts': '🌰', 'soy': '🫘', 'celery': '🥬',
    'mustard': '🌡', 'sesame': '⚪', 'sulphites': '🧪', 'lupin': '🌸',
}

ALLERGEN_NAMES_ES = {
    'gluten': 'Gluten', 'milk': 'Lácteos', 'eggs': 'Huevos',
    'fish': 'Pescado', 'crustaceans': 'Crustáceos', 'molluscs': 'Moluscos',
    'peanuts': 'Cacahuetes', 'tree_nuts': 'Frutos secos', 'soy': 'Soja',
    'celery': 'Apio', 'mustard': 'Mostaza', 'sesame': 'Sésamo',
    'sulphites': 'Sulfitos', 'lupin': 'Altramuces',
}

ALLERGEN_COLORS = {
    'gluten': '#D97706', 'milk': '#3B82F6', 'eggs': '#FBBF24',
    'fish': '#06B6D4', 'crustaceans': '#F97316', 'molluscs': '#8B5CF6',
    'peanuts': '#A16207', 'tree_nuts': '#92400E', 'soy': '#65A30D',
    'celery': '#22C55E', 'mustard': '#EAB308', 'sesame': '#78716C',
    'sulphites': '#7C3AED', 'lupin': '#EC4899',
}


@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame | None:
    path = RESULTS_DIR / "allergenscan_products.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['undeclared_certain'] = df['undeclared_certain'].fillna(0).astype(int)
    df['undeclared_possible'] = df['undeclared_possible'].fillna(0).astype(int)
    df['undeclared_count'] = df['undeclared_count'].fillna(0).astype(int)
    df['declared_allergens'] = df['declared_allergens'].fillna('').astype(str)
    df['declared_traces'] = df['declared_traces'].fillna('').astype(str)
    df['undeclared_allergens'] = df['undeclared_allergens'].fillna('').astype(str)
    return df


@st.cache_data(ttl=300)
def load_alerts() -> pd.DataFrame | None:
    path = RESULTS_DIR / "allergenscan_alerts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['match_score'] = df['match_score'].fillna(0.0)
    df['allergen_label'] = df['allergen'].map(
        lambda a: f"{ALLERGEN_EMOJIS.get(a, '')} {ALLERGEN_NAMES_ES.get(a, a)}"
    )
    return df


@st.cache_data(ttl=300)
def load_comparison() -> pd.DataFrame | None:
    path = RESULTS_DIR / "allergenscan_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_metrics_text() -> str | None:
    path = RESULTS_DIR / "allergenscan_metrics.txt"
    if not path.exists():
        return None
    return path.read_text(encoding='utf-8')


def data_available() -> bool:
    return (RESULTS_DIR / "allergenscan_products.csv").exists()


def show_no_data_message():
    st.error("No se han encontrado resultados de AllergenScan.")
    st.info(
        "Para generar los resultados, ejecuta el Step 4 del pipeline:\n\n"
        "```bash\n"
        "python scripts/run_pipeline.py --skip-products --skip-terms --skip-matching\n"
        "```\n\n"
        "Los CSV se guardarán en `allergenscan/data/allergenscan_results/`."
    )


def product_status(row: pd.Series) -> str:
    if row['undeclared_certain'] > 0:
        return 'certain'
    if row['undeclared_possible'] > 0:
        return 'possible'
    return 'clean'


STATUS_ICON = {'certain': '🔴', 'possible': '🟡', 'clean': '🟢'}
STATUS_LABEL = {'certain': 'Alerta CERTAIN', 'possible': 'Alerta POSSIBLE', 'clean': 'Sin alertas'}
