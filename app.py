"""Streamlit dashboard for E-commerce Business Intelligence.

Provides interactive BI dashboards, customer segmentation,
ML model comparison, SQL analysis, data exploration, and an API playground.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.model_selection import train_test_split

from utils import (
    SQL_QUERIES,
    build_preprocessor,
    compare_models,
    compute_rfm,
    create_sqlite_db,
    engineer_features,
    get_classifiers,
    get_param_grids,
    hypothesis_tests,
    join_sales_with_inventory,
    load_and_clean_data,
    load_model_pipeline,
    save_model_pipeline,
    segment_customers,
    train_and_evaluate,
)

st.set_page_config(
    page_title="E-commerce BI Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

PRIMARY = "#1a73e8"
SECONDARY = "#0d47a1"
BG_DARK = "#0f0f11"
BG_CARD = "#1a1a1f"
BG_HOVER = "#22222a"
TEXT_PRIMARY = "#e8e8ed"
TEXT_SECONDARY = "#9aa0a6"
ACCENT_GREEN = "#34a853"
ACCENT_ORANGE = "#fbbc04"
ACCENT_RED = "#ea4335"
BORDER = "#2a2a32"

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}

    .stApp {{
        background: {BG_DARK};
        color: {TEXT_PRIMARY};
    }}

    /* ---- Typography ---- */
    .main-title {{
        font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em;
        color: {TEXT_PRIMARY}; margin-bottom: 0.25rem;
    }}
    .main-subtitle {{
        font-size: 0.875rem; color: {TEXT_SECONDARY};
        margin-bottom: 2rem; font-weight: 400;
    }}
    .section-title {{
        font-size: 1.25rem; font-weight: 600; color: {TEXT_PRIMARY};
        margin: 1.5rem 0 1rem 0; letter-spacing: -0.01em;
    }}
    .subsection-title {{
        font-size: 1rem; font-weight: 500; color: {TEXT_PRIMARY};
        margin: 1rem 0 0.5rem 0;
    }}

    /* ---- Metric cards ---- */
    .metric-card {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 1.25rem 1.5rem;
        transition: border-color 0.2s, transform 0.15s;
    }}
    .metric-card:hover {{
        border-color: {PRIMARY}; transform: translateY(-1px);
    }}
    .metric-label {{
        font-size: 0.75rem; font-weight: 500; text-transform: uppercase;
        letter-spacing: 0.05em; color: {TEXT_SECONDARY}; margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 1.625rem; font-weight: 700; color: {TEXT_PRIMARY};
        letter-spacing: -0.02em;
    }}
    .metric-delta {{ font-size: 0.75rem; color: {ACCENT_GREEN}; }}

    /* ---- Stat card (info boxes) ---- */
    .stat-card {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 8px; padding: 0.875rem 1.125rem;
        margin-bottom: 0.5rem;
    }}
    .stat-card .stat-label {{
        font-size: 0.75rem; color: {TEXT_SECONDARY};
    }}
    .stat-card .stat-value {{
        font-size: 1rem; font-weight: 600; color: {TEXT_PRIMARY};
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent; border-bottom: 1px solid {BORDER};
        gap: 0; padding: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {TEXT_SECONDARY};
        font-size: 0.8125rem; font-weight: 500; padding: 0.75rem 1.25rem;
        border: none; border-bottom: 2px solid transparent;
        transition: color 0.15s, border-color 0.15s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {TEXT_PRIMARY}; background: transparent;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY}; background: transparent;
        border-bottom: 2px solid {PRIMARY};
    }}

    /* ---- DataFrames ---- */
    .stDataFrame {{
        border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden;
    }}
    .stDataFrame th {{
        background: {BG_CARD} !important; color: {TEXT_SECONDARY} !important;
        font-weight: 500; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.05em; border-bottom: 1px solid {BORDER};
    }}
    .stDataFrame td {{
        background: transparent !important; color: {TEXT_PRIMARY};
        border-bottom: 1px solid {BORDER}; font-size: 0.8125rem;
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        background: {PRIMARY}; color: white; border: none;
        border-radius: 6px; padding: 0.5rem 1.25rem;
        font-size: 0.8125rem; font-weight: 500;
        transition: background 0.15s, transform 0.1s;
    }}
    .stButton > button:hover {{
        background: {SECONDARY}; transform: translateY(-0.5px);
    }}
    .stButton > button:active {{ transform: translateY(0); }}

    /* ---- Sidebar ---- */
    .css-1d391kg, .css-12oz5g7 {{
        background: {BG_CARD} !important;
    }}
    .sidebar-section {{
        padding: 1rem 0; border-bottom: 1px solid {BORDER};
    }}
    .sidebar-section:last-child {{ border-bottom: none; }}

    /* ---- Info/Warning/Success boxes ---- */
    .info-box {{
        background: rgba(26,115,232,0.08); border: 1px solid rgba(26,115,232,0.2);
        border-radius: 8px; padding: 0.875rem 1rem; margin: 0.75rem 0;
        font-size: 0.8125rem; color: {TEXT_SECONDARY};
    }}
    .info-box strong {{ color: {TEXT_PRIMARY}; }}
    .success-box {{
        background: rgba(52,168,83,0.08); border: 1px solid rgba(52,168,83,0.2);
        border-radius: 8px; padding: 0.875rem 1rem; margin: 0.75rem 0;
        font-size: 0.8125rem;
    }}
    .warning-box {{
        background: rgba(251,188,4,0.08); border: 1px solid rgba(251,188,4,0.2);
        border-radius: 8px; padding: 0.875rem 1rem; margin: 0.75rem 0;
        font-size: 0.8125rem;
    }}
    .danger-box {{
        background: rgba(234,67,53,0.08); border: 1px solid rgba(234,67,53,0.2);
        border-radius: 8px; padding: 0.875rem 1rem; margin: 0.75rem 0;
        font-size: 0.8125rem;
    }}

    /* ---- Code blocks ---- */
    .code-block {{
        background: #121216; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 1rem; font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8125rem; overflow-x: auto; color: {TEXT_SECONDARY};
    }}

    /* ---- Divider ---- */
    .divider {{
        border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0;
    }}

    /* ---- Key-value inline ---- */
    .kv {{ font-size: 0.8125rem; color: {TEXT_SECONDARY}; margin-bottom: 0.25rem; }}
    .kv strong {{ color: {TEXT_PRIMARY}; }}

    /* ---- Prediction result ---- */
    .pred-result {{
        border-radius: 12px; padding: 1.5rem; text-align: center;
        margin-top: 1rem;
    }}
    .pred-result.low {{ background: rgba(52,168,83,0.1); border: 1px solid {ACCENT_GREEN}; }}
    .pred-result.moderate {{ background: rgba(251,188,4,0.1); border: 1px solid {ACCENT_ORANGE}; }}
    .pred-result.high {{ background: rgba(234,67,53,0.1); border: 1px solid {ACCENT_RED}; }}
    .pred-value {{ font-size: 2rem; font-weight: 700; }}
    .pred-label {{ font-size: 0.8125rem; color: {TEXT_SECONDARY}; margin-top: 0.25rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data / model helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> Dict[str, pd.DataFrame]:
    """Load and clean all datasets (cached)."""
    return load_and_clean_data()


@st.cache_data
def get_joined_data() -> pd.DataFrame:
    """Return sales joined with inventory data (cached)."""
    dfs = load_data()
    return join_sales_with_inventory(dfs['sales'], dfs['inventory'])


@st.cache_data
def get_features() -> pd.DataFrame:
    """Return engineered features (cached)."""
    dfs = load_data()
    return engineer_features(dfs['sales'])


@st.cache_data
def get_rfm_segments() -> Tuple[pd.DataFrame, str]:
    """Compute RFM and K-means segmentation (cached)."""
    dfs = load_data()
    rfm = compute_rfm(dfs['sales'])
    segmented, _ = segment_customers(rfm, n_clusters=4)
    summary = segmented.groupby('Segment_Label').agg(
        Count=('OrderID', 'nunique'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
    ).round(1).to_html(classes='segment-table')
    return segmented, summary


@st.cache_resource
def get_sql_connection():
    """Create in-memory SQLite DB (cached resource)."""
    dfs = load_data()
    return create_sqlite_db(dfs)


@st.cache_resource
def train_models() -> Dict[str, object]:
    """Train and compare ML models with caching."""
    fe = get_features()
    target = fe['Cancelled']
    preprocessor = build_preprocessor()
    numeric_cols = ['Qty', 'Amount', 'Month', 'IsWeekend', 'PricePerUnit',
                    'HasPromotion', 'IsHighValue', 'Revenue_7d_MA']
    cat_cols = ['Fulfilment', 'ship-service-level', 'B2B']

    X = fe[numeric_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    valid = target.notna() & X.notna().all(axis=1)
    X = X[valid]
    y = target[valid].astype(int)

    if y.nunique() < 2:
        return {'error': 'Not enough class variation in filtered data.'}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results_df = compare_models(X_train, y_train, X_test, y_test, use_gridsearch=True)

    best_name = results_df.iloc[0]['Model']
    classifiers = get_classifiers()
    grids = get_param_grids()
    best_clf = classifiers[best_name]
    best_result = train_and_evaluate(
        best_clf, X_train, y_train, X_test, y_test, grids.get(best_name)
    )

    return {
        'results_df': results_df,
        'best_model': best_result['model'],
        'best_metrics': best_result,
        'feature_importance': _get_feature_importance(best_result['model'], numeric_cols, cat_cols),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }


def _get_feature_importance(
    model: object, num_cols: List[str], cat_cols: List[str]
) -> pd.DataFrame:
    """Extract feature importance or coefficients from a trained model."""
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        names = num_cols + cat_cols
        return pd.DataFrame({'Feature': names[:len(coefs)], 'Importance': coefs}).sort_values(
            'Importance', ascending=False
        )
    if hasattr(model, 'feature_importances_'):
        names = num_cols + cat_cols
        return pd.DataFrame({
            'Feature': names[:len(model.feature_importances_)],
            'Importance': model.feature_importances_,
        }).sort_values('Importance', ascending=False)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def render_bi_dashboard(filtered: pd.DataFrame, full_data: pd.DataFrame) -> None:
    """Render the Business Intelligence dashboard tab.

    Args:
        filtered: date-filtered sales DataFrame.
        full_data: unfiltered sales DataFrame.
    """
    rev = filtered['Amount'].sum()
    orders = filtered['Order ID'].nunique()
    aov = rev / orders if orders > 0 else 0
    cancel_mask = filtered['Status'].str.contains('cancel', case=False, na=False)
    cancel_rate = cancel_mask.sum() / len(filtered) if len(filtered) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Revenue</div>'
            f'<div class="metric-value">&#8377;{rev:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Orders</div>'
            f'<div class="metric-value">{orders:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Order Value</div>'
            f'<div class="metric-value">&#8377;{aov:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Cancellation Rate</div>'
            f'<div class="metric-value">{cancel_rate:.1%}</div></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        monthly = filtered.groupby(filtered['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly['Date'] = monthly['Date'].astype(str)
        fig = px.line(
            monthly, x='Date', y='Amount',
            title='Monthly Revenue Trend',
            color_discrete_sequence=[PRIMARY],
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        cats = filtered.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=cats.index, y=cats.values,
            title='Top Categories by Revenue',
            color_discrete_sequence=[PRIMARY],
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    c1, c2 = st.columns(2)
    with c1:
        states = filtered.groupby('ship-state')['Amount'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=states.index, y=states.values,
            title='Top States by Revenue',
            color_discrete_sequence=[ACCENT_GREEN],
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        status_dist = filtered['Status'].value_counts()
        fig = px.pie(
            values=status_dist.values, names=status_dist.index,
            title='Order Status Distribution',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hypothesis Testing</div>', unsafe_allow_html=True)
    tests = hypothesis_tests(filtered)
    for name, result in tests.items():
        sig = result.get('significant', False)
        icon = "Significant" if sig else "Not significant"
        box_cls = "success-box" if sig else "info-box"
        st.markdown(
            f'<div class="{box_cls}"><strong>{name}</strong><br/>'
            f'p-value: {result.get("p-value", "N/A")} &mdash; {icon}'
            f'{"<br/>B2B mean AOV: &#8377;{:,}".format(result.get("B2B mean AOV", 0)) if "B2B mean AOV" in result else ""}'
            f'{" | B2C mean AOV: &#8377;{:,}".format(result.get("B2C mean AOV", 0)) if "B2C mean AOV" in result else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Enrichment</div>', unsafe_allow_html=True)
    joined = get_joined_data()
    st.markdown(
        f'<div class="info-box"><strong>Sales &times; Inventory Join</strong><br/>'
        f'{len(joined):,} total rows &mdash; '
        f'{joined["Total_Stock"].notna().sum():,} rows with stock data</div>',
        unsafe_allow_html=True,
    )


def render_segmentation() -> None:
    """Render the customer segmentation tab."""
    st.markdown('<div class="section-title">Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Customers are scored using <strong>RFM</strong> '
        '(Recency, Frequency, Monetary) and segmented via '
        '<strong>K-means clustering</strong> on log-transformed, standardised values.</div>',
        unsafe_allow_html=True,
    )

    segmented, _ = get_rfm_segments()

    seg_profile = segmented.groupby('Segment_Label').agg(
        Customers=('OrderID', 'nunique'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
    ).round(1).sort_values('Avg_Monetary', ascending=False)

    st.markdown('<div class="subsection-title">Segment Profiles</div>', unsafe_allow_html=True)
    st.dataframe(
        seg_profile.style.format({
            'Avg_Recency': '{:.0f}d',
            'Avg_Frequency': '{:.1f}',
            'Avg_Monetary': '\u20b9{:.0f}',
            'Total_Revenue': '\u20b9{:.0f}',
        }),
        use_container_width=True,
    )

    fig = px.scatter_3d(
        segmented.sample(min(2000, len(segmented))),
        x='Recency', y='Frequency', z='Monetary',
        color='Segment_Label',
        title='Customer Segments (log-transformed)',
        labels={'Recency': 'Recency (days)', 'Frequency': 'Order count', 'Monetary': 'Total spent'},
        template='plotly_dark',
        opacity=0.7,
    )
    fig.update_layout(
        plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
        font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
        title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(font=dict(size=10)),
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    c1, c2 = st.columns(2)
    with c1:
        seg_customers = segmented.groupby('Segment_Label').agg(
            Customers=('OrderID', 'nunique'),
            Revenue=('Monetary', 'sum'),
        ).sort_values('Revenue', ascending=False)
        fig = px.pie(
            seg_customers, values='Customers', names=seg_customers.index,
            title='Distribution: Customers by Segment',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        seg_customers['Revenue_Share'] = (
            seg_customers['Revenue'] / seg_customers['Revenue'].sum() * 100
        ).round(1)
        fig = px.pie(
            seg_customers, values='Revenue', names=seg_customers.index,
            title='Distribution: Revenue by Segment',
            color_discrete_sequence=px.colors.sequential.Greens_r,
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="subsection-title">Strategic Recommendations</div>', unsafe_allow_html=True)

    recommendations = [
        ("Champions / Loyal", "Reward with exclusive loyalty programmes and early access to new products."),
        ("Big Spenders", "Cross-sell and upsell premium products via targeted email campaigns."),
        ("Potential Loyalists", "Send personalised product recommendations to increase order frequency."),
        ("At Risk", "Launch win-back campaigns with discount incentives and re-engagement emails."),
        ("Needs Attention", "Gather feedback via surveys and offer support to re-engage."),
    ]
    for segment, advice in recommendations:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">{segment}</span><br/>'
            f'<span class="stat-value">{advice}</span></div>',
            unsafe_allow_html=True,
        )


def render_ml_comparison() -> None:
    """Render the ML model comparison tab."""
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Three classifiers are trained on engineered features '
        '(quantity, amount, price-per-unit, rolling revenue, fulfilment method, etc.) '
        'with <strong>3-fold cross-validation</strong> and <strong>GridSearchCV</strong> '
        'hyperparameter tuning. Performance is measured on a held-out 20% test set.</div>',
        unsafe_allow_html=True,
    )

    model_data = train_models()

    if 'error' in model_data:
        st.warning(model_data['error'])
        return

    results_df: pd.DataFrame = model_data['results_df']

    st.markdown('<div class="subsection-title">Performance Metrics</div>', unsafe_allow_html=True)
    st.dataframe(
        results_df.style.format({
            'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}',
            'F1': '{:.2%}', 'ROC-AUC': '{:.3f}',
        }),
        use_container_width=True,
    )

    melted = results_df.melt(
        id_vars='Model',
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    )
    fig = px.bar(
        melted, x='Model', y='value', color='variable', barmode='group',
        title='Cross-Metric Comparison',
        color_discrete_sequence=px.colors.sequential.Blues_r[1:],
        template='plotly_dark',
    )
    fig.update_layout(
        plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
        font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
        title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    best_name = results_df.iloc[0]['Model']
    best_auc = results_df.iloc[0]['ROC-AUC']
    st.markdown(
        f'<div class="success-box"><strong>Best Model</strong>: {best_name} &mdash; '
        f'ROC-AUC: {best_auc:.3f}</div>',
        unsafe_allow_html=True,
    )

    fi = model_data.get('feature_importance', pd.DataFrame())
    if not fi.empty:
        st.markdown('<div class="subsection-title">Feature Importance</div>', unsafe_allow_html=True)
        fig = px.bar(
            fi.head(10), x='Importance', y='Feature', orientation='h',
            title=f'Top Predictive Features &mdash; {best_name}',
            color_discrete_sequence=[PRIMARY],
            template='plotly_dark',
        )
        fig.update_layout(
            plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
            font_color=TEXT_SECONDARY, title_font_color=TEXT_PRIMARY,
            title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="subsection-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Adjust the order parameters below and run a live '
        'cancellation risk prediction using the best trained model.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        month = st.slider("Order Month", 1, 12, 6)
        qty = st.number_input("Quantity", 1, 100, 2)
        amount = st.number_input("Order Amount (&#8377;)", 1, 20000, 800)
    with c2:
        fulfillment = st.selectbox("Fulfilment Method", ["Amazon", "Merchant"])
        service = st.selectbox("Service Level", ["Standard", "Expedited"])
        b2b = st.selectbox("Customer Type", ["B2C", "B2B"])

    promo = st.checkbox("Promotion Applied", value=False)
    weekend = st.checkbox("Weekend Delivery", value=False)

    if st.button("Predict Cancellation Risk"):
        best_model = model_data['best_model']
        fe = get_features()
        avg_price = fe['PricePerUnit'].median()
        avg_ma = fe['Revenue_7d_MA'].median()

        inp = pd.DataFrame([{
            'Qty': qty,
            'Amount': amount,
            'Month': month,
            'IsWeekend': 1 if weekend else 0,
            'PricePerUnit': amount / qty if qty > 0 else avg_price,
            'HasPromotion': 1 if promo else 0,
            'IsHighValue': 1 if amount > fe['Amount'].quantile(0.75) else 0,
            'Revenue_7d_MA': avg_ma,
            'Fulfilment': fulfillment,
            'ship-service-level': service,
            'B2B': b2b,
        }])
        for c in ['Fulfilment', 'ship-service-level', 'B2B']:
            inp[c] = inp[c].astype(str)

        try:
            prob = best_model.predict_proba(inp)[0, 1]
            if prob < 0.3:
                cls = "low"
                level = "Low Risk"
            elif prob < 0.6:
                cls = "moderate"
                level = "Moderate Risk"
            else:
                cls = "high"
                level = "High Risk"
            st.markdown(
                f'<div class="pred-result {cls}">'
                f'<div class="pred-value">{prob:.1%}</div>'
                f'<div class="pred-label">{level} &mdash; cancellation probability</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")


def render_sql_analysis() -> None:
    """Render the SQL analysis tab."""
    st.markdown('<div class="section-title">SQL Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Data is loaded into an in-memory <strong>SQLite</strong> '
        'database. Run pre-built analytical queries or write your own SQL.</div>',
        unsafe_allow_html=True,
    )

    conn = get_sql_connection()

    query_name = st.selectbox(
        "Query Template",
        list(SQL_QUERIES.keys()),
        label_visibility="collapsed",
    )
    query = SQL_QUERIES[query_name]

    st.markdown(
        f'<div class="code-block">{query}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Execute Query"):
        result = pd.read_sql_query(query, conn)
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download Results (CSV)",
            result.to_csv(index=False),
            f"{query_name}.csv",
            "text/csv",
        )

    with st.expander("Custom Query"):
        custom_query = st.text_area("Write SQL:", height=100, label_visibility="collapsed")
        if custom_query and st.button("Run Custom"):
            try:
                result = pd.read_sql_query(custom_query, conn)
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")


def render_data_explorer(dfs: Dict[str, pd.DataFrame]) -> None:
    """Render the data explorer tab.

    Args:
        dfs: dictionary of DataFrames from load_and_clean_data().
    """
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)

    name_map = {
        'Amazon Sales': 'sales',
        'International Sales': 'international',
        'Inventory': 'inventory',
        'Expenses': 'expenses',
        'Warehouse Comparison': 'warehouse',
    }
    choice = st.selectbox(
        "Dataset",
        list(name_map.keys()),
        label_visibility="collapsed",
    )
    df = dfs[name_map[choice]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Rows</span><br/>'
            f'<span class="stat-value">{len(df):,}</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Columns</span><br/>'
            f'<span class="stat-value">{len(df.columns)}</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        missing = df.isna().sum().sum()
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Missing Values</span><br/>'
            f'<span class="stat-value">{missing:,}</span></div>',
            unsafe_allow_html=True,
        )

    st.dataframe(df.head(20), use_container_width=True)

    if st.checkbox("Show Summary Statistics"):
        stats_df = df.describe(include='all').transpose().reset_index()
        st.dataframe(stats_df, use_container_width=True)


def render_api_playground() -> None:
    """Render the API playground tab.

    Users can test the cancellation prediction API endpoint,
    view the generated curl command, and see formatted responses.
    """
    st.markdown('<div class="section-title">API Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">The cancellation prediction model is also available '
        'as a <strong>REST API</strong> via FastAPI. Use this playground to test '
        'endpoints interactively or copy the <strong>curl</strong> command for '
        'integration into your own applications.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="subsection-title">Authentication</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "API Key",
        value="aQtSWtfMuBj6o18zT879yvybijQwMkmbRYvkvhuprWI",
        type="password",
    )
    st.markdown(
        f'<div class="info-box">Pass this key via the <code>X-API-Key</code> header. '
        f'In production, replace with a secure key generated via '
        f'<code>python -c "import secrets; print(secrets.token\\_urlsafe(32))"</code></div>',
        unsafe_allow_html=True,
    )

    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
    )

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="subsection-title">Test Prediction Endpoint</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        p_month = st.slider("Month", 1, 12, 6, key="api_month")
        p_qty = st.number_input("Quantity", 1, 100, 2, key="api_qty")
        p_amount = st.number_input("Order Amount (&#8377;)", 1, 20000, 1500, key="api_amt")
        p_weekend = st.checkbox("Weekend Delivery", key="api_we")
    with c2:
        p_fulfilment = st.selectbox("Fulfilment", ["Amazon", "Merchant"], key="api_ful")
        p_service = st.selectbox("Service Level", ["Standard", "Expedited"], key="api_svc")
        p_b2b = st.selectbox("Customer Type", ["B2C", "B2B"], key="api_b2b")
        p_promo = st.checkbox("Promotion Applied", key="api_promo")

    payload = {
        "Month": p_month,
        "Qty": p_qty,
        "Amount": p_amount,
        "IsWeekend": p_weekend,
        "HasPromotion": p_promo,
        "Fulfilment": p_fulfilment,
        "ServiceLevel": p_service,
        "B2B": p_b2b == "B2B",
        "Category": "Set",
    }

    curl_cmd = (
        f'curl -s -X POST "{api_url}/predict" \\\n'
        f'  -H "X-API-Key: {api_key}" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f'  -d \'{payload}\''
    )
    st.markdown('<div class="subsection-title">Equivalent curl Command</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="code-block">{curl_cmd}</div>', unsafe_allow_html=True)

    if st.button("Send Request"):
        if not api_key or api_key == "your-api-key-here":
            st.error("Please enter a valid API key.")
        else:
            with st.spinner("Sending request..."):
                try:
                    resp = requests.post(
                        f"{api_url}/predict",
                        json=payload,
                        headers={"X-API-Key": api_key},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        prob = data.get("cancellation_probability", 0)
                        risk = data.get("risk_level", "Unknown")
                        model_used = data.get("model_used", "")
                        features = data.get("features_used", [])

                        if risk == "Low":
                            box_cls = "success-box"
                        elif risk == "Moderate":
                            box_cls = "warning-box"
                        else:
                            box_cls = "danger-box"

                        st.markdown(
                            f'<div class="{box_cls}">'
                            f'<strong>Risk Level:</strong> {risk} &mdash; '
                            f'<strong>Probability:</strong> {prob:.1%}<br/>'
                            f'<strong>Model:</strong> {model_used} &mdash; '
                            f'<strong>Features:</strong> {", ".join(features[:5])} +'
                            f' {max(0, len(features)-5)} more'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(
                            f"HTTP {resp.status_code}: {resp.json().get('detail', resp.text)}"
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not connect. Ensure the FastAPI server is running "
                        f"at {api_url} (uvicorn api:app --reload)."
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown('<div class="subsection-title">API Endpoints</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="code-block">'
        'POST /predict     &mdash;  Predict cancellation risk  &mdash;  auth: X-API-Key\n'
        'GET  /model-info  &mdash;  Model metadata             &mdash;  auth: X-API-Key\n'
        'GET  /            &mdash;  Health check               &mdash;  no auth'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app."""
    st.markdown(
        '<div class="main-title">E-commerce BI Platform</div>'
        '<div class="main-subtitle">Business Intelligence &middot; Machine Learning &middot; Analytics</div>',
        unsafe_allow_html=True,
    )

    dfs = load_data()
    sales = dfs['sales']

    st.sidebar.markdown(
        '<div class="sidebar-section"><strong style="font-size:0.8125rem;color:'
        f'{TEXT_SECONDARY};">DATE RANGE</strong></div>',
        unsafe_allow_html=True,
    )
    min_date, max_date = sales['Date'].min(), sales['Date'].max()
    start = st.sidebar.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end = st.sidebar.date_input("End", max_date, min_value=min_date, max_value=max_date)

    show_all = st.sidebar.checkbox("Use full date range", value=False)
    filtered = sales if show_all else sales[
        (sales['Date'] >= pd.to_datetime(start)) &
        (sales['Date'] <= pd.to_datetime(end))
    ]

    tab_labels = [
        "BI Dashboard",
        "Segmentation",
        "ML Models",
        "SQL Analysis",
        "Data Explorer",
        "API Playground",
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

    with tab1:
        render_bi_dashboard(filtered, sales)
    with tab2:
        render_segmentation()
    with tab3:
        render_ml_comparison()
    with tab4:
        render_sql_analysis()
    with tab5:
        render_data_explorer(dfs)
    with tab6:
        render_api_playground()


if __name__ == "__main__":
    main()
