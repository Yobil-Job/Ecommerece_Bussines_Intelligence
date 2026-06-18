"""Streamlit dashboard for E-commerce Business Intelligence.

Provides interactive BI dashboards, customer segmentation,
ML model comparison, SQL analysis, and data exploration.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
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
    page_title="E-commerce Business Intelligence",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; color: #2E86C1; text-align: center; margin-bottom: 2rem; }
    .metric-card {
        background-color: #1e1e1e; color: white; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #2E86C1; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #ffffff; margin-bottom: 0.5rem; font-size: 1rem; }
    .metric-card h2 { color: #2E86C1; margin: 0; font-size: 1.8rem; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e1e1e; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 1rem 2rem; color: white; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #2E86C1; }
    .stDataFrame { background-color: #1e1e1e; color: white; }
</style>
""",
    unsafe_allow_html=True,
)


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

def render_bi_dashboard(filtered: pd.DataFrame) -> None:
    """Render the Business Intelligence dashboard tab."""
    st.markdown("## Business Intelligence Dashboard")
    rev = filtered['Amount'].sum()
    orders = filtered['Order ID'].nunique()
    aov = rev / orders if orders > 0 else 0
    cancel_mask = filtered['Status'].str.contains('cancel', case=False, na=False)
    cancel_rate = cancel_mask.sum() / len(filtered) if len(filtered) > 0 else 0

    cols = st.columns(4)
    for col, title, val in zip(
        cols,
        ['Total Revenue', 'Total Orders', 'Avg Order Value', 'Cancellation Rate'],
        [f'\u20b9{rev:,.0f}', f'{orders:,}', f'\u20b9{aov:.0f}', f'{cancel_rate:.1%}'],
    ):
        col.markdown(
            f'<div class="metric-card"><h3>{title}</h3><h2>{val}</h2></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        monthly = filtered.groupby(filtered['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly['Date'] = monthly['Date'].astype(str)
        fig = px.line(monthly, x='Date', y='Amount', title='Monthly Revenue Trend')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cats = filtered.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=cats.index, y=cats.values, title='Top 10 Categories by Revenue')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        states = filtered.groupby('ship-state')['Amount'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=states.index, y=states.values, title='Top States by Revenue')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        status = filtered['Status'].value_counts()
        fig = px.pie(values=status.values, names=status.index, title='Order Status')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Statistical Hypothesis Tests")
    tests = hypothesis_tests(filtered)
    for name, result in tests.items():
        sig = "Significant" if result.get('significant') else "Not significant"
        st.write(f"**{name}**: p={result.get('p-value', 'N/A')} — {sig}")
        if 'B2B mean AOV' in result:
            st.write(f"  B2B AOV: \u20b9{result['B2B mean AOV']:,.2f} | "
                     f"B2C AOV: \u20b9{result['B2C mean AOV']:,.2f}")

    st.markdown("### Dataset Join Summary")
    joined = get_joined_data()
    st.write(f"Sales with inventory: {len(joined):,} rows "
             f"(joined on Category), {joined['Total_Stock'].notna().sum():,} have stock data")


def render_segmentation() -> None:
    """Render the customer segmentation tab."""
    st.markdown("## Customer Segmentation (RFM + K-means)")
    segmented, summary_html = get_rfm_segments()

    st.markdown("### Segment Profiles")
    seg_profile = segmented.groupby('Segment_Label').agg(
        Customers=('OrderID', 'nunique'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
    ).round(1).sort_values('Avg_Monetary', ascending=False)

    st.dataframe(seg_profile.style.format({
        'Avg_Recency': '{:.0f}d',
        'Avg_Frequency': '{:.1f}',
        'Avg_Monetary': '\u20b9{:.0f}',
        'Total_Revenue': '\u20b9{:.0f}',
    }))

    fig = px.scatter_3d(
        segmented.sample(min(2000, len(segmented))),
        x='Recency', y='Frequency', z='Monetary',
        color='Segment_Label', title='Customer Segments (sampled)',
        labels={'Recency': 'Days since last order', 'Frequency': 'Order count', 'Monetary': 'Total spent'},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Breakdown")
    col1, col2 = st.columns(2)
    segments = segmented.groupby('Segment_Label').agg(
        Customers=('OrderID', 'nunique'),
        Revenue=('Monetary', 'sum'),
    ).sort_values('Revenue', ascending=False)
    segments['Revenue_Share'] = (segments['Revenue'] / segments['Revenue'].sum() * 100).round(1)

    with col1:
        fig = px.pie(segments, values='Customers', names=segments.index, title='Customers by Segment')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(segments, values='Revenue', names=segments.index, title='Revenue by Segment')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Actionable Insights")
    st.write("- **Champions**: Reward with loyalty programs and exclusive offers")
    st.write("- **Loyal Customers**: Nurture with personalized recommendations")
    st.write("- **At Risk**: Re-engage with win-back campaigns and discount offers")
    st.write("- **Big Spenders**: Cross-sell and upsell premium products")


def render_ml_comparison() -> None:
    """Render the ML model comparison tab."""
    st.markdown("## ML Model Comparison Dashboard")
    model_data = train_models()

    if 'error' in model_data:
        st.warning(model_data['error'])
        return

    results_df: pd.DataFrame = model_data['results_df']
    st.markdown("### Model Performance Comparison (GridSearchCV + 3-fold CV)")
    st.dataframe(
        results_df.style.format({
            'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}',
            'F1': '{:.2%}', 'ROC-AUC': '{:.3f}',
        })
    )

    fig = px.bar(
        results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']),
        x='Model', y='value', color='variable', barmode='group',
        title='Model Comparison Across Metrics',
    )
    st.plotly_chart(fig, use_container_width=True)

    best_name = results_df.iloc[0]['Model']
    st.success(f"**Best Model**: {best_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.3f})")

    fi = model_data.get('feature_importance', pd.DataFrame())
    if not fi.empty:
        st.markdown("### Feature Importance")
        fig = px.bar(
            fi.head(10), x='Importance', y='Feature', orientation='h',
            title=f'Top Features — {best_name}',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Make a Prediction")
    c1, c2 = st.columns(2)
    with c1:
        month = st.slider("Month", 1, 12, 6)
        qty = st.number_input("Quantity", 1, 100, 2)
        amount = st.number_input("Order Amount (\u20b9)", 1, 20000, 800)
        weekend = st.checkbox("Weekend Delivery")
        promo = st.checkbox("Has Promotion")
    with c2:
        fulfillment = st.selectbox("Fulfillment", ["Amazon", "Merchant"])
        service = st.selectbox("Service Level", ["Standard", "Expedited"])
        b2b = st.selectbox("Customer Type", ["B2C", "B2B"])
        cat = st.selectbox("Category", load_data()['sales']['Category'].dropna().unique())

    if st.button("Predict Cancellation Risk"):
        best_model = model_data['best_model']
        fe = get_features()
        avg_price = fe['PricePerUnit'].median()
        avg_ma = fe['Revenue_7d_MA'].median()

        inp = pd.DataFrame([{
            'Qty': qty, 'Amount': amount, 'Month': month,
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
                st.success(f"**Low Risk** — {prob:.1%} cancellation probability")
            elif prob < 0.6:
                st.warning(f"**Moderate Risk** — {prob:.1%} cancellation probability")
            else:
                st.error(f"**High Risk** — {prob:.1%} cancellation probability")
        except Exception as e:
            st.error(f"Prediction error: {e}")


def render_sql_analysis() -> None:
    """Render the SQL analysis tab."""
    st.markdown("## SQL Analysis")
    conn = get_sql_connection()

    query_name = st.selectbox("Select a query to run", list(SQL_QUERIES.keys()))
    query = SQL_QUERIES[query_name]

    st.code(query, language='sql')

    if st.button("Run Query"):
        result = pd.read_sql_query(query, conn)
        st.dataframe(result)
        st.download_button(
            "Download CSV", result.to_csv(index=False), f"{query_name}.csv", "text/csv"
        )

    st.markdown("### Custom Query")
    custom_query = st.text_area("Write your own SQL:", height=100)
    if custom_query and st.button("Run Custom Query"):
        try:
            result = pd.read_sql_query(custom_query, conn)
            st.dataframe(result)
        except Exception as e:
            st.error(f"Query error: {e}")


def render_data_explorer(dfs: Dict[str, pd.DataFrame]) -> None:
    """Render the data explorer tab."""
    st.markdown("## Data Explorer")
    name_map = {
        'Amazon Sales': 'sales',
        'International Sales': 'international',
        'Inventory': 'inventory',
        'Expenses': 'expenses',
        'Warehouse Comparison': 'warehouse',
    }
    choice = st.selectbox("Select Dataset", list(name_map.keys()))
    df = dfs[name_map[choice]]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
    with c2:
        st.metric("Missing Cells", f"{df.isna().sum().sum():,}")
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    st.dataframe(df.head(20))
    if st.checkbox("Show summary statistics"):
        st.dataframe(df.describe(include='all'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit app."""
    st.markdown('<h1 class="main-header">\U0001f4ca E-commerce Business Intelligence</h1>', unsafe_allow_html=True)

    dfs = load_data()
    sales = dfs['sales']

    st.sidebar.markdown("## Control Panel")
    min_date, max_date = sales['Date'].min(), sales['Date'].max()
    start = st.sidebar.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end = st.sidebar.date_input("End", max_date, min_value=min_date, max_value=max_date)
    filtered = sales[(sales['Date'] >= pd.to_datetime(start)) & (sales['Date'] <= pd.to_datetime(end))]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "\U0001f4c8 BI Dashboard", "\U0001f465 Segmentation",
        "\U0001f916 ML Models", "\U0001f4db SQL Analysis", "\U0001f4ca Data Explorer",
    ])

    with tab1:
        render_bi_dashboard(filtered)
    with tab2:
        render_segmentation()
    with tab3:
        render_ml_comparison()
    with tab4:
        render_sql_analysis()
    with tab5:
        render_data_explorer(dfs)


if __name__ == "__main__":
    main()
