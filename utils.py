"""Core utilities for E-commerce Business Intelligence analysis.

Provides data loading, feature engineering, customer segmentation,
ML model comparison, and statistical analysis functions.
"""

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_clean_data() -> Dict[str, pd.DataFrame]:
    """Load all CSV datasets and return cleaned DataFrames.

    Returns:
        dict with keys: 'sales', 'international', 'expenses', 'inventory', 'warehouse'
    """
    sales = pd.read_csv('Amazon Sale Report.csv', low_memory=False)
    international = pd.read_csv('International sale Report.csv')
    expenses = pd.read_csv('Expense IIGF.csv')
    inventory = pd.read_csv('Sale Report.csv')
    warehouse = pd.read_csv('Cloud Warehouse Compersion Chart.csv')

    sales.columns = sales.columns.str.strip()
    sales = sales.drop(columns=['Unnamed: 22', 'index'], errors='ignore')
    sales = sales.dropna(subset=['Amount'])
    sales = sales.drop_duplicates()
    sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
    sales['Amount'] = pd.to_numeric(sales['Amount'], errors='coerce')
    sales['Qty'] = pd.to_numeric(sales['Qty'], errors='coerce')

    international.columns = international.columns.str.strip()
    international = international.drop(columns=['index'], errors='ignore')

    expenses.columns = ['index', 'Income_Particular', 'Income_Amount',
                        'Expense_Particular', 'Expense_Amount']
    expenses = expenses.drop(columns=['index'], errors='ignore')

    inventory.columns = inventory.columns.str.strip()
    inventory = inventory.drop(columns=['index'], errors='ignore')

    warehouse.columns = warehouse.columns.str.strip()
    warehouse = warehouse.drop(columns=['index'], errors='ignore')

    return {
        'sales': sales,
        'international': international,
        'expenses': expenses,
        'inventory': inventory,
        'warehouse': warehouse,
    }


# ---------------------------------------------------------------------------
# Dataset joining
# ---------------------------------------------------------------------------

def join_sales_with_inventory(
    sales: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    """Join sales transactions with inventory/stock data on SKU/Category.

    Args:
        sales: cleaned sales DataFrame.
        inventory: cleaned inventory DataFrame.

    Returns:
        Joined DataFrame with inventory stock info appended to each sale.
    """
    inv_summary = (
        inventory.groupby('Category')
        .agg(Total_Stock=('Stock', 'sum'), Unique_SKUs=('SKU Code', 'nunique'))
        .reset_index()
        .rename(columns={'Category': 'Category'})
    )
    combined = sales.merge(inv_summary, on='Category', how='left')
    return combined


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML models.

    Adds temporal features, price-per-unit, promotion flags, and rolling stats.

    Args:
        df: sales DataFrame with at least Date, Amount, Qty, promotion-ids columns.

    Returns:
        DataFrame with additional feature columns.
    """
    fe = df.copy()
    fe['DayOfWeek'] = fe['Date'].dt.dayofweek
    fe['IsWeekend'] = fe['DayOfWeek'].isin([5, 6]).astype(int)
    fe['Month'] = fe['Date'].dt.month
    fe['PricePerUnit'] = np.where(fe['Qty'] > 0, fe['Amount'] / fe['Qty'], 0)
    fe['HasPromotion'] = fe['promotion-ids'].notna().astype(int)
    fe['IsHighValue'] = (fe['Amount'] > fe['Amount'].quantile(0.75)).astype(int)

    fe_sorted = fe.sort_values('Date').reset_index(drop=True)
    fe_sorted['Revenue_7d_MA'] = (
        fe_sorted.groupby('Category')['Amount']
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    fe = fe_sorted
    fe['Cancelled'] = fe['Status'].str.contains('cancel', case=False, na=False).astype(int)
    return fe


# ---------------------------------------------------------------------------
# SQLite in-memory database
# ---------------------------------------------------------------------------

def create_sqlite_db(
    dfs: Dict[str, pd.DataFrame]
) -> Any:
    """Load DataFrames into an in-memory SQLite database for SQL queries.

    Args:
        dfs: dictionary of DataFrames from load_and_clean_data().

    Returns:
        sqlite3.Connection object.
    """
    import sqlite3

    conn = sqlite3.connect(':memory:')
    for name, df in dfs.items():
        df.head(10000).to_sql(name, conn, index=False, if_exists='replace')
    return conn


SQL_QUERIES: Dict[str, str] = {
    "Revenue by category (top 10)": """
        SELECT Category, SUM(Amount) as total_revenue, COUNT(*) as order_count
        FROM sales
        GROUP BY Category
        ORDER BY total_revenue DESC
        LIMIT 10
    """,
    "Monthly revenue trend": """
        SELECT strftime('%Y-%m', Date) as month, SUM(Amount) as revenue
        FROM sales
        WHERE Date IS NOT NULL
        GROUP BY month
        ORDER BY month
    """,
    "Cancellation rate by fulfillment": """
        SELECT Fulfilment,
               SUM(CASE WHEN Status LIKE '%Cancel%' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as cancel_rate,
               COUNT(*) as total_orders
        FROM sales
        GROUP BY Fulfilment
    """,
    "B2B vs B2C performance": """
        SELECT B2B, COUNT(*) as orders, SUM(Amount) as revenue,
               AVG(Amount) as avg_order_value
        FROM sales
        GROUP BY B2B
    """,
    "Top states by revenue": """
        SELECT ship_state, SUM(Amount) as revenue, COUNT(*) as orders
        FROM sales
        GROUP BY ship_state
        ORDER BY revenue DESC
        LIMIT 10
    """,
}


# ---------------------------------------------------------------------------
# RFM analysis + K-means
# ---------------------------------------------------------------------------

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary values per customer.

    Args:
        df: sales DataFrame with Order ID, Date, Amount.

    Returns:
        DataFrame with one row per customer and RFM scores.
    """
    now = df['Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Order ID').agg(
        Recency=('Date', lambda x: (now - x.max()).days),
        Frequency=('Order ID', 'count'),
        Monetary=('Amount', 'sum'),
    ).reset_index()
    rfm.columns = ['OrderID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0].reset_index(drop=True)
    return rfm


def segment_customers(
    rfm: pd.DataFrame, n_clusters: int = 4
) -> Tuple[pd.DataFrame, KMeans]:
    """Segment customers using K-means on log-transformed RFM values.

    Args:
        rfm: DataFrame with Recency, Frequency, Monetary columns.
        n_clusters: number of segments to create.

    Returns:
        Tuple of (DataFrame with segment labels, fitted KMeans model).
    """
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_scaled['Recency'] = np.log1p(rfm_scaled['Recency'])
    rfm_scaled['Frequency'] = np.log1p(rfm_scaled['Frequency'])
    rfm_scaled['Monetary'] = np.log1p(rfm_scaled['Monetary'])

    scaler = StandardScaler()
    rfm_norm = scaler.fit_transform(rfm_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_norm)

    result = rfm.copy()
    result['Segment'] = labels
    result['Segment'] = result['Segment'].astype(str)

    segment_map = _label_segments(result, kmeans, scaler)
    result['Segment_Label'] = result['Segment'].map(segment_map)
    return result, kmeans


def _label_segments(
    rfm: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler
) -> Dict[str, str]:
    """Assign human-readable labels to K-means clusters based on RFM centroids."""
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    labels: Dict[str, str] = {}
    for i, c in enumerate(centroids):
        rec, freq, mon = c
        if freq > 3 and mon > 5000:
            label = 'Champions'
        elif rec < 30 and freq > 1:
            label = 'Loyal Customers'
        elif rec < 60 and mon > 2000:
            label = 'Potential Loyalists'
        elif mon > 3000 and freq < 2:
            label = 'Big Spenders'
        elif rec > 180:
            label = 'At Risk'
        elif rec > 90:
            label = 'Needs Attention'
        elif freq > 5:
            label = 'Frequent Buyers'
        else:
            label = f'Segment {i}'
        labels[str(i)] = label
    return labels


# ---------------------------------------------------------------------------
# ML pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_features = ['Qty', 'Amount', 'Month', 'IsWeekend', 'PricePerUnit',
                        'HasPromotion', 'IsHighValue', 'Revenue_7d_MA']
    categorical_features = ['Fulfilment', 'ship-service-level', 'B2B']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )


def get_classifiers() -> Dict[str, BaseEstimator]:
    """Return dict of classifier names to models for comparison.

    Returns:
        Mapping of model name to unfitted sklearn estimator.
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Return hyperparameter grids for each model.

    Returns:
        Nested dict mapping model name to param_grid for GridSearchCV.
    """
    return {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        },
    }


def train_and_evaluate(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Train a model (with optional GridSearch) and return evaluation metrics.

    Args:
        model: sklearn-compatible estimator.
        X_train: training features.
        y_train: training labels.
        X_test: test features.
        y_test: test labels.
        param_grid: optional hyperparameter grid for GridSearchCV.

    Returns:
        dict with keys: model, best_params, accuracy, precision, recall, f1,
        roc_auc, y_pred, y_prob, report
    """
    if param_grid:
        search = GridSearchCV(
            model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        best_model = model.fit(X_train, y_train)
        best_params = {}

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    result: Dict[str, Any] = {
        'model': best_model,
        'best_params': best_params,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': classification_report(y_test, y_pred, output_dict=True,
                                        zero_division=0),
    }
    return result


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_gridsearch: bool = True,
) -> pd.DataFrame:
    """Compare multiple classifiers and return results as a DataFrame.

    Args:
        X_train: training features.
        y_train: training labels.
        X_test: test features.
        y_test: test labels.
        use_gridsearch: whether to use GridSearchCV for hyperparameter tuning.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    classifiers = get_classifiers()
    grids = get_param_grids() if use_gridsearch else {}
    rows: List[Dict[str, Any]] = []

    for name, clf in classifiers.items():
        pg = grids.get(name, None)
        result = train_and_evaluate(clf, X_train, y_train, X_test, y_test, pg)
        rows.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1': result['f1'],
            'ROC-AUC': result['roc_auc'],
            'Best Params': str(result['best_params']),
        })

    return pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def hypothesis_tests(sales: pd.DataFrame) -> Dict[str, Any]:
    """Run statistical hypothesis tests on sales data.

    Tests performed:
        - Welch's t-test: B2B vs B2C average order value
        - Chi-square: fulfillment type vs cancellation

    Args:
        sales: cleaned sales DataFrame.

    Returns:
        dict with test names as keys and result dicts as values.
    """
    results: Dict[str, Any] = {}

    b2b = sales[sales['B2B'] == True]['Amount'].dropna()
    b2c = sales[sales['B2B'] == False]['Amount'].dropna()
    if len(b2b) > 1 and len(b2c) > 1:
        t_stat, p_val = stats.ttest_ind(b2b, b2c, equal_var=False)
        results['B2B vs B2C AOV (t-test)'] = {
            't-statistic': round(t_stat, 4),
            'p-value': round(p_val, 6),
            'B2B mean AOV': round(b2b.mean(), 2),
            'B2C mean AOV': round(b2c.mean(), 2),
            'significant': p_val < 0.05,
        }

    contingency = pd.crosstab(
        sales['Fulfilment'].fillna('Unknown'),
        sales['Status'].str.contains('cancel', case=False, na=False).map({True: 'Cancelled', False: 'Active'}),
    )
    if contingency.shape == (2, 2):
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
        results['Fulfillment vs Cancellation (chi-square)'] = {
            'chi2': round(chi2, 4),
            'p-value': round(p_chi, 6),
            'significant': p_chi < 0.05,
        }

    return results


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

MODEL_PATH = 'cancellation_model.pkl'


def save_model_pipeline(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    features: List[str],
    path: str = MODEL_PATH,
) -> None:
    """Save the fitted model, preprocessor, and feature list to disk.

    Args:
        model: trained sklearn estimator.
        preprocessor: fitted ColumnTransformer.
        features: list of feature names used.
        path: file path to save to.
    """
    joblib.dump({'model': model, 'preprocessor': preprocessor, 'features': features}, path)


def load_model_pipeline(path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Load a previously saved model pipeline from disk.

    Args:
        path: file path to load from.

    Returns:
        dict with keys model, preprocessor, features, or None if not found.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None
