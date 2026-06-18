import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="E-commerce Business Intelligence",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card h3 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .metric-card h2 {
        color: #2E86C1;
        margin: 0;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        color: white;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2E86C1;
    }
    .stDataFrame {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    amazon_sales = pd.read_csv('Amazon Sale Report.csv', low_memory=False)
    international_sales = pd.read_csv('International sale Report.csv')
    expenses = pd.read_csv('Expense IIGF.csv')
    inventory = pd.read_csv('Sale Report.csv')

    amazon_sales = amazon_sales.drop(columns=['Unnamed: 22', 'index'], errors='ignore')
    amazon_sales = amazon_sales.dropna(subset=['Amount'])
    amazon_sales = amazon_sales.drop_duplicates()

    amazon_sales['Date'] = pd.to_datetime(amazon_sales['Date'], errors='coerce')
    amazon_sales['Amount'] = pd.to_numeric(amazon_sales['Amount'], errors='coerce')
    amazon_sales['Qty'] = pd.to_numeric(amazon_sales['Qty'], errors='coerce')

    amazon_sales['Year'] = amazon_sales['Date'].dt.year
    amazon_sales['Month'] = amazon_sales['Date'].dt.month
    amazon_sales['Month_Name'] = amazon_sales['Date'].dt.month_name()
    amazon_sales['Quarter'] = amazon_sales['Date'].dt.quarter
    amazon_sales['Revenue'] = amazon_sales['Amount']
    amazon_sales['DayOfWeek'] = amazon_sales['Date'].dt.dayofweek
    amazon_sales['IsWeekend'] = amazon_sales['DayOfWeek'].isin([5, 6]).astype(int)

    expenses.columns = ['index', 'Income_Particular', 'Income_Amount',
                       'Expense_Particular', 'Expense_Amount']
    expenses = expenses.drop(0)
    expenses = expenses[expenses['Income_Particular'] != 'Total']
    expenses['Income_Amount'] = pd.to_numeric(expenses['Income_Amount'], errors='coerce')
    expenses['Expense_Amount'] = pd.to_numeric(expenses['Expense_Amount'], errors='coerce')

    return amazon_sales, international_sales, expenses, inventory

@st.cache_data
def calculate_kpis(df):
    total_revenue = df['Revenue'].sum()
    total_orders = df['Order ID'].nunique()
    total_units = df['Qty'].sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    cancelled_mask = df['Status'].str.contains('cancel', case=False, na=False)
    cancelled_revenue = df[cancelled_mask]['Revenue'].sum()
    cancelled_orders = df[cancelled_mask]['Order ID'].nunique()
    cancellation_rate = cancelled_orders / total_orders if total_orders > 0 else 0

    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'total_units': total_units,
        'avg_order_value': avg_order_value,
        'cancelled_revenue': cancelled_revenue,
        'cancellation_rate': cancellation_rate
    }

@st.cache_data
def train_cancellation_model(df):
    df_model = df.copy()
    df_model['IsCancelled'] = df_model['Status'].str.contains('cancel', case=False, na=False).astype(int)

    features = pd.DataFrame(index=df_model.index)
    features['Qty'] = df_model['Qty'].fillna(0)
    features['Amount'] = df_model['Amount'].fillna(0)
    features['Month'] = df_model['Month'].fillna(1)
    features['IsWeekend'] = df_model['IsWeekend'].fillna(0)
    features['B2B'] = df_model['B2B'].astype(int)

    fulfillment_map = {'Amazon': 1, 'Merchant': 0}
    features['Fulfillment'] = df_model['Fulfilment'].map(fulfillment_map).fillna(0)

    service_map = {'Expedited': 1, 'Standard': 0}
    features['ServiceLevel'] = df_model['ship-service-level'].map(service_map).fillna(0)

    le = LabelEncoder()
    features['Category'] = le.fit_transform(df_model['Category'].fillna('Unknown'))

    target = df_model['IsCancelled']

    valid = features.notna().all(axis=1)
    features = features[valid]
    target = target[valid]

    if target.nunique() < 2:
        return None, features, target, 0

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = 0.5

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': report['1']['precision'] if '1' in report else 0,
        'recall': report['1']['recall'] if '1' in report else 0,
        'f1': report['1']['f1-score'] if '1' in report else 0,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    return model, features, target, metrics

def main():
    amazon_sales, international_sales, expenses, inventory = load_data()
    kpis = calculate_kpis(amazon_sales)

    st.markdown('<h1 class="main-header">\U0001f4ca E-commerce Business Intelligence</h1>',
                unsafe_allow_html=True)

    st.sidebar.markdown("## Control Panel")

    min_date = amazon_sales['Date'].min()
    max_date = amazon_sales['Date'].max()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    filtered_data = amazon_sales[
        (amazon_sales['Date'] >= pd.to_datetime(start_date)) &
        (amazon_sales['Date'] <= pd.to_datetime(end_date))
    ]

    filtered_kpis = calculate_kpis(filtered_data)

    tab1, tab2, tab3 = st.tabs([
        "\U0001f4c8 Business Intelligence",
        "\U0001f916 ML Cancellation Predictor",
        "\U0001f4ca Data Explorer"
    ])

    with tab1:
        st.markdown("## Business Intelligence Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Revenue</h3>
                <h2>\u20b9{filtered_kpis['total_revenue']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Orders</h3>
                <h2>{filtered_kpis['total_orders']:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Order Value</h3>
                <h2>\u20b9{filtered_kpis['avg_order_value']:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cancellation Rate</h3>
                <h2>{filtered_kpis['cancellation_rate']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Monthly Revenue Trend")
            monthly_rev = filtered_data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
            monthly_rev['Date'] = pd.to_datetime(monthly_rev['Year'].astype(str) + '-' + monthly_rev['Month'].astype(str))

            fig = px.line(monthly_rev, x='Date', y='Revenue',
                         title='Monthly Revenue Trend',
                         labels={'Revenue': 'Revenue (\u20b9)', 'Date': 'Month'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Top Categories by Revenue")
            category_rev = filtered_data.groupby('Category')['Revenue'].sum().sort_values(ascending=False).head(10)

            fig = px.bar(x=category_rev.index, y=category_rev.values,
                        title='Top 10 Categories by Revenue',
                        labels={'x': 'Category', 'y': 'Revenue (\u20b9)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top States by Revenue")
            state_rev = filtered_data.groupby('ship-state')['Revenue'].sum().sort_values(ascending=False).head(10)

            fig = px.bar(x=state_rev.index, y=state_rev.values,
                        title='Top 10 States by Revenue',
                        labels={'x': 'State', 'y': 'Revenue (\u20b9)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Order Status Distribution")
            status_counts = filtered_data['Status'].value_counts()

            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Order Status Distribution')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Additional Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Growth Metrics**")
            monthly_rev['Growth_Rate'] = monthly_rev['Revenue'].pct_change() * 100
            avg_growth = monthly_rev['Growth_Rate'].mean()
            st.write(f"Average Monthly Growth: {avg_growth:.1f}%")

        with col2:
            st.markdown("**Customer Analysis**")
            b2b_revenue = filtered_data.groupby('B2B')['Revenue'].sum()
            st.write(f"B2B Revenue: \u20b9{b2b_revenue.get(True, 0):,.0f}")
            st.write(f"B2C Revenue: \u20b9{b2b_revenue.get(False, 0):,.0f}")

        with col3:
            st.markdown("**Product Insights**")
            top_sku = filtered_data.groupby('SKU')['Revenue'].sum().sort_values(ascending=False).head(1)
            if not top_sku.empty:
                st.write(f"Top SKU: {top_sku.index[0]}")
                st.write(f"Revenue: \u20b9{top_sku.iloc[0]:,.0f}")

    with tab2:
        st.markdown("## ML Cancellation Predictor")

        model, features, target, metrics = train_cancellation_model(filtered_data)

        if model is None:
            st.warning("Filtered dataset does not contain enough cancelled orders to train a model. Expand the date range.")
        else:
            st.success(f"Model trained on {metrics['train_size']} orders, tested on {metrics['test_size']} orders")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            with col2:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
            with col3:
                st.metric("Precision", f"{metrics['precision']:.2f}")
            with col4:
                st.metric("Recall", f"{metrics['recall']:.2f}")

            st.subheader("Enter Order Details")
            col1, col2 = st.columns(2)

            with col1:
                month = st.slider("Month (1-12)", 1, 12, datetime.now().month)
                qty = st.number_input("Quantity", min_value=1, max_value=100, value=1)
                amount = st.number_input("Order Amount (\u20b9)", min_value=1, max_value=10000, value=500)
                is_weekend = st.checkbox("Weekend Delivery?", value=False)

            with col2:
                fulfillment = st.selectbox("Fulfillment By", ["Amazon", "Merchant"])
                service_level = st.selectbox("Shipping Service", ["Standard", "Expedited"])
                b2b = st.selectbox("Customer Type", ["B2C", "B2B"])
                category = st.selectbox("Category", filtered_data['Category'].dropna().unique())

            if st.button("Predict Cancellation Risk"):
                le = LabelEncoder()
                le.fit(filtered_data['Category'].fillna('Unknown'))
                cat_encoded = le.transform([category])[0] if category in le.classes_ else 0

                input_data = pd.DataFrame([{
                    'Qty': qty,
                    'Amount': amount,
                    'Month': month,
                    'IsWeekend': 1 if is_weekend else 0,
                    'B2B': 1 if b2b == "B2B" else 0,
                    'Fulfillment': 1 if fulfillment == "Amazon" else 0,
                    'ServiceLevel': 1 if service_level == "Expedited" else 0,
                    'Category': cat_encoded
                }])

                prob = model.predict_proba(input_data)[0, 1]

                if prob < 0.3:
                    st.success(f"**Low Risk** - Cancellation probability: {prob:.1%}")
                elif prob < 0.6:
                    st.warning(f"**Moderate Risk** - Cancellation probability: {prob:.1%}")
                else:
                    st.error(f"**High Risk** - Cancellation probability: {prob:.1%}")

            st.subheader("Feature Importance (Model Coefficients)")
            feature_names = ['Qty', 'Amount', 'Month', 'IsWeekend', 'B2B',
                           'Fulfillment', 'ServiceLevel', 'Category']
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=False)

            fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                        title='Feature Impact on Cancellation Probability',
                        labels={'Coefficient': 'Coefficient Value'})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("## Data Explorer")

        st.markdown("### Dataset Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Amazon Sales Data**")
            st.write(f"Total Records: {len(amazon_sales):,}")
            st.write(f"Date Range: {amazon_sales['Date'].min().date()} to {amazon_sales['Date'].max().date()}")
            st.write(f"Columns: {len(amazon_sales.columns)}")

        with col2:
            st.markdown("**Other Datasets**")
            st.write(f"International Sales: {len(international_sales):,} records")
            st.write(f"Inventory: {len(inventory):,} records")
            st.write(f"Expenses: {len(expenses):,} records")

        st.markdown("### Data Preview")

        dataset_choice = st.selectbox("Select Dataset",
                                     ["Amazon Sales", "International Sales", "Inventory", "Expenses"])

        if dataset_choice == "Amazon Sales":
            st.dataframe(amazon_sales.head(10))
        elif dataset_choice == "International Sales":
            st.dataframe(international_sales.head(10))
        elif dataset_choice == "Inventory":
            st.dataframe(inventory.head(10))
        else:
            st.dataframe(expenses.head(10))

        if dataset_choice == "Amazon Sales":
            st.markdown("### Statistical Summary")
            st.dataframe(amazon_sales.describe())

if __name__ == "__main__":
    main()
