import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="E-commerce Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    /* Dark theme fixes */
    .stDataFrame {
        background-color: #1e1e1e;
        color: white;
    }
    .css-1d391kg {
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess all datasets"""
    # Load main datasets
    amazon_sales = pd.read_csv('Amazon Sale Report.csv', low_memory=False)
    international_sales = pd.read_csv('International sale Report.csv')
    expenses = pd.read_csv('Expense IIGF.csv')
    inventory = pd.read_csv('Sale Report.csv')
    
    # Data cleaning (from notebook)
    amazon_sales = amazon_sales.drop(columns=['Unnamed: 22', 'index'])
    amazon_sales = amazon_sales.dropna(subset=['Amount'])
    amazon_sales = amazon_sales.drop_duplicates()
    
    # Convert data types
    amazon_sales['Date'] = pd.to_datetime(amazon_sales['Date'])
    amazon_sales['Amount'] = pd.to_numeric(amazon_sales['Amount'])
    amazon_sales['Qty'] = pd.to_numeric(amazon_sales['Qty'])
    
    # Add time features
    amazon_sales['Year'] = amazon_sales['Date'].dt.year
    amazon_sales['Month'] = amazon_sales['Date'].dt.month
    amazon_sales['Month_Name'] = amazon_sales['Date'].dt.month_name()
    amazon_sales['Quarter'] = amazon_sales['Date'].dt.quarter
    amazon_sales['Revenue'] = amazon_sales['Amount']
    
    # Clean expenses data
    expenses.columns = ['index', 'Income_Particular', 'Income_Amount', 
                       'Expense_Particular', 'Expense_Amount']
    expenses = expenses.drop(0)
    expenses = expenses[expenses['Income_Particular'] != 'Total']
    expenses['Income_Amount'] = pd.to_numeric(expenses['Income_Amount'], errors='coerce')
    expenses['Expense_Amount'] = pd.to_numeric(expenses['Expense_Amount'], errors='coerce')
    
    return amazon_sales, international_sales, expenses, inventory

@st.cache_data
def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_revenue = df['Revenue'].sum()
    total_orders = df['Order ID'].nunique()
    total_units = df['Qty'].sum()
    avg_order_value = total_revenue / total_orders
    
    cancelled_revenue = df[df['Status'].str.contains('cancel', case=False, na=False)]['Revenue'].sum()
    cancelled_orders = df[df['Status'].str.contains('cancel', case=False, na=False)]['Order ID'].nunique()
    cancellation_rate = cancelled_orders / total_orders
    
    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'total_units': total_units,
        'avg_order_value': avg_order_value,
        'cancelled_revenue': cancelled_revenue,
        'cancellation_rate': cancellation_rate
    }

def train_ml_model(df):
    """Train linear regression model for revenue prediction"""
    # Create Revenue column from Amount
    df['Revenue'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Extract Month from Date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    
    # Drop rows with missing values in our features
    df_clean = df.dropna(subset=['Month', 'Qty'])
    
    # Prepare data for ML
    X = df_clean[['Month', 'Qty']]
    y = df_clean['Revenue']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df_clean

def main():
    # Load data
    amazon_sales, international_sales, expenses, inventory = load_data()
    kpis = calculate_kpis(amazon_sales)
    
    # Main header
    st.markdown('<h1 class="main-header">📊 E-commerce Business Intelligence</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Date filter
    min_date = amazon_sales['Date'].min()
    max_date = amazon_sales['Date'].max()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on date selection
    filtered_data = amazon_sales[
        (amazon_sales['Date'] >= pd.to_datetime(start_date)) & 
        (amazon_sales['Date'] <= pd.to_datetime(end_date))
    ]
    
    # Recalculate KPIs for filtered data
    filtered_kpis = calculate_kpis(filtered_data)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Business Intelligence", "🤖 ML Revenue Prediction", "📋 Data Explorer"])
    
    with tab1:
        st.markdown("## 📈 Business Intelligence Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Revenue</h3>
                <h2>₹{filtered_kpis['total_revenue']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Total Orders</h3>
                <h2>{filtered_kpis['total_orders']:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Avg Order Value</h3>
                <h2>₹{filtered_kpis['avg_order_value']:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>❌ Cancellation Rate</h3>
                <h2>{filtered_kpis['cancellation_rate']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Monthly Revenue Trend")
            monthly_rev = filtered_data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
            monthly_rev['Date'] = pd.to_datetime(monthly_rev['Year'].astype(str) + '-' + monthly_rev['Month'].astype(str))
            
            fig = px.line(monthly_rev, x='Date', y='Revenue', 
                         title='Monthly Revenue Trend',
                         labels={'Revenue': 'Revenue (₹)', 'Date': 'Month'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏷️ Top Categories by Revenue")
            category_rev = filtered_data.groupby('Category')['Revenue'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(x=category_rev.index, y=category_rev.values,
                        title='Top 10 Categories by Revenue',
                        labels={'x': 'Category', 'y': 'Revenue (₹)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗺️ Top States by Revenue")
            state_rev = filtered_data.groupby('ship-state')['Revenue'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(x=state_rev.index, y=state_rev.values,
                        title='Top 10 States by Revenue',
                        labels={'x': 'State', 'y': 'Revenue (₹)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Order Status Distribution")
            status_counts = filtered_data['Status'].value_counts()
            
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Order Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional Insights
        st.markdown("### 📊 Additional Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Growth Metrics**")
            monthly_rev['Growth_Rate'] = monthly_rev['Revenue'].pct_change() * 100
            avg_growth = monthly_rev['Growth_Rate'].mean()
            st.write(f"Average Monthly Growth: {avg_growth:.1f}%")
        
        with col2:
            st.markdown("**👥 Customer Analysis**")
            b2b_revenue = filtered_data.groupby('B2B')['Revenue'].sum()
            st.write(f"B2B Revenue: ₹{b2b_revenue.get(True, 0):,.0f}")
            st.write(f"B2C Revenue: ₹{b2b_revenue.get(False, 0):,.0f}")
        
        with col3:
            st.markdown("**📦 Product Insights**")
            top_sku = filtered_data.groupby('SKU')['Revenue'].sum().sort_values(ascending=False).head(1)
            if not top_sku.empty:
                st.write(f"Top SKU: {top_sku.index[0]}")
                st.write(f"Revenue: ₹{top_sku.iloc[0]:,.0f}")
    
    with tab2:
        st.markdown("## 🤖 ML Revenue Prediction Dashboard")
        
        # Train model using the reference approach
        model, df_clean = train_ml_model(amazon_sales)
        
        # Show data info
        st.write(f"Using {len(df_clean)} rows for training")
        
        # User input section
        st.subheader("Enter Prediction Details")
        col1, col2 = st.columns(2)
        
        with col1:
            month = st.slider("Month (1-12)", 1, 12, datetime.now().month)
        with col2:
            qty = st.number_input("Quantity", min_value=1, max_value=100, value=10)
        
        # Make prediction
        if st.button("Predict Revenue"):
            # Create DataFrame with proper column names
            input_data = pd.DataFrame([[month, qty]], columns=['Month', 'Qty'])
            prediction = model.predict(input_data)[0]
            
            st.success(f"**Predicted Revenue: ₹{prediction:,.2f}**")
            
            # Show some context
            st.info(f"Based on {len(df_clean)} previous transactions")
        
        # Data Overview
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        # Calculate metrics
        X = df_clean[['Month', 'Qty']]
        y = df_clean['Revenue']
        
        with col1:
            st.metric("Average Revenue", f"₹{y.mean():,.2f}")
        with col2:
            st.metric("Average Quantity", f"{X['Qty'].mean():.1f}")
        with col3:
            st.metric("Data Range", f"{int(X['Month'].min())}-{int(X['Month'].max())} months")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_names = ['Month', 'Quantity']
        coefficients = model.coef_
        
        fig = px.bar(x=feature_names, y=coefficients,
                    title='Feature Importance (Coefficients)',
                    labels={'x': 'Features', 'y': 'Coefficient Value'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions table
        st.subheader("Sample Predictions")
        sample_data = pd.DataFrame({
            'Month': [3, 6, 9, 12],
            'Qty': [1, 5, 10, 15]
        })
        sample_data['Predicted_Revenue'] = model.predict(sample_data[['Month', 'Qty']])
        st.dataframe(sample_data.style.format({'Predicted_Revenue': '₹{:.2f}'}))
    
    with tab3:
        st.markdown("## 📋 Data Explorer")
        
        # Data summary
        st.markdown("### 📊 Dataset Summary")
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
        
        # Data preview
        st.markdown("### 👀 Data Preview")
        
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
        
        # Data statistics
        if dataset_choice == "Amazon Sales":
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(amazon_sales.describe())

if __name__ == "__main__":
    main()
