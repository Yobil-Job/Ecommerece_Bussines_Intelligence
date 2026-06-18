# E-commerce Business Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecommerecebussinesintelligence.streamlit.app/)

A production-grade **Business Intelligence and Machine Learning** portfolio project demonstrating data analysis, predictive modeling, customer segmentation, and interactive dashboards using real-world e-commerce data from Amazon India.

## Live Demo

https://ecommerecebussinesintelligence.streamlit.app/

---

## Features

### Business Intelligence Dashboard
- **Revenue analysis** with monthly trends, category breakdowns, and geographic distribution
- **Statistical hypothesis testing** (Welch's t-test for B2B vs B2C AOV, chi-square for fulfillment vs cancellation)
- **KPI tracking** with date-range filtering: Total Revenue, Orders, AOV, Cancellation Rate
- **Dataset joins** — sales transactions merged with inventory data

### Customer Segmentation (RFM + K-means)
- **Recency, Frequency, Monetary** (RFM) analysis per customer
- **K-means clustering** on log-transformed RFM values with automated segment labeling
- **3D visualization** of customer segments
- **Actionable insights** for each segment (Champions, Loyal, At Risk, Big Spenders)

### ML Model Comparison
- **3 classifiers** compared with GridSearchCV hyperparameter tuning:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Production-grade pipeline** using `sklearn.Pipeline` + `ColumnTransformer`
- **Full metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Feature importance** analysis and interactive cancellation risk prediction

### SQL Analysis
- **In-memory SQLite database** populated from CSV data
- **Pre-built analytical queries** (revenue by category, monthly trends, cancellation rates)
- **Custom SQL editor** for ad-hoc exploration
- **CSV export** for query results

### Data Explorer
- Browse all 5 datasets with row/column statistics
- Preview data and summary statistics

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (interactive dashboard) |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Machine Learning** | Logistic Regression, Random Forest, Gradient Boosting + GridSearchCV |
| **Statistics** | SciPy (t-tests, chi-square) |
| **Database** | SQLite (in-memory analytical queries) |
| **Visualization** | Plotly Express (interactive charts), Streamlit native |
| **Testing** | Pytest |
| **Model Persistence** | Joblib |

## Project Structure

```
├── app.py                    # Main Streamlit application
├── utils.py                  # Core utilities (data loading, features, models, stats)
├── tests/
│   └── test_analysis.py      # Unit tests for analysis functions
├── requirements.txt          # Production dependencies
├── requirements_minimal.txt  # Minimal dependencies for deployment
└── .gitignore
```

## Key Skills Demonstrated

- **Data Engineering**: Multi-dataset loading, cleaning, joining, type conversion
- **Feature Engineering**: Temporal features, price-per-unit, rolling averages, promotion flags
- **Machine Learning**: Pipeline construction, cross-validation, hyperparameter tuning, model comparison
- **Statistics**: Hypothesis testing, distribution analysis
- **Customer Analytics**: RFM analysis, K-means clustering, segment profiling
- **SQL**: Analytical queries, data aggregation, in-memory database
- **Software Engineering**: Modular code, type hints, unit testing, git version control
- **Deployment**: Streamlit Cloud, environment management

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Yobil-Job/Ecommerece_Bussines_Intelligence.git
cd Ecommerece_Bussines_Intelligence

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run tests
pytest tests/
```

## Contact

Available for consulting and full-time roles in **Data Analytics**, **Business Intelligence**, and **Machine Learning**.

*Built with Python, Streamlit, scikit-learn, and love for data.*
