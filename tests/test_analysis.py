"""Unit tests for E-commerce BI analysis utilities."""

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from utils import (
    compute_rfm,
    engineer_features,
    get_classifiers,
    hypothesis_tests,
    join_sales_with_inventory,
    segment_customers,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create a minimal sales DataFrame for testing."""
    return pd.DataFrame({
        'Order ID': ['A1', 'A1', 'B2', 'C3', 'D4'],
        'Date': pd.to_datetime(['2022-01-01', '2022-01-15', '2022-02-01', '2022-03-01', '2022-01-10']),
        'Status': ['Shipped', 'Shipped', 'Cancelled', 'Shipped', 'Cancelled'],
        'Amount': [100.0, 200.0, 50.0, 300.0, 75.0],
        'Qty': [1, 2, 1, 3, 1],
        'Fulfilment': ['Amazon', 'Merchant', 'Amazon', 'Merchant', 'Amazon'],
        'ship-service-level': ['Standard', 'Expedited', 'Standard', 'Standard', 'Expedited'],
        'Category': ['Set', 'Kurta', 'Set', 'Top', 'Kurta'],
        'B2B': [False, True, False, False, True],
        'promotion-ids': [None, 'PROMO1', None, 'PROMO2', None],
        'ship-state': ['MH', 'KA', 'TN', 'MH', 'DL'],
        'Size': ['M', 'L', 'S', 'XL', 'M'],
        'SKU': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'Style': ['ST1', 'ST2', 'ST3', 'ST4', 'ST5'],
        'ASIN': ['ASIN1', 'ASIN2', 'ASIN3', 'ASIN4', 'ASIN5'],
        'Courier Status': ['Delivered', 'Delivered', 'Cancelled', 'Shipped', 'Cancelled'],
        'currency': ['INR', 'INR', 'INR', 'INR', 'INR'],
        'ship-postal-code': [400001, 560001, 600001, 400002, 110001],
        'ship-country': ['IN', 'IN', 'IN', 'IN', 'IN'],
        'fulfilled-by': ['Amazon', 'Merchant', 'Amazon', 'Merchant', 'Easy Ship'],
    })


@pytest.fixture
def sample_inventory() -> pd.DataFrame:
    """Create a minimal inventory DataFrame for testing."""
    return pd.DataFrame({
        'SKU Code': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'Design No.': ['D1', 'D2', 'D3', 'D4', 'D5'],
        'Stock': [10.0, 5.0, 0.0, 20.0, 3.0],
        'Category': ['Set', 'Kurta', 'Set', 'Top', 'Kurta'],
        'Size': ['M', 'L', 'S', 'XL', 'M'],
        'Color': ['Red', 'Blue', 'Green', 'Black', 'White'],
    })


class TestEngineerFeatures:
    """Tests for feature engineering."""

    def test_adds_expected_columns(self, sample_sales: pd.DataFrame) -> None:
        result = engineer_features(sample_sales)
        expected = {'DayOfWeek', 'IsWeekend', 'PricePerUnit', 'HasPromotion',
                    'IsHighValue', 'Revenue_7d_MA', 'Cancelled'}
        assert expected.issubset(result.columns), f"Missing: {expected - set(result.columns)}"

    def test_price_per_unit(self, sample_sales: pd.DataFrame) -> None:
        result = engineer_features(sample_sales)
        assert result['PricePerUnit'].iloc[0] == 100.0  # 100 / 1

    def test_cancelled_flag(self, sample_sales: pd.DataFrame) -> None:
        result = engineer_features(sample_sales)
        assert result['Cancelled'].iloc[2] == 1
        assert result['Cancelled'].iloc[0] == 0


class TestComputeRFM:
    """Tests for RFM computation."""

    def test_rfm_shape(self, sample_sales: pd.DataFrame) -> None:
        rfm = compute_rfm(sample_sales)
        assert len(rfm) == sample_sales['Order ID'].nunique()
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns

    def test_rfm_values(self, sample_sales: pd.DataFrame) -> None:
        rfm = compute_rfm(sample_sales)
        order_a1 = rfm[rfm['OrderID'] == 'A1']
        assert order_a1['Frequency'].iloc[0] == 2
        assert order_a1['Monetary'].iloc[0] == 300.0


class TestSegmentCustomers:
    """Tests for K-means customer segmentation."""

    def test_segment_output(self, sample_sales: pd.DataFrame) -> None:
        rfm = compute_rfm(sample_sales)
        segmented, _ = segment_customers(rfm, n_clusters=2)
        assert 'Segment' in segmented.columns
        assert 'Segment_Label' in segmented.columns
        assert segmented['Segment'].nunique() == 2


class TestJoinSalesWithInventory:
    """Tests for dataset joining."""

    def test_join_adds_columns(self, sample_sales: pd.DataFrame, sample_inventory: pd.DataFrame) -> None:
        result = join_sales_with_inventory(sample_sales, sample_inventory)
        assert 'Total_Stock' in result.columns
        assert 'Unique_SKUs' in result.columns

    def test_join_stock_values(self, sample_sales: pd.DataFrame, sample_inventory: pd.DataFrame) -> None:
        result = join_sales_with_inventory(sample_sales, sample_inventory)
        set_rows = result[result['Category'] == 'Set']
        assert set_rows['Total_Stock'].iloc[0] == 10.0


class TestGetClassifiers:
    """Tests for model factory."""

    def test_returns_three_models(self) -> None:
        classifiers = get_classifiers()
        assert len(classifiers) == 3

    def test_keys_are_strings(self) -> None:
        classifiers = get_classifiers()
        for name in classifiers:
            assert isinstance(name, str)


class TestHypothesisTests:
    """Tests for statistical hypothesis testing."""

    def test_returns_dict(self, sample_sales: pd.DataFrame) -> None:
        results = hypothesis_tests(sample_sales)
        assert isinstance(results, dict)

    def test_has_expected_keys(self, sample_sales: pd.DataFrame) -> None:
        results = hypothesis_tests(sample_sales)
        if results:
            for key in results:
                assert 'p-value' in results[key]
                assert 'significant' in results[key]
