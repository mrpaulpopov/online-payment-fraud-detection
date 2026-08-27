import logging
import sys

import numpy as np
import pandas as pd

from src.data.data_loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)


def calculate_psi(expected, actual, bins=10):
    expected_series = pd.Series(expected).dropna()
    actual_series = pd.Series(actual).dropna()

    if pd.api.types.is_numeric_dtype(expected_series):
        # числовая колонка -> делим на bins по границам train
        try:
            _, bin_edges = pd.qcut(expected_series, q=bins, retbins=True, duplicates='drop')
        except ValueError:
            bin_edges = np.linspace(expected_series.min(), expected_series.max(), bins + 1)

        expected_binned = pd.cut(expected_series, bins=bin_edges, include_lowest=True)
        actual_binned = pd.cut(actual_series, bins=bin_edges, include_lowest=True)

        expected_perc = expected_binned.value_counts(normalize=True, sort=False).sort_index()
        actual_perc = (
            actual_binned.value_counts(normalize=True, sort=False)
            .reindex(expected_perc.index, fill_value=0)
        )
    else:
        # категориальная колонка -> бины = сами категории
        expected_perc = expected_series.value_counts(normalize=True)
        actual_perc = actual_series.value_counts(normalize=True).reindex(expected_perc.index, fill_value=0)

    p = np.where(actual_perc.values == 0, 1e-4, actual_perc.values)
    q = np.where(expected_perc.values == 0, 1e-4, expected_perc.values)

    return float(np.sum((p - q) * np.log(p / q)))


def get_overall_psi(train_df, test_df, bins=10):
    common_cols = [c for c in train_df.columns if c in test_df.columns]

    psi_values = []
    for col in common_cols:
        psi_values.append(calculate_psi(train_df[col], test_df[col], bins=bins))

    return float(np.mean(psi_values)) if psi_values else 0.0


def get_top_drifted_features(train_df, test_df, bins=10, top_n=5):
    common_cols = [c for c in train_df.columns if c in test_df.columns]

    psi_per_feature = {}
    for col in common_cols:
        psi_per_feature[col] = calculate_psi(train_df[col], test_df[col], bins=bins)

    return sorted(psi_per_feature.items(), key=lambda item: -item[1])[:top_n]


def main():
    X_trainval, _, _ = load_data(table_name='train_final_features')
    X_test, _, _ = load_data(table_name='test_final_features')

    train_df = pd.DataFrame(data=X_trainval)
    test_df = pd.DataFrame(data=X_test)

    overall_psi = get_overall_psi(train_df, test_df)
    logging.info(f"Overall PSI: {overall_psi:.4f}")

    if overall_psi > 0.1:
        logging.info("Data drift detected (PSI > 0.1). Top features by PSI:")
        for col, val in get_top_drifted_features(train_df, test_df):
            print(f"{col}: {val:.4f}")
    else:
        logging.info("Data drift is within normal limits (PSI < 0.1).")


if __name__ == '__main__':
    main()