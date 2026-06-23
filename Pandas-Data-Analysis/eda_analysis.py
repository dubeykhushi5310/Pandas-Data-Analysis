"""
Basic Exploratory Data Analysis project.

Run:
    python eda_analysis.py --csv data/sample_sales.csv --target Revenue

The script:
    - Loads a CSV dataset with pandas
    - Displays first rows, missing values, and summary statistics
    - Calculates average, median, minimum, and maximum for numeric columns
    - Creates a bar chart, scatter plot, and correlation heatmap
    - Prints practical observations based on the analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic EDA on a CSV dataset.")
    parser.add_argument(
        "--csv",
        default="data/sample_sales.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Numerical target column to analyze. Defaults to Revenue if present, otherwise the first numeric column.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Categorical column for the bar chart. Defaults to the first categorical column.",
    )
    parser.add_argument(
        "--x",
        default=None,
        help="Numerical column for the scatter plot x-axis.",
    )
    parser.add_argument(
        "--y",
        default=None,
        help="Numerical column for the scatter plot y-axis.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where charts will be saved.",
    )
    return parser.parse_args()


def choose_target(df: pd.DataFrame, numeric_cols: list[str], requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target column '{requested}' was not found in the dataset.")
        if requested not in numeric_cols:
            raise ValueError(f"Target column '{requested}' must be numerical.")
        return requested

    if "Revenue" in numeric_cols:
        return "Revenue"

    if not numeric_cols:
        raise ValueError("No numerical columns were found in the dataset.")

    return numeric_cols[0]


def print_basic_eda(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> None:
    print("\n=== First Few Rows ===")
    print(df.head())

    print("\n=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n=== Column Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Summary Statistics ===")
    print(df.describe(include="all"))

    stats = pd.DataFrame(
        {
            "Average": df[numeric_cols].mean(),
            "Median": df[numeric_cols].median(),
            "Minimum": df[numeric_cols].min(),
            "Maximum": df[numeric_cols].max(),
        }
    )

    print("\n=== Average, Median, Minimum, and Maximum ===")
    print(stats)

    print(f"\n=== Selected Target Column: {target_col} ===")
    print(f"Average: {df[target_col].mean():,.2f}")
    print(f"Median: {df[target_col].median():,.2f}")
    print(f"Minimum: {df[target_col].min():,.2f}")
    print(f"Maximum: {df[target_col].max():,.2f}")


def create_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
    output_dir: Path,
) -> None:
    category_summary = df.groupby(category_col, dropna=False)[target_col].mean().sort_values()

    plt.figure(figsize=(10, 6))
    category_summary.plot(kind="bar", color="#4C78A8", edgecolor="black", label=f"Average {target_col}")
    plt.title(f"Average {target_col} by {category_col}")
    plt.xlabel(category_col)
    plt.ylabel(f"Average {target_col}")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_chart.png", dpi=150)
    plt.close()

    top_category = category_summary.idxmax()
    bottom_category = category_summary.idxmin()
    print(
        f"\nBar chart insight: {top_category} has the highest average {target_col}, "
        f"while {bottom_category} has the lowest average {target_col}."
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_dir: Path,
) -> None:
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.75, color="#59A14F", label="Records")
    plt.title(f"Relationship Between {x_col} and {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plot.png", dpi=150)
    plt.close()

    direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "weak or no"
    print(
        f"\nScatter plot insight: {x_col} and {y_col} show a {direction} linear relationship "
        f"with a correlation of {correlation:.2f}."
    )


def create_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nHeatmap insight: Strong values near 1 indicate features that rise together.")
    print("Values near -1 indicate inverse relationships, and values near 0 indicate weak linear relationships.")

    return correlation_matrix


def detect_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    outlier_counts = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    return pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["Potential Outliers"])


def get_categorical_columns(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    categorical_cols = []

    for col in df.columns:
        dtype = df[col].dtype
        if col not in numeric_cols and (pd.api.types.is_string_dtype(dtype) or isinstance(dtype, CategoricalDtype)):
            categorical_cols.append(col)

    return categorical_cols


def print_final_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str,
    correlation_matrix: pd.DataFrame,
) -> None:
    print("\n=== Detailed Insights and Observations ===")

    target_mean = df[target_col].mean()
    target_median = df[target_col].median()
    skew_note = "right-skewed" if target_mean > target_median else "left-skewed or balanced"

    print(f"\nTrends and patterns:")
    print(f"- The target column '{target_col}' has an average of {target_mean:,.2f} and a median of {target_median:,.2f}.")
    print(f"- This suggests the target distribution is {skew_note}.")

    target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
    strong_positive = target_correlations[target_correlations >= 0.7]
    strong_negative = target_correlations[target_correlations <= -0.7]

    print("\nStrong correlations:")
    if not strong_positive.empty:
        print("- Strong positive correlations with the target:")
        print(strong_positive)
    else:
        print("- No strong positive correlations with the target were found.")

    if not strong_negative.empty:
        print("- Strong negative correlations with the target:")
        print(strong_negative)
    else:
        print("- No strong negative correlations with the target were found.")

    print("\nPotential outliers or anomalies:")
    outliers = detect_outliers(df, numeric_cols)
    print(outliers)

    print("\nBusiness or practical recommendations:")
    print("- Prioritize categories and products associated with higher average target values.")
    print("- Investigate columns strongly correlated with the target because they may be useful for forecasting.")
    print("- Review potential outliers before removing them; they may represent data errors or important edge cases.")
    print("- Handle missing values before using the dataset for reporting, dashboards, or machine learning.")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = get_categorical_columns(df, numeric_cols)

    target_col = choose_target(df, numeric_cols, args.target)

    print_basic_eda(df, numeric_cols, target_col)

    if categorical_cols:
        category_col = args.category if args.category else categorical_cols[0]
        if category_col not in df.columns:
            raise ValueError(f"Category column '{category_col}' was not found in the dataset.")
        create_bar_chart(df, category_col, target_col, output_dir)
    else:
        print("\nBar chart skipped: no categorical columns were found.")

    if len(numeric_cols) >= 2:
        x_col = args.x if args.x else numeric_cols[0]
        y_col = args.y if args.y else target_col

        if x_col not in numeric_cols:
            raise ValueError(f"Scatter x-axis column '{x_col}' must be numerical.")
        if y_col not in numeric_cols:
            raise ValueError(f"Scatter y-axis column '{y_col}' must be numerical.")

        create_scatter_plot(df, x_col, y_col, output_dir)
        correlation_matrix = create_correlation_heatmap(df, numeric_cols, output_dir)
        print_final_insights(df, numeric_cols, target_col, correlation_matrix)
    else:
        print("\nScatter plot, heatmap, and correlation insights skipped: fewer than two numerical columns were found.")

    print(f"\nCharts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
