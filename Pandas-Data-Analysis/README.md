# Pandas EDA Project

This project loads a CSV dataset with Pandas, performs basic exploratory data analysis, and creates three Matplotlib/Seaborn visualizations:

- Bar chart comparing average target values across categories
- Scatter plot showing the relationship between two numerical variables
- Correlation heatmap for numerical features

## Project Structure

```text
.
├── data/
│   └── sample_sales.csv
├── eda_analysis.py
├── requirements.txt
└── README.md
```

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run With the Sample Dataset

```bash
python eda_analysis.py --csv data/sample_sales.csv --target Revenue
```

The script prints:

- First few rows
- Dataset shape
- Column data types
- Missing values
- Summary statistics
- Average, median, minimum, and maximum values
- Target-column statistics
- Written insights and recommendations

It also saves charts into the `outputs/` folder:

- `bar_chart.png`
- `scatter_plot.png`
- `correlation_heatmap.png`

## Run With Your Own Dataset

```bash
python eda_analysis.py --csv path/to/your_dataset.csv --target YourTargetColumn
```

Optional arguments:

```bash
python eda_analysis.py \
  --csv path/to/your_dataset.csv \
  --target Revenue \
  --category Region \
  --x Marketing_Spend \
  --y Revenue
```

## Notes

The script automatically chooses useful defaults:

- Target column: `Revenue` if available, otherwise the first numerical column
- Category column: first text/category column
- Scatter plot x-axis: first numerical column
- Scatter plot y-axis: selected target column

For best results, use a dataset with at least one categorical column and two numerical columns.
