import pandas as pd
import numpy as np
from scipy import stats

# Load the CSV files
file1_path = 'text_analysis_results_human.csv'
file2_path = 'text_analysis_results_LLM.csv'

# Read the files into dataframes
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Clean the data (remove commas and convert columns to numeric format)
def clean_data(df):
    for column in df.columns:
        # Try to convert to numeric, skip if it fails
        try:
            df[column] = df[column].replace({',': ''}, regex=True).astype(float)
        except ValueError:
            pass  # Skip columns that cannot be converted to numeric
    return df

# Clean both dataframes
df1_clean = clean_data(df1)
df2_clean = clean_data(df2)

# Select only rows for low, medium, or high bias (0 low, 1 medium, 2 high)
df1_filtered = df1_clean.iloc[[0, 3, 6]]
df2_filtered = df2_clean.iloc[[0, 3, 6]]

# Select only numeric columns
numeric_columns_df1 = df1_filtered.select_dtypes(include=['number']).columns
numeric_columns_df2 = df2_filtered.select_dtypes(include=['number']).columns

# Ensure both dataframes have the same numeric columns
common_columns = numeric_columns_df1.intersection(numeric_columns_df2)


# Calculate the differences between corresponding columns (for paired samples)
differences = df1_filtered[common_columns] - df2_filtered[common_columns]

# Perform paired t-tests + calculate CIs
results = []
alpha = 0.05

for column in common_columns:
    x = df1_filtered[column]
    y = df2_filtered[column]
    diff = x - y

    # Independent-samples t-test (Welch’s t-test)
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)

    # Mean difference
    mean_diff = diff.mean()

    # Standard error of the mean difference
    se = diff.std(ddof=1) / np.sqrt(len(diff))

    # Critical t-value
    df = len(diff) - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # 95% CI
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    results.append({
        "Difference": mean_diff,
        "P-value": p_value,
        "CI Lower": ci_low,
        "CI Upper": ci_high
    })


# Create a summary DataFrame
summary = pd.DataFrame(results, index=common_columns)

# Export
output_file = 'comparison_high_bias.csv'
summary.to_csv(output_file)

print(f"Comparison summary with 95% CI has been exported to {output_file}")