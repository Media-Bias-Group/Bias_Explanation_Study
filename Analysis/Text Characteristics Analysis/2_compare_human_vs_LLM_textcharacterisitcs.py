import pandas as pd
from scipy import stats

# Load the CSV files
file1_path = '/text_analysis_results_human.csv'
file2_path = '/text_analysis_results_LLM.csv'

# Read the files into dataframes
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Show the first few rows to understand the structure of the data
df1.head(), df2.head()

# Remove commas and convert columns to numeric format, excluding non-numeric columns
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

# Print the first row of the cleaned dataframes
print("First row of df1_clean:")
print(df1_clean.iloc[0])

print("\nFirst row of df2_clean:")
print(df2_clean.iloc[0])

# Select only numeric columns
numeric_columns_df1 = df1_clean.select_dtypes(include=['number']).columns
numeric_columns_df2 = df2_clean.select_dtypes(include=['number']).columns

# Ensure both dataframes have the same numeric columns
common_columns = numeric_columns_df1.intersection(numeric_columns_df2)

# Calculate the differences between corresponding columns
differences = df1_clean[common_columns] - df2_clean[common_columns]

# Perform statistical significance tests (using a paired t-test for each column)
p_values = {}
for column in common_columns:
    t_stat, p_value = stats.ttest_rel(df1_clean[column], df2_clean[column])
    p_values[column] = p_value

# Create a summary table for differences and p-values
summary = pd.DataFrame({
    'Difference': differences.mean(),  # Average difference across all rows for each column
    'P-value': p_values
})

# Export the summary table to a CSV file
output_file = 'comparison_summary.csv'
summary.to_csv(output_file)

print(f"Comparison summary has been exported to {output_file}")
