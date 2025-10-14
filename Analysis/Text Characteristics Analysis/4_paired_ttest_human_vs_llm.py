import pandas as pd
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

# Select only numeric columns
numeric_columns_df1 = df1_clean.select_dtypes(include=['number']).columns
numeric_columns_df2 = df2_clean.select_dtypes(include=['number']).columns

# Ensure both dataframes have the same numeric columns
common_columns = numeric_columns_df1.intersection(numeric_columns_df2)

# Initialize lists to store results
t_stats = []
p_values = []

# Loop over rows and perform paired t-test for each column in the row
for index, row in df1_clean.iterrows():
    # Get the corresponding row in df2
    row_df2 = df2_clean.iloc[index]
    
    # Compare values for the same row across both dataframes
    row_t_stats = []
    row_p_values = []
    
    for column in common_columns:
        t_stat, p_val = stats.ttest_rel([row[column]], [row_df2[column]])
        row_t_stats.append(t_stat)
        row_p_values.append(p_val)
    
    # Store the t-statistics and p-values for each row
    t_stats.append(row_t_stats)
    p_values.append(row_p_values)

# Convert results into a DataFrame
t_stat_df = pd.DataFrame(t_stats, columns=common_columns)
p_value_df = pd.DataFrame(p_values, columns=common_columns)

# Concatenate t-statistics and p-values for final result
final_result = pd.concat([t_stat_df, p_value_df], axis=1, keys=['T-Statistic', 'P-Value'])

# Export the summary table to a CSV file
output_file = 'comparison_rowwise_summary.csv'
final_result.to_csv(output_file)

print(f"Row-wise comparison summary has been exported to {output_file}")
