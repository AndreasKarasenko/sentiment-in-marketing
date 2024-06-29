import pandas as pd
from io import StringIO

data = "./results/overview/summary_all_F1.csv"
df = pd.read_csv(data)
df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
df.dropna(inplace=True)

sort_cols = list(df.columns)
sort_cols.remove("Model")

df.sort_values(by=sort_cols, ascending=False)

df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
df_melted["Score"] = (df_melted["Score"]).round(2)

# Create a new column combining Model and Score
df_melted['Model_Score'] = df_melted['Model'] + ' (' + df_melted['Score'].astype(str) + ')'

# Sort by Score in descending order and reset index
df_melted = df_melted.sort_values(['Metric', 'Score'], ascending=[True, False]).reset_index(drop=True)

# Create a rank column
df_melted['Rank'] = df_melted.groupby('Metric').cumcount() + 1

# Pivot the dataframe
df_pivot = df_melted.pivot(index='Rank', columns='Metric', values='Model_Score')
df_pivot.to_excel("./results/overview/overall_scores.xlsx")