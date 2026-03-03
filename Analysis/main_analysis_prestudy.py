"""
Pre-study analysis for media bias explanations study.
Evaluates two LLM prompting strategies (role priming vs. definition prompt)
on medium-bias articles with left/right political slant.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ── Data loading and preparation ──────────────────────────────────────────────

df = pd.read_csv("../Survey Results/Prestudy Results.csv")
df.columns.values[13] = "group_number"
df.columns.values[9] = "prolific_id"


def remove_ids(dataframe, ids_to_remove):
    """Drop rows whose prolific_id is in *ids_to_remove*."""
    return dataframe[~dataframe["prolific_id"].isin(ids_to_remove)]


# Exclude known problematic participants
df = remove_ids(df, ["60469275f75cd607f61fb973", "66187caf2a94f9051c85c431"])

# Map group numbers to experimental conditions
# Groups 1-4 cover right/left × LLM_simple/LLM_complex
mapping = {
    1: ("right", "medium", "LLM_simple"),
    2: ("right", "medium", "LLM_complex"),
    3: ("left", "medium", "LLM_simple"),
    4: ("left", "medium", "LLM_complex"),
}
df[["article_orientation", "article_bias_level", "explanation_type"]] = (
    df["group_number"].map(mapping).apply(pd.Series)
)

# Rename long survey column names to something workable
df = df.rename(
    columns={
        "After reading the explanation, please tell us what you think about the following sentence: In my opinion, this article is biased.": "bias_t2",
        "Please tell us what you think about the following sentence: In my opinion, this article is biased.": "bias_t1",
        "Do you consider yourself to be liberal, conservative or somewhere in between?   [Political Orientation|Liberal|Conservative]": "Participant_Political_Lean",
        "The explanation was useful. [agree|disagree]": "Usefulness",
        "The explanation was complete. [agree|disagree]": "Completeness",
    }
)

relevant_columns = [
    "article_orientation",
    "article_bias_level",
    "explanation_type",
    "bias_t1",
    "bias_t2",
    "Participant_Political_Lean",
    "Usefulness",
    "Completeness",
]
df_relevant = df[relevant_columns].copy()

# ── Convert Likert responses to numeric ───────────────────────────────────────

for col in ("bias_t1", "bias_t2"):
    df_relevant[col] = (
        df_relevant[col]
        .str.replace("\u200b", "", regex=True)  # strip zero-width spaces
        .str.strip()
    )

bias_scale = {
    "Strongly disagree": -3,
    "Disagree": -2,
    "Somewhat disagree": -1,
    "Somewhat agree": 1,
    "Agree": 2,
    "Strongly agree": 3,
}

df_relevant["bias_t1_num"] = df_relevant["bias_t1"].map(bias_scale)
df_relevant["bias_t2_num"] = df_relevant["bias_t2"].map(bias_scale)
df_relevant["biaschange"] = df_relevant["bias_t2_num"] - df_relevant["bias_t1_num"]

# ── Descriptive statistics ────────────────────────────────────────────────────

print("=== Overall bias-change statistics ===")
print(f"  Mean:   {df_relevant['biaschange'].mean():.3f}")
print(f"  Median: {df_relevant['biaschange'].median():.1f}")
print(f"  SD:     {df_relevant['biaschange'].std():.4f}")
print(f"  Positive changes: {(df_relevant['biaschange'] > 0).sum()}")
print(f"  Negative changes: {(df_relevant['biaschange'] < 0).sum()}")
print(f"  Zero changes:     {(df_relevant['biaschange'] == 0).sum()}")

# ── Visualisations ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_relevant["biaschange"], kde=True, bins=15, color="skyblue", ax=ax)
ax.set_title("Distribution of Bias Change from t1 to t2")
ax.set_xlabel("Bias Change")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("prestudy_biaschange_histogram.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x=df_relevant["biaschange"], color="lightcoral", ax=ax)
ax.set_title("Boxplot of Bias Change from t1 to t2")
ax.set_xlabel("Bias Change")
plt.tight_layout()
plt.savefig("prestudy_biaschange_boxplot.pdf", bbox_inches="tight")
plt.show()

# ── H1: Does explanation technique influence bias change? ─────────────────────
# IV: explanation_type (LLM_complex vs LLM_simple)
# DV: biaschange

df_relevant["explanation_type"] = pd.Categorical(df_relevant["explanation_type"])
print("\nExplanation types:", df_relevant["explanation_type"].unique().tolist())

model = ols("biaschange ~ C(explanation_type)", data=df_relevant).fit()
anova_results = anova_lm(model, typ=2)
print("\n=== ANOVA: explanation_type → biaschange ===")
print(anova_results)

# Group means
mean_by_type = df_relevant.groupby("explanation_type")["biaschange"].mean()
print("\nMean bias change by explanation type:")
print(mean_by_type)

# Post-hoc comparison (Tukey HSD) — run regardless for reporting
tukey = pairwise_tukeyhsd(
    endog=df_relevant["biaschange"],
    groups=df_relevant["explanation_type"],
    alpha=0.05,
)
print("\n=== Tukey HSD ===")
print(tukey.summary())

# Boxplot by explanation technique
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="explanation_type", y="biaschange", data=df_relevant, ax=ax)
ax.set_title("Comparison of Bias Change by Explanation Technique")
ax.set_xlabel("Explanation Technique")
ax.set_ylabel("Bias Change")
plt.tight_layout()
plt.savefig("prestudy_biaschange_by_technique.pdf", bbox_inches="tight")
plt.show()
