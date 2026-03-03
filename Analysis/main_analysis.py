"""
Main study analysis for "Plausibility Machines: How LLM-Generated
Explanations Shape Media Bias Perception".

Covers:
  - Data loading & cleaning
  - Quantitative hypothesis tests (H1-H4)
  - UEQ and user experience analysis (H3)
  - Qualitative thematic analysis of open-ended responses
  - Figures
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import ttest_ind
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import re
from math import sqrt

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("../Survey Results/Study Results.csv", delimiter=";")
df.rename(columns={df.columns[13]: "group_number"}, inplace=True)

# ── Group → condition mapping (27 groups: 3 slants × 3 bias levels × 3 explanation types)
mapping = {
    1: ("left", "low", "human"),
    2: ("left", "low", "LLM"),
    3: ("left", "low", "control"),
    4: ("left", "medium", "human"),
    5: ("left", "medium", "LLM"),
    6: ("left", "medium", "control"),
    7: ("left", "high", "human"),
    8: ("left", "high", "LLM"),
    9: ("left", "high", "control"),
    10: ("center", "low", "human"),
    11: ("center", "low", "LLM"),
    12: ("center", "low", "control"),
    13: ("center", "medium", "human"),
    14: ("center", "medium", "LLM"),
    15: ("center", "medium", "control"),
    16: ("center", "high", "human"),
    17: ("center", "high", "LLM"),
    18: ("center", "high", "control"),
    19: ("right", "low", "human"),
    20: ("right", "low", "LLM"),
    21: ("right", "low", "control"),
    22: ("right", "medium", "human"),
    23: ("right", "medium", "LLM"),
    24: ("right", "medium", "control"),
    25: ("right", "high", "human"),
    26: ("right", "high", "LLM"),
    27: ("right", "high", "control"),
}

df[["article_orientation", "article_bias_level", "explanation_type"]] = (
    df["group_number"].map(mapping).apply(pd.Series)
)

# ── Column renaming ───────────────────────────────────────────────────────────

df = df.rename(columns={
    "After reading the short text after the article, please tell us what you think about the following sentence: In my opinion, this article is biased.": "bias_t2",
    "Please tell us what you think about the following sentence: In my opinion, this article is biased.": "bias_t1",
    "Do you consider yourself to be liberal, conservative or somewhere in between?   [Political Orientation|Liberal|Conservative]": "Participant_Political_Lean",
    "The explanation was useful. [agree|disagree]": "Usefulness",
    "The explanation was complete. [agree|disagree]": "Completeness",
})

# ── UEQ score computation ────────────────────────────────────────────────────
df["valuable_rc"] = -1 * df["The explanation was… [valuable|inferior]"]
df["good_rc"] = -1 * df["The explanation was… [good|bad]"]
df["clear_rc"] = -1 * df["The explanation was… [clear|confusing]"]

ueq_items = [
    "The explanation was… [not understandable|understandable]",
    "valuable_rc",
    "The explanation was… [obstructive|supportive]",
    "good_rc",
    "The explanation was… [complicated|easy]",
    "clear_rc",
]
df["UEQ"] = df[ueq_items].mean(axis=1)

# ── Likert encoding ───────────────────────────────────────────────────────────

BIAS_SCALE = {
    "Strongly disagree": -3,
    "Disagree": -2,
    "Somewhat disagree": -1,
    "Somewhat agree": 1,
    "Agree": 2,
    "Strongly agree": 3,
}

for col in ("bias_t1", "bias_t2"):
    df[col] = df[col].astype(str).str.replace("\u200b", "", regex=True).str.strip()

df["bias_t1_num"] = df["bias_t1"].map(BIAS_SCALE)
df["bias_t2_num"] = df["bias_t2"].map(BIAS_SCALE)
df["BiasChange"] = df["bias_t2_num"] - df["bias_t1_num"]
df["biaschange_abs"] = df["BiasChange"].abs()

# ── Political congruence ──────────────────────────────────────────────────────

def compute_congruence(row):
    lean = row["Participant_Political_Lean"]
    orient = row["article_orientation"]
    if pd.isna(lean) or pd.isna(orient):
        return np.nan
    if orient == "center":
        return "neutral"
    if lean <= 10 and orient == "left":
        return "congruent"
    if lean > 10 and orient == "right":
        return "congruent"
    return "incongruent"

df["congruence"] = df.apply(compute_congruence, axis=1)

# ── Cast categorical types ───────────────────────────────────────────────────
df["explanation_type"] = df["explanation_type"].astype("category")
df["article_bias_level"] = df["article_bias_level"].astype("category")

# ── Build relevant subset ────────────────────────────────────────────────────
relevant_columns = [
    "article_orientation", "article_bias_level", "explanation_type",
    "bias_t1", "bias_t2", "bias_t1_num", "bias_t2_num",
    "BiasChange", "biaschange_abs",
    "Participant_Political_Lean", "Usefulness", "Completeness",
    "UEQ", "congruence",
]
df_relevant = df[[c for c in relevant_columns if c in df.columns]].copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

mean_bias_change = df_relevant['BiasChange'].mean()
median_bias_change = df_relevant['BiasChange'].median()
std_bias_change = df_relevant['BiasChange'].std()
count_positive = len(df_relevant[df_relevant['BiasChange'] > 0])
count_negative = len(df_relevant[df_relevant['BiasChange'] < 0])
count_zero = len(df_relevant[df_relevant['BiasChange'] == 0])

print(f"Mean bias change: {mean_bias_change}")
print(f"Median bias change: {median_bias_change}")
print(f"Standard deviation: {std_bias_change}")
print(f"Count of positive changes: {count_positive}")
print(f"Count of negative changes: {count_negative}")
print(f"Count of zero changes: {count_zero}")

# ── Histogram of bias change ──────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.histplot(df_relevant['BiasChange'], kde=True, bins=15, color='skyblue')
plt.title('Distribution of Bias Change from t1 to t2')
plt.xlabel('Bias Change')
plt.ylabel('Frequency')
plt.savefig("biaschange_histogram.pdf", bbox_inches="tight")
plt.show()

# ── Boxplot of bias change ────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_relevant['BiasChange'], color='lightcoral')
plt.title('Boxplot of Bias Change from t1 to t2')
plt.xlabel('Bias Change')
plt.savefig("biaschange_boxplot.pdf", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("HYPOTHESIS TESTING")
print("=" * 80)

# ---------- H1: Explanations vs. control — magnitude of change ----------------

explanation_mask = df_relevant["explanation_type"].isin(["human", "LLM"])
control_mask = df_relevant["explanation_type"] == "control"

explained = df_relevant.loc[explanation_mask, "biaschange_abs"]
control = df_relevant.loc[control_mask, "biaschange_abs"]

t_h1, p_h1 = ttest_ind(explained, control, alternative="greater")
d_h1 = pg.compute_effsize(explained, control, eftype="cohen")
print("\n--- H1: Explanations vs Control (magnitude of change) ---")
print(f"  Explanation group: M={explained.mean():.3f}, n={len(explained)}")
print(f"  Control group:     M={control.mean():.3f}, n={len(control)}")
print(f"  t={t_h1:.3f}, p={p_h1:.6f}, Cohen's d={d_h1:.3f}")

for expl_type in ("LLM", "human"):
    grp = df_relevant.loc[df_relevant["explanation_type"] == expl_type, "biaschange_abs"]
    t, p = ttest_ind(grp, control, alternative="greater")
    d = pg.compute_effsize(grp, control, eftype="cohen")
    print(f"  {expl_type} vs control: t={t:.3f}, p={p:.6f}, d={d:.3f}")


# ---------- H2: LLM vs Human — magnitude of change ----------------------------

llm = df_relevant.loc[df_relevant["explanation_type"] == "LLM", "biaschange_abs"]
human = df_relevant.loc[df_relevant["explanation_type"] == "human", "biaschange_abs"]

t_h2, p_h2 = ttest_ind(llm, human)
d_h2 = pg.compute_effsize(llm, human, eftype="cohen")
print("\n--- H2: LLM vs Human (magnitude of change) ---")
print(f"  LLM:   M={llm.mean():.3f}, n={len(llm)}")
print(f"  Human: M={human.mean():.3f}, n={len(human)}")
print(f"  t={t_h2:.3f}, p={p_h2:.6f}, d={d_h2:.3f}")


# ---------- H3: UEQ comparison (LLM vs Human) ---------------------------------

print("\n--- H3: UEQ and User Experience (LLM vs Human) ---")

df_expl = df_relevant[df_relevant["explanation_type"].isin(["LLM", "human"])].copy()

ueq_llm = df_expl.loc[df_expl["explanation_type"] == "LLM", "UEQ"].dropna()
ueq_human = df_expl.loc[df_expl["explanation_type"] == "human", "UEQ"].dropna()
t_ueq, p_ueq = ttest_ind(ueq_llm, ueq_human)
print(f"  UEQ — LLM: M={ueq_llm.mean():.3f}, Human: M={ueq_human.mean():.3f}")
print(f"  t={t_ueq:.3f}, p={p_ueq:.6f}")

use_llm = df_expl.loc[df_expl["explanation_type"] == "LLM", "Usefulness"].dropna()
use_human = df_expl.loc[df_expl["explanation_type"] == "human", "Usefulness"].dropna()
t_use, p_use = ttest_ind(use_llm, use_human)
print(f"  Usefulness — LLM: M={use_llm.mean():.3f}, Human: M={use_human.mean():.3f}")
print(f"  t={t_use:.3f}, p={p_use:.6f}")

comp_llm = df_expl.loc[df_expl["explanation_type"] == "LLM", "Completeness"].dropna()
comp_human = df_expl.loc[df_expl["explanation_type"] == "human", "Completeness"].dropna()
t_comp, p_comp = ttest_ind(comp_llm, comp_human)
print(f"  Completeness — LLM: M={comp_llm.mean():.3f}, Human: M={comp_human.mean():.3f}")
print(f"  t={t_comp:.3f}, p={p_comp:.6f}")

try:
    df_manova = df_expl[["explanation_type", "UEQ", "Usefulness", "Completeness"]].dropna()
    manova = MANOVA.from_formula(
        "UEQ + Usefulness + Completeness ~ explanation_type", data=df_manova
    )
    print("\n  MANOVA result:")
    print(manova.mv_test())
except Exception as e:
    print(f"  MANOVA could not be computed: {e}")


# ---------- H4: Political congruence moderation--------------------

from math import sqrt

# Function to compute Cohen's d for independent samples
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

print("\n--- H4: Political congruence moderation ---")

# Build the LLM+Human subset (drop rows missing UEQ/Usefulness/Completeness/explanation_type)
df_llm_human = df_relevant.dropna(
    subset=['UEQ', 'Usefulness', 'Completeness', 'explanation_type']
).copy()

# Compute congruence using the ORIGINAL NOTEBOOK method:
# 1. Map article orientation to numeric: left=-1, center=0, right=1
# 2. Multiply by Participant_Political_Lean
# 3. Take sign of product
# 4. Congruent if sign > 0, else Incongruent (center articles become Incongruent)
orientation_map = {'left': -1, 'center': 0, 'right': 1}
df_llm_human['orientation_numeric'] = df_llm_human['article_orientation'].map(orientation_map)
df_llm_human['congruent'] = np.sign(
    df_llm_human['orientation_numeric'] * df_llm_human['Participant_Political_Lean']
)
df_llm_human['congruence'] = np.where(
    df_llm_human['congruent'] > 0, 'Congruent', 'Incongruent'
)
df_llm_human['congruence'] = df_llm_human['congruence'].astype('category')

# Also compute BiasChange_Abs if not already present
if 'BiasChange_Abs' not in df_llm_human.columns:
    df_llm_human['BiasChange_Abs'] = df_llm_human['BiasChange'].abs()


# HYPOTHESIS 4: BiasChange_Abs ~ explanation_type × congruence

print("\n\nHYPOTHESIS 4: BiasChange_Abs ~ explanation_type × congruence")
print("-" * 50)

model_h4 = smf.ols('BiasChange_Abs ~ C(explanation_type) * C(congruence)', data=df_llm_human).fit()
anova_h4 = sm.stats.anova_lm(model_h4, typ=2)
print(anova_h4)

print("\nMeans by condition:")
means_table = df_llm_human.groupby(['explanation_type', 'congruence'])['BiasChange_Abs'].mean().unstack()
print(means_table)

# Test simple effects of congruence within each explanation type
print("\nSimple effects of congruence within each explanation type:")
for exp_type in ['human', 'LLM']:
    subset = df_llm_human[df_llm_human['explanation_type'] == exp_type]
    congruent = subset[subset['congruence'] == 'Congruent']['BiasChange_Abs']
    incongruent = subset[subset['congruence'] == 'Incongruent']['BiasChange_Abs']
    n_cong, n_incong = len(congruent), len(incongruent)
    t_stat, p_val_two = ttest_ind(congruent, incongruent)
    p_val_one = p_val_two / 2 if np.mean(congruent) > np.mean(incongruent) else 1 - (p_val_two / 2)
    d_val = cohen_d(congruent, incongruent)
    print(f"\n{exp_type.upper()} explanations:")
    print(f"  n_congruent = {n_cong}, n_incongruent = {n_incong}")
    print(f"  t = {t_stat:.3f}, p_one-tailed = {p_val_one:.4f}, Cohen's d = {d_val:.3f}")
    print(f"  Means: Congruent = {np.mean(congruent):.3f}, Incongruent = {np.mean(incongruent):.3f}")


# HYPOTHESIS 4.1: Human vs LLM explanations in CONGRUENT condition

print("\n\nHYPOTHESIS 4.1: Human vs LLM explanations in CONGRUENT condition")
df_congruent = df_llm_human[df_llm_human['congruence'] == 'Congruent']
human_cong = df_congruent[df_congruent['explanation_type'] == 'human']['BiasChange_Abs']
llm_cong = df_congruent[df_congruent['explanation_type'] == 'LLM']['BiasChange_Abs']
t_stat, p_val_two = ttest_ind(human_cong, llm_cong)
p_val_one = p_val_two / 2 if np.mean(human_cong) > np.mean(llm_cong) else 1 - (p_val_two / 2)
d_val = cohen_d(human_cong, llm_cong)
print(f"n_human = {len(human_cong)}, n_LLM = {len(llm_cong)}")
print(f"t = {t_stat:.3f}, p_one-tailed = {p_val_one:.4f}, Cohen's d = {d_val:.3f}")
print(f"Means: Human = {np.mean(human_cong):.3f}, LLM = {np.mean(llm_cong):.3f}")

# Also do the incongruent condition
print("\nHuman vs LLM in INCONGRUENT condition:")
df_incongruent = df_llm_human[df_llm_human['congruence'] == 'Incongruent']
human_inc = df_incongruent[df_incongruent['explanation_type'] == 'human']['BiasChange_Abs']
llm_inc = df_incongruent[df_incongruent['explanation_type'] == 'LLM']['BiasChange_Abs']
t_stat, p_val_two = ttest_ind(human_inc, llm_inc)
p_val_one = p_val_two / 2 if np.mean(human_inc) > np.mean(llm_inc) else 1 - (p_val_two / 2)
d_val = cohen_d(human_inc, llm_inc)
print(f"n_human = {len(human_inc)}, n_LLM = {len(llm_inc)}")
print(f"t = {t_stat:.3f}, p_one-tailed = {p_val_one:.4f}, Cohen's d = {d_val:.3f}")
print(f"Means: Human = {np.mean(human_inc):.3f}, LLM = {np.mean(llm_inc):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 2: Combined bias change heatmap ─────────

def create_bias_change_summary(df_relevant, explanation_type):
    """
    Create a summary table for a specific explanation type showing:
    - Average bias change
    - Percentage of participants with increased bias perception
    - Percentage of participants with decreased bias perception
    """
    df_filtered = df_relevant[df_relevant['explanation_type'] == explanation_type].copy()
    summary_data = []

    for orientation in ['left', 'center', 'right']:
        for bias_level in ['low', 'medium', 'high']:
            group_data = df_filtered[
                (df_filtered['article_orientation'] == orientation) &
                (df_filtered['article_bias_level'] == bias_level)
                ]

            if len(group_data) > 0:
                avg_change = group_data['BiasChange'].mean()
                n_total = len(group_data)
                n_increased = len(group_data[group_data['BiasChange'] > 0])
                n_decreased = len(group_data[group_data['BiasChange'] < 0])
                pct_increased = (n_increased / n_total) * 100
                pct_decreased = (n_decreased / n_total) * 100

                summary_data.append({
                    'orientation': orientation,
                    'bias_level': bias_level,
                    'avg_change': avg_change,
                    'pct_increased': pct_increased,
                    'pct_decreased': pct_decreased,
                    'n_total': n_total
                })

    return pd.DataFrame(summary_data)

def plot_combined_bias_change(df_relevant):
    """
    Create a single figure with three subplots (one per condition) showing bias change
    """
    fig, axes = plt.subplots(1, 3, figsize=(36, 14), constrained_layout=True)

    # Calculate global vmax for consistent color scale
    all_summaries = []
    for condition in ['LLM', 'human', 'control']:
        all_summaries.append(create_bias_change_summary(df_relevant, condition))
    all_changes = pd.concat(all_summaries)['avg_change']
    vmax = max(abs(all_changes.min()), abs(all_changes.max()))

    for idx, (ax, condition) in enumerate(zip(axes, ['LLM', 'human', 'control'])):
        summary = create_bias_change_summary(df_relevant, condition)

        pivot_avg = summary.pivot(index='bias_level', columns='orientation', values='avg_change')
        pivot_avg = pivot_avg.reindex(['low', 'medium', 'high'])
        pivot_avg = pivot_avg[['left', 'center', 'right']]

        sns.heatmap(pivot_avg, annot=False, cmap='RdYlGn_r',
                    center=0, vmin=-vmax, vmax=vmax,
                    cbar=(idx == 2),
                    cbar_kws={'label': 'Avg Bias Change', 'shrink': 0.7, 'pad': 0.02} if idx == 2 else None,
                    linewidths=5, linecolor='black', ax=ax,
                    square=True)

        if idx == 2:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=32, width=2, length=8)
            cbar.set_label('Avg Bias Change', fontsize=36, fontweight='bold', rotation=270, labelpad=50)

        for i, bias_level in enumerate(['low', 'medium', 'high']):
            for j, orientation in enumerate(['left', 'center', 'right']):
                cell_data = summary[
                    (summary['bias_level'] == bias_level) &
                    (summary['orientation'] == orientation)
                    ]

                if not cell_data.empty:
                    avg_change = cell_data['avg_change'].values[0]
                    pct_increased = cell_data['pct_increased'].values[0]
                    pct_decreased = cell_data['pct_decreased'].values[0]

                    direction_arrow = '↑' if avg_change > 0 else '↓' if avg_change < 0 else '→'
                    text = f'{direction_arrow} {avg_change:.2f}\n↑{pct_increased:.0f}% ↓{pct_decreased:.0f}%'

                    color = 'black' if abs(avg_change) < vmax * 0.5 else 'white'
                    ax.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            fontsize=32, fontweight='bold', color=color, linespacing=1.3)

        ax.set_title(f'{condition.upper()}', fontsize=40, fontweight='bold', pad=25)

        if idx == 0:
            ax.set_ylabel('Bias Level', fontsize=36, fontweight='bold', labelpad=20)
            ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=34, rotation=0)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        ax.set_xlabel('Political Orientation', fontsize=36, fontweight='bold', labelpad=20)
        ax.set_xticklabels(['Left', 'Center', 'Right'], fontsize=34, rotation=0)

    return fig

fig = plot_combined_bias_change(df_relevant)
plt.savefig('bias_change_combined.png', dpi=300, bbox_inches='tight')
plt.show()


# ── Figure: Congruence effect (LLM vs Human) ─────────────────────────────────

if "congruence" in df_relevant.columns:
    df_cong_plot = df_relevant[
        df_relevant["explanation_type"].isin(["LLM", "human"])
        & df_relevant["congruence"].isin(["congruent", "incongruent", "Congruent", "Incongruent"])
    ].copy()
    df_cong_plot["congruence"] = df_cong_plot["congruence"].str.title()
    if len(df_cong_plot) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="explanation_type", y="biaschange_abs", hue="congruence",
            data=df_cong_plot, errorbar=('ci', 95), ax=ax
        )
        ax.set_title("Magnitude of Bias Change by Congruence and Explanation Type")
        ax.set_xlabel("Explanation Type")
        ax.set_ylabel("|Bias Change|")
        plt.tight_layout()
        plt.savefig("congruence_effect.pdf", bbox_inches="tight")
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 — QUALITATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

col1 = "What did you find most useful/helpful about the explanation?"
col2 = "What improvements would you suggest?"

# Drop rows where both columns are either NaN or empty string
df_qual = df[~((df[col1].isna() | (df[col1].str.strip() == "")) &
               (df[col2].isna() | (df[col2].str.strip() == "")))]

selected_df = df_qual[[col1, col2]]
selected_df.to_excel("qualitative_responses.xlsx", index=False)


def count_no_responses(series):
    empty_responses = series.isna().sum()
    no_suggestions = series.str.lower().str.contains('none|n/a|no|nothing', na=False).sum()
    return empty_responses + no_suggestions


stats_qual = {
    'Total_Responses': len(df_qual),
    'No_Answer_Most_Useful': count_no_responses(selected_df[col1]),
    'No_Answer_Improvements': count_no_responses(selected_df[col2]),
    'Average_Length_Most_Useful': selected_df[col1].str.len().mean(),
    'Average_Length_Improvements': selected_df[col2].str.len().mean()
}

print("\n" + "=" * 80)
print("QUALITATIVE ANALYSIS")
print("=" * 80)
for key, value in stats_qual.items():
    print(f"{key}: {value:.2f}")



