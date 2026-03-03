# Plausibility Machines Repository

This repository contains all analysis code, survey data, and supplementary materials for the paper:

> **"Plausibility Machines: How LLM-Generated Explanations Shape Media Bias Perception"**


---

## Overview

We conduct a preregistered randomized controlled trial (n = 504) to measure whether LLM-generated bias explanations can change audiences' perception of media bias, and how these compare to human-written explanations and no-explanation controls.

---

## Repository Structure

---

## Analysis

### Main Study (`Analysis/main_analysis.py`)

Reproduces all quantitative and qualitative results reported in the paper.

### Pre-Study (`Analysis/main_analysis_prestudy.py`)

Contains the analysis for the pre-study (n = 50), which informed the selection of the LLM prompting strategy used in the main experiment.

### Text Characteristic Analysis (`Analysis/Text Characteristic Analysis/`)

Analyzes readability, sentiment, complexity, and moral-emotional language of human-written and LLM-generated explanations.

---

## Supplementary Materials

| File | Description |
|------|-------------|
| `News Articles and Treatments.pdf` | All 9 news articles and corresponding explanations (human, LLM, control) |
| `Study Questions.pdf` | Complete survey instrument, including all question items and scales |
| `SurveyScreenshots.pdf` | Visual documentation of the survey as presented to participants |

---

## Data

| File | Description |
|------|-------------|
| `Prestudy Results.csv` | Responses from the pre-study (n = 50) |
| `Study Results.csv` | Responses from the main study (n = 504) |

Both files use `;` as the delimiter.

---

## Requirements

The main analysis script requires the following Python packages:

- `pandas`
- `numpy`
- `statsmodels`
- `scipy`
- `pingouin`
- `matplotlib`
- `seaborn`

Install all dependencies with:

```bash
pip install pandas numpy statsmodels scipy pingouin matplotlib seaborn

