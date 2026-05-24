# AlphaCare Auto-Insurance Risk Analytics & Predictive Modeling

[![CI Pipeline State](https://github.com/your-username/insurance-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/insurance-risk-analytics/actions/workflows/ci.yml)
[![Data Version Control Status](https://img.shields.io/badge/DVC-tracked-blue.svg)](https://dvc.org)
[![Python Version Target](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

An end-to-end data engineering, exploratory risk profiling, and predictive modeling system developed for **AlphaCare Insurance Solutions (ACIS)**. This project transforms raw historic transactional data in South Africa into auditable data pipelines, statistical insights, and advanced risk pricing segments.

---

## 📌 Business Case & Strategic Mandate

AlphaCare Insurance Solutions (ACIS) is navigating an aggressive expansion phase in the competitive South African car insurance market. Moving past static, legacy underwriting matrices, this project creates an evidence-driven framework leveraging 18 months of comprehensive portfolio data (February 2014 – August 2015). 

The primary business goal is to uncover **low-risk segments** (by geography, demographic profile, or vehicle attribute) where premiums can be reduced to aggressively attract high-quality clients while optimizing pricing models to maintain underwriting profitability.

---

## 📂 Repository Directory Layout

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions testing & linting pipeline
├── data/                        # Core data registry (Excluded from Git tracking)
│   ├── insurance_data.csv.dvc   # DVC pointer tracking raw dataset bytes
│   └── processed/
│       └── cleaned_insurance_data.csv # DVC pipeline output artifact target
├── notebooks/
│   ├── insurance_risk_analytics_eda.ipynb # Comprehensive EDA, anomaly audits & profiling
│   ├── 02_hypothesis_testing.ipynb        # A/B testing matrix & statistical significance logs
│   └── 03_modeling.ipynb                   # Supervised predictive learning & pricing framework
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py     # Production data cleaning pipeline & structural routing
│   ├── eda_utils.py             # Custom plotting utilities for heavily skewed distributions
│   ├── hypothesis_tests.py      # Automated statistical test runners (Chi2, t-tests)
│   └── modeling.py              # ML training, evaluation pipelines, and SHAP calculators
├── reports/
│   └── final_report.md          # Comprehensive executive & technical findings dossier
├── tests/                       # Unit tests suite verifying pipeline stability
├── .dvc/                        # Local Data Version Control runtime configurations
├── dvc.yaml                     # Reproducible pipeline step configurations
└── README.md                    # System documentation manual (This file)

```

---

## 🧮 Core Insurance Analytics Metric Definitions

Analytical benchmarks across notebooks and scripts leverage these actuarial formulas:

* **Loss Ratio:** Measures baseline portfolio profitability. Values $> 100\%$ signify an underwriting loss.

$$\text{Loss Ratio} = \frac{\text{TotalClaims}}{\text{TotalPremium}}$$


* **Net Profit Margin:** Captures absolute policy-level dollar contribution.

$$\text{Margin} = \text{TotalPremium} - \text{TotalClaims}$$


* **Claim Frequency:** The proportion of active policyholders filing an asset recovery.

$$\text{Claim Frequency} = \frac{\text{Count}(\text{TotalClaims} > 0)}{\text{Total Count of Active Policies}}$$



---

## 📊 Key Insights from Exploratory Data Analysis (EDA)

The foundational analysis in `notebooks/insurance_risk_analytics_eda.ipynb` evaluated **618,614 records across 55 discrete columns**. Key discoveries include:

### 1. Data Cleaning & Portfolio Audit (The 38% Leakage)

* **The Constraint:** System sweeps flagged that **~38% of rows contained precisely $0 Premium AND $0 Claims simultaneously**. These profiles represent immediate transactional churn—either cancellations, lapsed policies, or non-paying users.
* **The Strategy:** The script drops these rows via a combined index mask to protect statistical calculations from dilution:
```python
df_clean = df[~((df['TotalClaims'] == 0) & (df['TotalPremium'] == 0))]

```



### 2. Heavy-Tail Outlier Anomalies

* Risk fields (`TotalClaims`, `TotalPremium`, `SumInsured`) follow an extreme, highly right-skewed distribution.
* Whiskers on a baseline standard scale compress completely flat at $0 because the vast majority of policies do not file claims. However, high-severity commercial fleet exposure and catastrophic impacts push the extreme tail up to **R8,000,000 for Sum Insured** and **R400,000+ for Claim Severity**.

### 3. Feature Engineering: Dynamic Vehicle Age

* Calculated historical vehicle durations relative to transactional intervals, and split them into risk bins:
```python
df_clean['Vehicle_Age_Years'] = ((pd.Timestamp.now() - df_clean['VehicleIntroDate']).dt.days / 365.25).round(0)
bins = [0, 15, 20, 25, 30, np.inf]
labels = ["0-15 Years", "16-20 Years", "21-25 Years", "26-30 Years", "30+ Years"]
df_clean['VehicleAgeGroup'] = pd.cut(df_clean['Vehicle_Age_Years'], bins=bins, labels=labels)

```



### 4. Segment Risk Tiers

* **Province:** Gauteng (GP) records a much higher accident density and loss ratio than the Western Cape (WC).
* **Demographics:** Male drivers and single policyholders display elevated claim frequencies compared to female and married driver segments.

---

## 📦 Data Version Control (DVC) Pipeline Setup

To meet audit requirements in regulated financial domains, raw ingestion, intermediate processing, and artifact generation are fully version-tracked via DVC. This bypasses Git tracking for massive files, ensuring complete reproducibility.

### 1. System Synchronization

To pull tracking pointers from your configured storage repo, pull the dataset using:

```bash
# Initialize and link tracking indicators
dvc init
dvc pull

```

### 2. Reproducing or Forcing the Pipeline Execution

The data preprocessing steps are defined within `dvc.yaml`. If code dependencies or raw input source configurations change, run `dvc repro` to update the data pipeline:

```bash
# Re-run all dependent pipeline tracking stages
dvc repro

```

If you need to overwrite an existing workflow node while defining new absolute paths relative to the project root, register the stage with the `--force` flag:

```bash
dvc stage add --force -n preprocess \
  -d MachineLearningRating_v3.txt -d src/data_preprocessor.py \
  -o data/processed/cleaned_insurance_data.csv \
  python src/data_preprocessor.py

```

### 3. Absolute Code Rule for DVC Pipelines

Because DVC executes runtime stages **directly from the project root**, relative step-out routing (`../data/...`) will fail. All paths within `src/data_preprocessor.py` are absolute relative to the repository root:

```python
#  CORRECT PRODUCTION ROUTING
output_path = 'data/processed/cleaned_insurance_data.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cleaned_data.to_csv(output_path, index=False)

```

---

## 🔮 Modeling & Hypothesis Testing Roadmap (Next Steps)

### Task 3: Statistical Hypothesis Matrix

Observational trends from the EDA notebook will be validated or rejected across control and test variants using statistical tests ($p\text{-value} < 0.05$ threshold):

* **Regional Variation:** $\text{H}_0$: No risk variance across South African provinces (Chi-Square test on Frequency).
* **Granular Location:** $\text{H}_0$: No significant margin or profit difference across `PostalCode` indices (Two-Sample t-test).
* **Demographics:** $\text{H}_0$: Risk metrics are statistically equivalent between Male and Female drivers (Z-test of proportions).

### Task 4: Predictive Modeling Engine

The pipeline will train predictive risk structures across three core algorithms: **Linear Regression**, **Random Forest**, and **XGBoost Regressors**, optimized using an actuarial target formula:

$$\text{Risk Premium} = \big(P(\text{Claim Occurrence}) \times \text{Predicted Loss Severity}\big) + \text{Expense Loadings} + \text{Target Profit Margin}$$

* **Interpretability:** Model pricing decisions will be fully unpacked using **SHAP (SHapley Additive exPlanations)** to isolate and explain the top 10 pricing drivers for underwriters and regulatory auditors.

---

## 🛠️ Installation & Getting Started

```bash
# Clone and access the repository workspace
git clone [https://github.com/your-username/insurance-risk-analytics.git](https://github.com/your-username/insurance-risk-analytics.git)
cd insurance-risk-analytics

# Create isolated environment and download dependencies
python -m venv .venv
source .venv/bin/activate # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

# Pull matching data binary from tracking registry
dvc pull

# Test environment script functionality
python src/data_preprocessor.py

```

---

### Code Commitment Guidelines (Git Tracking Separation)

Always remember to keep large files out of Git. When saving pipeline adjustments, **never use `git add -f` on `.csv` or `.txt` objects**. Run your commits exactly like this:

```bash
git add dvc.yaml dvc.lock src/data_preprocessor.py notebooks/*.ipynb
git commit -m "feat: complete data processing stage execution and register dvc locks"
dvc push

```

---

```

```
