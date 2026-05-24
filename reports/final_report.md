# Insurance Risk Analytics Final Report

## Executive Summary
This report summarizes the key findings from the `insurance_risk_analytics_eda.ipynb` exploratory analysis. The dataset shows that claims exceed collected premiums, and the portfolio is operating with a negative margin.

## Key Findings
- Total claims paid are greater than total premiums collected, resulting in a loss position.
- Approximately 38% of records have both `TotalPremium` and `TotalClaims` equal to zero, indicating inactive or cancelled policies.
- The premium structure appears to be relatively flat for many policies, while claim severity is highly skewed and heavy-tailed.
- Outlier analysis shows a small tail of extreme claims driving a disproportionate share of losses, especially in high-value vehicles and certain provinces.
- High-risk segments include heavy commercial vehicles, certain vehicle makes/models, and specific provinces.
- Gauteng, KwaZulu-Natal, and Western Cape show elevated loss ratios and claim frequency.
- Vehicle attributes such as `cubiccapacity`, `kilowatts`, `Cylinders`, and `mmcode` have strong correlations and are good candidates for risk-based pricing.
- Vehicle age is informative: mid-aged cars are most common, new vehicles pay higher premiums, and older vehicles can still have substantial claim risk.

## Plot-Based Insights
- `plot_summarization`: highlights distribution skew and the need for data transformation.
- `plot_policy_distribution`: identifies four major policy groups and exposes suspicious zero-premium claim cases.
- `plot_box_outliers`: shows that raw premium and claims are so heavy-tailed that standard IQR box plot interpretation is limited.
- `plot_box_log_outliers`: confirms that log scaling is essential to compare and flag outliers more robustly while preserving the heavy right tail.
- `plot_box_percentile_outliers`: isolates the top 1% of claims, making the highest-severity policies visible for operational review.
- `plot_log_summarization` / `plot_log_claim_premium` / `plot_log_premium_claim_severity`: confirm that log transformation makes premium and claim relationships clearer.
- `plot_corr_numericals`: shows clear clustering among vehicle-related numeric features.
- `plot_diagnostics` and related diagnostics plots: reveal that Gender, MaritalStatus, Province, VehicleType, Model, make, and VehicleAgeGroup all influence claims and premiums.

## Recommendations
- Remove or separately model inactive zero-premium/zero-claim records to reduce noise in underwriting analytics.
- Implement risk-based pricing using vehicle type, province, vehicle make/model, and age.
- Investigate zero-premium policies with active claims for underwriting or fraud risk.
- Prioritize monitoring of heavy commercial vehicles and high-loss provinces.
- Use log-scaled modeling for claim and premium features to handle skew and extreme values.
- Use percentile-based and log-based outlier rules to flag the highest-severity claims for special review, reserving, and underwriting adjustment.

## Conclusion
The EDA shows significant underwriting risk and evidence that the current pricing approach is not aligned with actual claim severity. The strongest next step is to build a risk-based pricing model using the identified vehicle, demographic, and geographic drivers.
