import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, poisson, norm

def chi_squared_test(contingency_table, H,  H0, H1):
    chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    print('===============================')
    print(f'{H} CHI-SQUARED TEST')
    print('===============================')
    print(f'H₀: {H0}')
    print(f'H₁: {H1}')
    print(f'---')
    print(f'Chi² statistic:    {chi2_stat:.4f}')
    print(f'Degrees of freedom: {dof}')
    print(f'P-value:            {p_value_chi2:.6f}')
    print(f'---')
    decision = (
        'Reject H₀'
        if p_value_chi2 < 0.05
        else 'Fail to Reject H₀'
    )

    print()
    print('=== Expected Frequencies (under H₀) ===')
    expected_df = pd.DataFrame(expected, 
                            index=contingency_table.index, 
                            columns=contingency_table.columns)
    print(expected_df.round(1))
    return {
        'Hypothesis': H,
        'Test': 'Chi-Squared',
        'Chi2 Statistic': chi2_stat,
        'Degrees of Freedom': dof,
        'P-Value': p_value_chi2,
        'Decision': decision
    }



def t_test(group_a, group_b, H, H0, H1):
    t_stat, p_value = ttest_ind(group_a,group_b,equal_var=False)

    alpha = 0.05
    print('===============================')
    print(f'  {H} T-TEST')
    print('===============================')
    print(f'H₀: {H0}')
    print(f'H₁: {H1}')
    print(f'---')
    print(f'T-statistic: {t_stat:.4f}')
    print(f'P-value:     {p_value:.6f}')
    print(f'Alpha:       {alpha}')
    print(f'---')
    decision = (
        'Reject H₀'
        if p_value < 0.05
        else 'Fail to Reject H₀'
    )
    return {
        'Hypothesis': H,
        'Test': 'T-Test',
        'T-statistic': t_stat,
        'P-Value': p_value,
        'Decision': decision
    }
