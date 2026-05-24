import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Set the baseline Seaborn theme
sns.set_theme(style="whitegrid", context="notebook")

# 2. Fine-tune Matplotlib to force grids on both major and minor ticks (crucial for log scales)
plt.rcParams['axes.grid'] = True          # Turn grids on by default
plt.rcParams['grid.alpha'] = 0.5          # Make grid lines subtle and semi-transparent
plt.rcParams['grid.linestyle'] = '--'     # Use dashed lines for the grid
plt.rcParams['grid.color'] = '#CBD5E1'    # Clean light-gray grid color (Tailwind Slate-300)
plt.rcParams['figure.dpi'] = 100          # Set a sharp, consistent rendering resolution

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#dddddd'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

PALETTE = {
    'primary': '#1d3557',   
    'secondary': '#457b9d',   
    'accent': '#e63946',      
    'light': '#f1faee',       
    'highlight': '#a8dadc'    
}
sns.set_palette([PALETTE['primary'], PALETTE['secondary'], PALETTE['highlight'], PALETTE['accent']])

def plot_summarization(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution of Annual Premiums
    sns.histplot(df['TotalPremium'], kde=True, color=PALETTE['primary'], ax=axes[0], bins=30)
    axes[0].set_title('Distribution of Total Premiums Paid', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[0].set_xlabel('Total Premium ($)', fontsize=12)
    axes[0].set_ylabel('Policyholder Count', fontsize=12)

    # Distribution of Claims Paid (Excluding 0 claims)
    sns.histplot(df['TotalClaims'], kde=True, color=PALETTE['accent'], ax=axes[1], bins=30)
    axes[1].set_title('Distribution of Paid Claim Amounts (Excluding Zero)', fontsize=14, fontweight='bold', color=PALETTE['accent'])
    axes[1].set_xlabel('Claim Amount ($)', fontsize=12)
    axes[1].set_ylabel('Claim Count', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_policy_distribution(df):
    conditions = [
        (df['TotalPremium'] == 0) & (df['TotalClaims'] == 0),
        (df['TotalPremium'] > 0) & (df['TotalClaims'] == 0),
        (df['TotalPremium'] > 0) & (df['TotalClaims'] > 0),
        (df['TotalPremium'] == 0) & (df['TotalClaims'] > 0)
    ]

    labels = [
        'Both Zero',
        'Premium Only',
        'Premium & Claims',
        'Claims Only'
    ]

    df['PolicyStatus'] = np.select(conditions, labels, default='Other')

    df['PolicyStatus'].value_counts().plot(
        kind='bar',
        figsize=(8,5)
    )

    print({((df['TotalPremium'] == 0) & (df['TotalClaims'] == 0)).sum()})

    plt.title('Policy Status Distribution')
    plt.ylabel('Policy Count')

    plt.show()

def plot_log_summarization(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution of Annual Premiums
    sns.histplot(df['TotalPremium'], kde=True, color=PALETTE['primary'], ax=axes[0], bins=30, log_scale=True)
    axes[0].set_title('Distribution of Total Premiums Paid', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[0].set_xlabel('Total Premium ($)', fontsize=12)
    axes[0].set_ylabel('Policyholder Count', fontsize=12)


    # Filter out NaNand  Infs for claims 
    clean_claims = df['TotalClaims'].dropna()
    clean_claims = clean_claims[np.isfinite(clean_claims)]
    sns.histplot(data = clean_claims, kde=True, color=PALETTE['accent'], ax=axes[1], bins=30, log_scale=True)
    axes[1].set_title('Distribution of Paid Claim Amounts (Excluding Zero)', fontsize=14, fontweight='bold', color=PALETTE['accent'])
    axes[1].set_xlabel('Claim Amount ($)', fontsize=12)
    axes[1].set_ylabel('Claim Count', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_province_vType_claimRate(df):
    # Province and Vehicle Type Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Claim Rate by Province
    prov_claims = df.groupby('Province')['TotalClaims'].mean().reset_index()
    prov_claims['TotalClaims'] *= 100  # Convert to %
    prov_claims = prov_claims.sort_values(by='TotalClaims', ascending=False)

    sns.barplot(x='TotalClaims', y='Province', data=prov_claims, palette="Blues_r", ax=axes[0])
    axes[0].set_title('Claim Rate (%) by Province', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[0].set_xlabel('Claim Rate (%)', fontsize=12)
    axes[0].set_ylabel('Province', fontsize=12)

    # Claim Rate by Vehicle Type
    veh_claims = df.groupby('VehicleType')['TotalClaims'].mean().reset_index()
    veh_claims['TotalClaims'] *= 100  # Convert to %
    veh_claims = veh_claims.sort_values(by='TotalClaims', ascending=False)

    sns.barplot(x='TotalClaims', y='VehicleType', data=veh_claims, palette="Oranges_r", ax=axes[1])
    axes[1].set_title('Claim Rate (%) by Vehicle Type', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[1].set_xlabel('Claim Rate (%)', fontsize=12)
    axes[1].set_ylabel('Vehicle Type', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_vType_AvePremium_claimRate(df):
    # Province and Vehicle Type Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Claim Rate by Province
    vtype_premiums = df.groupby('VehicleType')['TotalPremium'].mean().reset_index()
    vtype_premiums = vtype_premiums.sort_values(by='TotalPremium', ascending=False)

    sns.barplot(x='TotalPremium', y='VehicleType', data=vtype_premiums, palette="Blues_r", ax=axes[0])
    axes[0].set_title('Average Premium by Vehicle Type', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[0].set_xlabel('Average Premium', fontsize=12)
    axes[0].set_ylabel('Vehicle Type', fontsize=12)

    # Claim Rate by Vehicle Type
    veh_claims = df.groupby('VehicleType')['TotalClaims'].mean().reset_index()
    veh_claims['TotalClaims'] *= 100  # Convert to %
    veh_claims = veh_claims.sort_values(by='TotalClaims', ascending=False)

    sns.barplot(x='TotalClaims', y='VehicleType', data=veh_claims, palette="Oranges_r", ax=axes[1])
    axes[1].set_title('Claim Rate (%) by Vehicle Type', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    axes[1].set_xlabel('Claim Rate (%)', fontsize=12)
    axes[1].set_ylabel('Vehicle Type', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_log_premium_claim_province(df):
    plt.figure(figsize=(14,10))

    sns.scatterplot(data=df,x='TotalPremium',y='TotalClaims',hue='Province',alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')

    plt.title('Premium vs Claims by Province')

    plt.show()


def plot_corr_numericals(df, numerical_columns):
    corr_matrix = df[numerical_columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Customer & Claims Metrics', fontsize=14, fontweight='bold', color=PALETTE['primary'])
    plt.show()


def plot_numericals_frequency(df, numerical_columns):
    for col in numerical_columns:

        plt.figure(figsize=(8,4))

        sns.histplot(df[col].dropna(),bins=30,kde=True)

    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Explicitly ensure the grid lines are drawn cleanly
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()


def plot_categoricals_count(df, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(5, 5))

        df[col].value_counts().head(10).plot(kind='bar')

        plt.title(f'Top Categories in {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

        plt.xticks(rotation=45)
        plt.show()

def plot_claim_premium(df):
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x='TotalPremium',
        y='TotalClaims',
        alpha=0.5
    )

    plt.title('Total Premium vs Total Claims')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')

    plt.show()


def plot_log_claim_premium(df):
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x='TotalPremium',
        y='TotalClaims',
        alpha=0.4
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.title('Log-Scaled Premium vs Claims')

    plt.show()

    plt.figure(figsize=(10,6))



def plot_diagnostics(df_diag, column):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))

    # Claim Rate
    sns.barplot(data=df_diag,x=column,y='Claim_Rate',ax=axes[0, 0])
    axes[0, 0].set_title('Claim Rate')

    # Average Premium
    sns.barplot(data=df_diag,x=column,y='Average_Premium',ax=axes[0, 1])
    axes[0, 1].set_title('Average Premium')

    # Average Claim Amount
    sns.barplot(data=df_diag,x=column,y='Average_Claim_Amount',ax=axes[1, 0])
    axes[1, 0].set_title('Claim Severity')

    # Total Policyholders
    sns.barplot(data=df_diag,x=column,y='Total_Policyholders',ax=axes[1, 1])
    axes[1, 1].set_title('Total Policyholders')

    plt.tight_layout()
    plt.show()

def plot_diagnostics_row(df_diag, column):
    fig, axes = plt.subplots(2, 2, figsize=(10, 14))

    # Claim Rate
    sns.barplot(data=df_diag,x='Claim_Rate', y=column,ax=axes[0, 0])
    axes[0, 0].set_title('Claim Rate')

    # Average Premium
    sns.barplot(data=df_diag,x='Average_Premium',y=column,ax=axes[0, 1])
    axes[0, 1].set_title('Average Premium')

    # Average Claim Amount
    sns.barplot(data=df_diag,x='Average_Claim_Amount',y=column,ax=axes[1, 0])
    axes[1, 0].set_title('Claim Severity')

    # Total Policyholders
    sns.barplot(data=df_diag,x='Total_Policyholders',y=column,ax=axes[1, 1])
    axes[1, 1].set_title('Total Policyholders')

    plt.tight_layout()
    plt.show()

def plot_diagnostics_vehicleModel(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # 1. Top 15 Models by Claim Rate
    top_claim_rate = df.nlargest(15, 'Claim_Rate')
    sns.barplot(data=top_claim_rate,y='Model',x='Claim_Rate',ax=axes[0, 0])
    axes[0, 0].set_title('Claim Rate by Vehicle Model')

    # 2. Top 15 Models by Average Premium
    top_premium = df.nlargest(15, 'Average_Premium')
    sns.barplot(data=top_premium,y='Model',x='Average_Premium',ax=axes[0, 1])
    axes[0, 1].set_title('Average Premium by Vehicle Model')

    # 3. Top 15 Models by Average Claim Amount (Severity)
    top_severity = df.nlargest(15, 'Average_Claim_Amount')
    sns.barplot(data=top_severity,y='Model',x='Average_Claim_Amount',ax=axes[1, 0])
    axes[1, 0].set_title('Average Claim Amount by Vehicle Model')

    # 4. Top 15 Models by Total Exposure (Policyholder Volume)
    top_volume = df.nlargest(15, 'Total_Policyholders')
    sns.barplot(data=top_volume,y='Model',x='Total_Policyholders',ax=axes[1, 1])
    axes[1, 1].set_title('Total Policyholders by Vehicle Model')

    plt.tight_layout()
    plt.show()

def plot_lossRatio(df, column):
    plt.figure(figsize=(12,6))

    sns.barplot(data=df,x=column,y='LossRatio')

    plt.xticks(rotation=45)

    plt.title('Loss Ratio by {column}',fontsize=14,fontweight='bold')

    plt.ylabel('Loss Ratio')

    plt.show()


def plot_premium_claim_zipcode(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x='TotalPremium',
        y='TotalClaims',
        size='PolicyCount',
        alpha=0.7
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Average Premium vs Claims by PostalCode')

    plt.xlabel('Average Premium')
    plt.ylabel('Average Claims')

    plt.show()


def plot_log_premium_claim_severity(df):
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=df[df['TotalClaims'] > 0],
        x='TotalPremium',
        y='TotalClaims',
        alpha=0.5
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.title(
        'Log-Scaled Relationship Between Premiums and Claims',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')

    plt.show()


def plot_box_outliers(df, column):
    for col in column:
        plt.figure(figsize=(8, 6))

        sns.boxplot(x=df[col], color=PALETTE['highlight'])

        plt.set_yscale('log')  # Set y-axis to log scale for better visibility of outliers

        plt.title(f'Box Plot of {col}', fontsize=14, fontweight='bold', color=PALETTE['primary'])
        plt.xlabel(col, fontsize=12)

        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.show()

def plot_box_log_outliers(df, column):
    for col in column:
        plt.figure(figsize=(8, 6))

        sns.boxplot(x=np.log1p(df[col]), color=PALETTE['highlight'])

        plt.title(f'Box Plot of {col} (Log Scale)', fontsize=14, fontweight='bold', color=PALETTE['primary'])
        plt.xlabel(col, fontsize=12)

        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.show()

def plot_box_percentile_outliers(df):
    plt.figure(figsize=(12,5))

    sns.boxplot(
        x=np.log1p(
            df[df['TotalClaims'] > 0]['TotalClaims']
        )
    )

    plt.title(
        'Log-Scaled Claim Amount Distribution with Outliers',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('log(1 + TotalClaims)')

    plt.show()



 