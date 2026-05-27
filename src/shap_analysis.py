import pandas as pd
import matplotlib.pyplot as plt
import shap

def shap_tree_explainer(model, X_test):
    explainer = shap.TreeExplainer(model)

    # 2. Calculate SHAP values on your test set features
    # (Using a sample of 500-1000 rows is recommended if your test set is massive)
    shap_values = explainer(X_test)

    # 3. Generate the standard Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
    plt.title("Top 10 Most Influential Features - Claims Severity Model", fontsize=14)
    plt.tight_layout()
    plt.show()


def shap_linear_explainer(model, X_test):
    explainer = shap.LinearExplainer(model, X_test)

    # 2. Calculate SHAP values on your test set features
    shap_values = explainer(X_test)

    # 3. Generate the standard Summary Plot
    plt.figure(figsize=(10, 6))
    
    # Note: For some versions of SHAP, LinearExplainer outputs a custom explanation object. 
    # Passing shap_values directly or extracting its .values ensures compatibility.
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
    
    plt.title("Top 10 Most Influential Features - Claims Severity Model", fontsize=14)
    plt.tight_layout()
    plt.show()
