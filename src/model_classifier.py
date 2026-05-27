from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt



def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

import pandas as pd
import numpy as np

def optimize_premiums(X_test_full, classifier, regressor, expense_loading=50, profit_margin=0.15):
    """
    Optimizes insurance premiums using a two-stage machine learning framework.
    """
    # 1. Predict P(claim) using the Classifier
    # predict_proba returns [prob_0, prob_1]. We harvest prob_1 (probability of claim)
    p_claim = classifier.predict_proba(X_test_full)[:, 1]
    
    # 2. Predict the Severity using the Regressor
    # We pass the same features to see what the cost would be if they claim
    predicted_severity = regressor.predict(X_test_full)
    
    # Force negative predictions to 0 (just in case Linear Regression acts up)
    predicted_severity = np.clip(predicted_severity, a_min=0, a_max=None)
    
    # 3. Create a results summary DataFrame
    results = pd.DataFrame({
        'Probability_of_Claim': p_claim,
        'Predicted_Severity': predicted_severity
    }, index=X_test_full.index)
    
    # 4. Apply your formula
    # Pure Premium = P(claim) * Expected Severity
    results['Pure_Premium'] = results['Probability_of_Claim'] * results['Predicted_Severity']
    
    # Final Premium = Pure Premium + Fixed Expense + Profit Margin %
    results['Optimized_Premium'] = (results['Pure_Premium'] + expense_loading) * (1 + profit_margin)
    
    return results

# Example Usage:
# optimized_df = optimize_premiums(X_test, rfc_model, rfr_model)
# print(optimized_df[['Probability_of_Claim', 'Predicted_Severity', 'Optimized_Premium']].head())


def train_models(X_train, y_train):

    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Random Forest Classifier
    rfc_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )

    # Train models
    lr_model.fit(X_train, y_train)

    rfc_model.fit(X_train, y_train)

    xgb_model.fit(X_train, y_train)

    return lr_model, rfc_model, xgb_model


def evaluate_model(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return accuracy, precision, recall, f1, y_pred


def plot_metrics(
    models,
    accuracy_scores,
    precision_scores,
    recall_scores,
    f1_scores
):

    # Accuracy
    plt.figure(figsize=(6, 4))

    plt.bar(models, accuracy_scores, color='skyblue')

    plt.xlabel('Models')

    plt.ylabel('Accuracy')

    plt.title('Comparison of Accuracy Scores — Higher is Better')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    # Precision
    plt.figure(figsize=(6, 4))

    plt.bar(models, precision_scores, color='lightgreen')

    plt.xlabel('Models')

    plt.ylabel('Precision')

    plt.title('Comparison of Precision Scores — Higher is Better')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    # Recall
    plt.figure(figsize=(6, 4))

    plt.bar(models, recall_scores, color='salmon')

    plt.xlabel('Models')

    plt.ylabel('Recall')

    plt.title('Comparison of Recall Scores — Higher is Better')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    # F1 Score
    plt.figure(figsize=(6, 4))

    plt.bar(models, f1_scores, color='orange')

    plt.xlabel('Models')

    plt.ylabel('F1 Score')

    plt.title('Comparison of F1 Scores — Higher is Better')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(model, X_test, y_test):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap='Blues')

    plt.title('Confusion Matrix')

    plt.show()


def optimize_premiums(X_test_full, classifier, regressor, expense_loading=50, profit_margin=0.15):
    """
    Optimizes insurance premiums using a two-stage machine learning framework.
    """
    # 1. Predict P(claim) using the Classifier
    # predict_proba returns [prob_0, prob_1]. We harvest prob_1 (probability of claim)
    p_claim = classifier.predict_proba(X_test_full)[:, 1]
    
    # 2. Predict the Severity using the Regressor
    # We pass the same features to see what the cost would be if they claim
    predicted_severity = regressor.predict(X_test_full)
    
    # Force negative predictions to 0 (just in case Linear Regression acts up)
    predicted_severity = np.clip(predicted_severity, a_min=0, a_max=None)
    
    # 3. Create a results summary DataFrame
    results = pd.DataFrame({
        'Probability_of_Claim': p_claim,
        'Predicted_Severity': predicted_severity
    }, index=X_test_full.index)
    
    # 4. Apply your formula
    # Pure Premium = P(claim) * Expected Severity
    results['Pure_Premium'] = results['Probability_of_Claim'] * results['Predicted_Severity']
    
    # Final Premium = Pure Premium + Fixed Expense + Profit Margin %
    results['Optimized_Premium'] = (results['Pure_Premium'] + expense_loading) * (1 + profit_margin)
    
    return results

# Example Usage:
# optimized_df = optimize_premiums(X_test, rfc_model, rfr_model)
# print(optimized_df[['Probability_of_Claim', 'Predicted_Severity', 'Optimized_Premium']].head())