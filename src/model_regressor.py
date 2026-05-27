
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train, y_train):

    lr_model = LinearRegression()

    rfr_model = RandomForestRegressor(random_state=42)

    xgb_model = xgb.XGBRegressor(random_state=42)

    lr_model.fit(X_train, y_train)
    rfr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return lr_model, rfr_model, xgb_model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred


def plot_metrics(models, mae_scores, mse_scores, r2_scores):

    plt.figure(figsize=(6, 4))
    plt.bar(models, mae_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of MAE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.title('Comparison of R² Scores — Higher is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
