import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import numpy as np
import xgboost as xgb

data_train = pd.read_csv("train.csv")

df_train, df_test = train_test_split(data_train, test_size=0.1, random_state=42)

X_train = df_train.drop(columns=["BeatsPerMinute", "id"])
y_train = df_train["BeatsPerMinute"]

X_test = df_test.drop(columns=["BeatsPerMinute", "id"])
y_test = df_test["BeatsPerMinute"]

xgb_model = xgb.XGBRegressor(
    tree_method="gpu_hist", 
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=3,              
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Résultats
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score (RMSE CV) :", -grid_search.best_score_)

best_model = grid_search.best_estimator_
joblib.dump(best_model, "best_xgboost_gpu_model.pkl")
print("Modèle GPU sauvegardé dans best_xgboost_gpu_model.pkl")