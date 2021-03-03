from sklearn import linear_model
import xgboost as xgb
from sklearn import ensemble
import lightgbm as lgb
from sklearn import tree
import catboost as cb
import lightgbm as lgb

models = {
    "linear_regression": linear_model.LinearRegression(),
    "ridge": linear_model.Ridge(alpha=0.5),
    "xgboost": xgb.XGBRegressor(
        n_jobs=-1,
        max_depth=14,
        n_estimators=100
    ),
    "rf": ensemble.RandomForestRegressor(
        max_depth=19,
        n_estimators=500,
        min_samples_leaf=3
    ),
    "lasso": linear_model.Lasso(alpha=0.5),
    "dt": tree.DecisionTreeRegressor(max_depth=7),
    "cb": cb.CatBoostRegressor(
        verbose=0,
        learning_rate=0.0936768374028697,
        max_depth=6,
        n_estimators=789
    ),
    "lgbm": lgb.LGBMRegressor()
}