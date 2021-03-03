import pandas as pd 
import numpy as np 

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import itertools
from sklearn.metrics import make_scorer
import catboost as cb

import calc_metric
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CLEANED_1)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df = df.drop(drop_cols, axis=1)

    X = df.drop(["Customer Id", "Cost"], axis=1).values
    y = df['Cost'].values

    regressor = cb.CatBoostRegressor(verbose=0)
    param_grid = {
        "n_estimators": np.arange(100, 800, 100),
        "max_depth": np.arange(1, 20, 1),
        "learning_rate": np.arange(0.01, 0.1, 0.01)
    }

    model = model_selection.RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_grid,
        n_iter=5,
        scoring=make_scorer(calc_metric.calc_score, greater_is_better=True),
        verbose=10,
        cv=5,
        n_jobs=4
    )
    model.fit(X, np.log(y))
    
    print()
    print("Best score: ", model.best_score_)
    print()
    print("Best params: ", model.best_params_)
    print()
    print("Best estimator: ", model.best_estimator_)
    print()

# n_estimators=700, max_depth=3, score=-0.715, total= 1.7min