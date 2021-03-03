import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import itertools

from skopt import space
from functools import partial
from skopt import gp_minimize
import catboost as cb
import calc_metric

import config

def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    # model = ensemble.RandomForestRegressor(**params)
    model = cb.CatBoostRegressor(**params, verbose=0)

    scores = []

    for fold in range(5):

        # all columns are features except income and kfold columns
        features = [
            f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
        ]

        # get training data using folds
        df_train = df[df.kfold != fold].reset_index(drop=True)

        # get validation data using folds
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # get training data
        x_train = df_train[features].values

        # get validation data
        x_valid = df_valid[features].values

        # fit model on training data
        model.fit(x_train, np.log(df_train.Cost).values)

        # predict on validation data
        valid_preds_log = model.predict(x_valid)

        valid_preds = np.exp(valid_preds_log)

        valid_preds = np.absolute(valid_preds)

        # calculate metric
        score = calc_metric.calc_score(df_valid.Cost.values, valid_preds)
        
        # append rmse in list
        scores.append(score)

    return -1 * np.mean(scores)

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE_1)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df = df.drop(drop_cols, axis=1)

    X = df.drop(["Cost"], axis=1).values
    y = df.Cost.values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 800, name="n_estimators"),
        space.Real(0.01, 0.1, name="learning_rate")
    ]

    param_names = [
        "max_depth",
        "n_estimators",
        "learning_rate"
    ]

    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    print(dict(zip(param_names, result.x)))
