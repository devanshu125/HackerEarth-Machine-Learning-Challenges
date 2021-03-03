import pandas as pd
import numpy as np

import os
import argparse
import joblib
import calc_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import itertools

from sklearn import metrics
from sklearn import preprocessing

import config
import model_dispatcher

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    """
    # this will create all 2-combinations of values
    # in the list
    # example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1,2), (1, 3),  (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold, arg_model_1, arg_model_2):

    df = pd.read_csv(config.TRAINING_FILE_1)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df = df.drop(drop_cols, axis=1)

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", 
                "Base Shipping Price", "Weight", "sculpture_shipping_price", 
                "Area", "price_per_wgt"
    ]

    # all columns are features except income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
    ]

    cat_cols = [
        f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
        and f not in num_cols
    ]

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialise model1 = lgbm
    model1 = model_dispatcher.models[arg_model_1]

    # fit model on training data
    model1.fit(x_train, np.log(df_train.Cost))

    # predict on validation data
    valid_preds_log_1 = model1.predict(x_valid)

    valid_preds_1 = np.exp(valid_preds_log_1)

    # initialise model2 = catboost
    model2 = model_dispatcher.models[arg_model_2]

    # fit model on training data
    model2.fit(x_train, np.log(df_train.Cost))

    # predict on validation data
    valid_preds_log_2 = model1.predict(x_valid)

    valid_preds_2 = np.exp(valid_preds_log_2)


    valid_preds_1 = np.absolute(valid_preds_1)

    valid_preds_2 = np.absolute(valid_preds_2)

    # save the model
    joblib.dump(
        model1,
        os.path.join(config.MODEL_OUTPUT, f"lgbm_cb/{arg_model_1}_{fold}.bin")
    )

    # save the model
    joblib.dump(
        model2,
        os.path.join(config.MODEL_OUTPUT, f"lgbm_cb/{arg_model_2}_{fold}.bin")
    )

    valid_preds = 0.4*valid_preds_1 + 0.6*valid_preds_2

    # pred_df = pd.DataFrame({"valid_preds": valid_preds, "df_valid_cost":df_valid.Cost.values})

    # print(pred_df.head())
    # print(pred_df.isna().sum())

    # calculate score
    score = calc_metric.calc_score(df_valid.Cost.values, valid_preds)
    

    # print rmse
    print(f"Fold = {fold}, Score = {score}")

    return score

if __name__ == "__main__":

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model1",
        type=str
    )

    parser.add_argument(
        "--model2",
        type=str
    )

    scores = []

    # read arguments from command line
    args = parser.parse_args()

    # run model specified by command line
    for fold_ in range(5):
        score = run(fold=fold_, arg_model_1=args.model1, arg_model_2=args.model2)
        scores.append(score)
    print(np.mean(scores))
    print()
    