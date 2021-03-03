import copy
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
import joblib
import os
import calc_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import config
import model_dispatcher

def mean_target_encoding(data):

    # make a copy of the dataframe
    df = copy.deepcopy(data)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df = df.drop(drop_cols, axis=1)

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", "Base Shipping Price", "Weight"]

    # all columns are features except Response and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "Customer Id", "Cost")
        and f not in num_cols
    ]

    print(features)

    # a list to store 5 validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all feature columns i.e. categorical columns
        for column in features:
            # create a dict of category: mean target
            mapping_dict = dict(
                df_train.groupby(column)["Cost"].mean()
            )
            # column_enc is the new column we have with mean_encoding
            df_valid.loc[
                :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df

def run(df, fold):
    
    try:
        drop_cols = [
            'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
            'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
            'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
            'delivery_weekend', "City", "Code"
        ]

        df = df.drop(drop_cols, axis=1)
    except:
        None

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", "Base Shipping Price"]

    # note that folds are same as before
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [
        f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
    ]

    # scale training data
    x_train = df_train[features].values

    # scale validation data
    x_valid = df_valid[features].values

    # initialize lgbm model
    model = model_dispatcher.models['rf']

    # fit model on training data
    model.fit(x_train, boxcox(df_train.Cost, -0.398686))

    # predict on validation data
    valid_preds_log = model.predict(x_valid)

    valid_preds = inv_boxcox(valid_preds_log, -0.398686)

    # get score
    score = calc_metric.calc_score(df_valid.Cost.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, Score = {score}")

    # joblib.dump(
    #     model,
    #     os.path.join(config.MODEL_OUTPUT, f"tg_enc/lgbm_{fold}.bin")
    # )

    return score

if __name__ == "__main__":
    # read data
    df = pd.read_csv(config.TRAINING_FILE_1)

    # create mean target encoded categories
    df = mean_target_encoding(df)

    scores = []

    # run training and validation for 5 folds
    for fold_ in range(5):
        score = run(df, fold_)
        scores.append(score)

    print("Average = ", np.mean(scores))