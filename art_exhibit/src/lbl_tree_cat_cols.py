import pandas as pd
import numpy as np

import os
import argparse
import joblib
import calc_metric
import itertools
from scipy.stats import boxcox
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

def run(fold, arg_model):

    df = pd.read_csv(config.TRAINING_FILE)

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", "Base Shipping Price"]

    # list of cat_cols
    cat_cols = [
        f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
        and f not in num_cols
    ]

    df = feature_engineering(df, cat_cols)

    # all columns are features except income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "Cost", "Customer Id")
    ]

    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()

            # fit label encoder on all data
            lbl.fit(df[col])

            # transform all data
            df.loc[:, col] = lbl.transform(df[col])
    
    # # initialize minmax scaler
    # scaler = preprocessing.MinMaxScaler()

    # # fit_transform scaler on num_cols
    # df[num_cols] = scaler.fit_transform(df[num_cols])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialise linear regression model
    model = model_dispatcher.models[arg_model]

    # fit model on training data
    model.fit(x_train, df_train.Cost.values)

    # predict on validation data
    valid_preds = model.predict(x_valid)

    valid_preds = np.absolute(valid_preds)

    # calculate score
    score = calc_metric.calc_score(df_valid.Cost.values, valid_preds)
    

    # print rmse
    print(f"Fold = {fold}, Score = {score}")
    
    # save the model
    # joblib.dump(
    #     model,
    #     os.path.join(config.MODEL_OUTPUT, f"lbl_rf/{arg_model}_{fold}.bin")
    # )

    return score

if __name__ == "__main__":

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str
    )

    scores = []

    # read arguments from command line
    args = parser.parse_args()

    # run model specified by command line
    for fold_ in range(5):
        score = run(fold=fold_, arg_model=args.model)
        scores.append(score)
    print(np.mean(scores))
    