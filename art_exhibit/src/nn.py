import tensorflow as tf
import pandas as pd
import joblib
import calc_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import os
import config
import numpy as np
from sklearn import preprocessing

def build_model(n_hidden=1, n_neurons=50, learning_rate=0.001, input_shape=[30]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):   
        model.add(tf.keras.layers.Dense(n_neurons, activation = "relu"))#, kernel_initializer="lecun_normal"))
    model.add(tf.keras.layers.Dense(1, activation = "linear"))
    
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.MSLE, optimizer=optimizer)
    return model

def run(fold):

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

    features_concat = [
        f for f in df.columns if f not in cat_cols
    ]

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialise linear regression model
    model = build_model(input_shape=len(x_train.T), learning_rate=0.1)

    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    # fit model on training data
    model.fit(x_train, np.log(df_train.Cost), epochs=1)

    # predict on validation data
    valid_preds_log = model.predict(x_valid)

    valid_preds = np.exp(valid_preds_log)

    valid_preds = np.absolute(valid_preds)

    print(valid_preds)

    # calculate score
    score = calc_metric.calc_score(df_valid.Cost.values, valid_preds)
    

    # print rmse
    print(f"Fold = {fold}, Score = {score}")
    
    # # save the model
    # joblib.dump(
    #     model,
    #     os.path.join(config.MODEL_OUTPUT, f"lbl_catboost/{arg_model}_{fold}.bin")
    # )

    return score

if __name__ == "__main__":
    scores = []

    # run model specified by command line
    for fold_ in range(5):
        score = run(fold=fold_)
        scores.append(score)
    print(np.mean(scores))