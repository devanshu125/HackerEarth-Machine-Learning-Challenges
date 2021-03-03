import os
import gc
import joblib
import pandas as pd 
import numpy as np 
import config
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K 
from tensorflow.keras import utils
import calc_metric

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model for entity embedding
    """
    # init list of inputs for embedding
    inputs = []

    # init list of outputs for embedding
    outputs = []

    # loop over all categorical columns
    for c in catcols:
        # find the number of unique values in column
        num_unique_values = int(data[c].nunique())
        # simple dimension of embedding calculator
        # min size is half of the number of unique values
        # max size is 50. max size depends on the number of unique
        # categories too. 50 is quite sufficient most of the times
        # but if you have millions of unique values, you might
        # need a larger dimension
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in the input
        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c
        )(inp)

        # 1-d spatial dropout is the standard for embedding layers
        # you can use it in NLP tasks too
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding
        # this becomes our output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)

    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layer
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with dropout 
    # start with 1 or two layers only
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax and treating it as a two class problem
    # you can a;ps use sigmoid, then you need to use only one output class
    y = layers.Dense(1, activation="linear")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    opt = SGD(lr=0.01, momentum=0.9)

    # compile the model
    # we use adam and binary cross entropy
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)

    return model

def run(fold):

    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE_1)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df = df.drop(drop_cols, axis=1)

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", "Base Shipping Price", "Weight", "sculpture_shipping_price", "Area"]

    features = [
        f for f in df.columns if f not in ("Customer Id", "Cost", "kfold")
        and f not in num_cols
    ]

    # for col in features:
    #     df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # # encode all features with a label encoder individually
    # # in a live setting you need to save all label encoders
    # for feat in features:
    #     lbl_enc = preprocessing.LabelEncoder()
    #     df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)

    # our features are lists of lists
    x_train = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    x_valid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    # fetch target columns
    ytrain = df_train.Cost.values
    yvalid = df_valid.Cost.values

    # convert target columns to categories
    # this is just binarization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(
        x_train, 
        ytrain_cat,
        validation_data=(x_valid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )

    # generate validation predictions
    valid_preds = model.predict(x_valid)

    # print roc auc score
    print(calc_metric.calc_score(yvalid, valid_preds))

    # clear session to free up some GPU memory
    K.clear_session()

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)