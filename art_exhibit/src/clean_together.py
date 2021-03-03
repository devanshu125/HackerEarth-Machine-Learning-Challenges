from sklearn.impute import KNNImputer
import pandas as pd
from sklearn import preprocessing
from scipy.stats import boxcox
import numpy as np
import itertools
import argparse

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

def clean_together(dataset, n_neighbors=9):

    # num_cols
    num_cols = ['Artist Reputation', 'Height', 'Width', 'Weight']

    dataset['Weight'] = np.log(dataset['Weight'])

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    knn_imputer.fit(dataset[num_cols])

    dataset[num_cols] = knn_imputer.transform(dataset[num_cols])

    dataset['sculpture_shipping_price'] = dataset['Price Of Sculpture'] / dataset['Base Shipping Price']
    dataset['price_per_wgt'] = dataset['Base Shipping Price'] / dataset['Weight']
    dataset['Area'] = dataset['Height'] * dataset['Width']

    # cat_cols
    cat_cols = ['Transport', "Material"]

    for col in cat_cols:
        dataset.loc[:, col] = dataset[col].fillna("NONE")

    dataset = feature_engineering(dataset, cat_cols)

    dataset['Remote Location'] = dataset['Remote Location'].fillna(pd.Series(np.random.choice(['Yes', 'No'],  p=[0.198115, 0.801885], size=len(dataset))))

    # extracting from address

    cities = []

    for address in dataset['Customer Location']:
        if ',' in address:
            city = address.split(',')[0]
        else:
            city = address.split(' ')[0]
            
        cities.append(city)
        
    dataset['City'] = cities

    states = []

    for address in dataset['Customer Location']:
        if ',' in address:
            state = (address.split(',')[1]).split(' ')[1]
        else:
            state = address.split(' ')[1]
            
        states.append(state)
        
    dataset['State'] = states

    codes = []

    for address in dataset['Customer Location']:
        if ',' in address:
            code = (address.split(',')[1]).split(' ')[2]
        else:
            code = address.split(' ')[2]
            
        codes.append(code)
        
    dataset['Code'] = codes

    dataset.drop('Customer Location', axis=1, inplace=True)

    datetime_cols = ['Scheduled Date', 'Delivery Date']

    for col in datetime_cols:
        dataset.loc[:, col] = pd.to_datetime(dataset[col], format='%m/%d/%y')

    for col in datetime_cols:
        string = (col.split(' ')[0]).lower()
        dataset.loc[:, string + '_year'] = dataset[col].dt.year
        dataset.loc[:, string + '_weekofyear'] = dataset[col].dt.weekofyear
        dataset.loc[:, string + '_month'] = dataset[col].dt.month
        dataset.loc[:, string + '_dayofweek'] = dataset[col].dt.dayofweek
        dataset.loc[:, string + '_weekend'] = (dataset[col].dt.weekday >= 5).astype(int)
    
    dataset.drop(datetime_cols, axis=1, inplace=True)

    dataset.drop(['Artist Name'], axis=1, inplace=True)

    # all columns are features except income and kfold columns
    features = [
        f for f in dataset.columns if f not in ("kfold", "Cost", "Customer Id")
    ]

    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()

            # fit label encoder on all dataset
            lbl.fit(dataset[col])

            # transform all dataset
            dataset.loc[:, col] = lbl.transform(dataset[col])

    return dataset

if __name__ == "__main__":

    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    test['Cost'] = -1

    dataset = pd.concat([train, test]).reset_index(drop=True)

    # # initialize ArgumentParser class of argparse
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--n_neighbors",
    #     type=int
    # )

    # # read arguments from command line
    # args = parser.parse_args()

    dataset_cleaned = clean_together(dataset)

    train = dataset_cleaned[dataset_cleaned.Cost != -1].reset_index(drop=True)
    test = dataset_cleaned[dataset_cleaned.Cost == -1].reset_index(drop=True)

    test = test.drop(['Cost'], axis=1)

    train['Cost'] = train['Cost'].abs()

    train.to_csv("../input/train_cleaned_1.csv", index=False)

    test.to_csv("../input/test_cleaned_1.csv", index=False)

