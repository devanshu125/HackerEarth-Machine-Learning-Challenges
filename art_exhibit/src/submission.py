import pandas as pd
import numpy as np
import itertools
import joblib
import os
from scipy.special import inv_boxcox

from sklearn import preprocessing

import config


def get_preds(test):

    test_preds = 0

    model_path = os.path.join(config.MODEL_OUTPUT, "lbl_cb_tuned/")

    for fold in range(5):

        model = joblib.load(os.path.join(model_path, f"cb_{fold}.bin"))

        temp_test_log = model.predict(test)

        temp_test = np.exp(temp_test_log)

        # print(f"fold: {fold}")

        # for tlog, t in zip(temp_test_log, temp_test):
        #     if tlog > 2.5:
        #         print(tlog, " | ", t)

        test_preds += temp_test / 5

    return test_preds

def main():

    # read test and sample_submission
    df_test = pd.read_csv(config.TEST_CLEANED_1)
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)

    drop_cols = [
        'scheduled_year', 'scheduled_weekofyear', 'scheduled_month', 
        'scheduled_dayofweek', 'scheduled_weekend', 'delivery_year', 
        'delivery_weekofyear', 'delivery_month', 'delivery_dayofweek', 
        'delivery_weekend', "City", "Code"
    ]

    df_test = df_test.drop(drop_cols, axis=1)

    # list of numerical columns
    num_cols = ["Artist Reputation", "Height", "Width", "Price Of Sculpture", 
                "Base Shipping Price", "Weight", "sculpture_shipping_price", 
                "Area", "price_per_wgt"
    ]

    # all columns are features except income and kfold columns
    features = [
        f for f in df_test.columns if f not in ("Cost", "Customer Id")
    ]

    print(df_test.head())

    # get test
    test = df_test[features].values

    final_preds = get_preds(test)

    final_preds = np.absolute(final_preds)

    sample_submission['Cost'] = final_preds

    print(sample_submission.head())

    print(sample_submission.isna().sum())

    # sample_submission = sample_submission.fillna(inv_boxcox(1 / 0.4, -0.398686))

    # print(sample_submission.isna().sum())

    sample_submission.to_csv("../submissions/lbl_cb_tuned_log_5.csv", index=False)

if __name__ == "__main__":
    main()