# wrapper for univariate feature selection

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
import pandas as pd
import matplotlib.pyplot as plt 

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on 
        different univariate feature selection models from
        scikit-learn.
        """
        # for a given problem type, there are only 
        # few valid scoring methods
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        
        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features)
            )
        else:
            raise Exception("Invalid type of feature")

    # same fit function
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    # same transform function
    def transform(self, X):
        return self.selection.transform(X)

    # same fit_transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)

# using this class
# ufs = UnivariateFeatureSelection(
#     n_features=0.1,
#     problem_type="regression",
#     scoring="f_regression"
# )
# ufs.fit(X, y)
# X_transformed = ufs.transform(X)

def main():

    train = pd.read_csv("../input/train_cleaned_1.csv")
    X = train.drop(['Cost', 'Customer Id'], axis=1)
    y = train['Cost']

    scorer_list = ["f_regression", "mutual_info_regression"]

    final_cols = []

    for scorer in scorer_list:

        ufs = UnivariateFeatureSelection(
        n_features = 15,
        problem_type='regression',
        scoring=scorer
        )

        fit = ufs.fit(X, y)
        X_transformed = ufs.transform(X)

        cols = [col for col in train.columns if col not in ("Customer Id", "Cost")]
        scores = list(fit.scores_)
        score_dict = {col:score for col, score in zip(cols, scores)}
        score_dict_sorted = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))

        for col in list(score_dict_sorted.keys())[:10]:
            if col not in final_cols:
                final_cols.append(col)

        plt.figure(figsize=(12,8))
        plt.barh(y=list(score_dict_sorted.keys())[:10], width=list(score_dict_sorted.values())[:10])
        plt.title(f"Feature Selection using {scorer} (top 10 columns)")
        plt.xlabel("score")
        plt.ylabel("column")
        plt.savefig(f"../feature_selection/{scorer}.png", bbox_inches ="tight", dpi=200)

    print(final_cols)

if __name__ == "__main__":
    main()

