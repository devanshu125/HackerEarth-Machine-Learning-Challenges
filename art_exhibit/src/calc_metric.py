from sklearn import metrics

def calc_score(y_true, y_pred):
    score = 100 * max(0, 1 - metrics.mean_squared_log_error(y_true, y_pred))

    return score
