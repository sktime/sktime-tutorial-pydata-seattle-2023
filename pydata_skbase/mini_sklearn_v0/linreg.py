# -*- coding: utf-8 -*-
"""Linear regression functions."""

import numpy as np
import pandas as pd


def fit_linreg(X, y):
    """Fit regressor to training data.

    Parameters
    ----------
    X : pandas DataFrame
        feature instances to fit regressor to
    y : pandas DataFrame, must be same length as X
        labels to fit regressor to

    Returns
    -------
    beta : pd.DataFrame
        coefficients of the linear regressor fit
        rows = variables in X
        columns = variables in y
    """
    beta = np.matmul(np.linalg.pinv(X.values), y.values)
    beta = pd.DataFrame(beta, index=X.columns, columns=y.columns)

    return beta


def predict_linreg(X, beta):
    """Predict labels for data from features.

    Parameters
    ----------
    X : pandas DataFrame, must have same columns as X in `fit`
        data to predict labels for
    beta : pd.DataFrame, as output by fit_linreg
        row index of beta must match column index of X

    Returns
    -------
    y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
        labels predicted for `X`
    """
    return X.dot(beta)
