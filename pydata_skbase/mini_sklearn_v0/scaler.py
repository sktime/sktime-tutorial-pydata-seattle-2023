# -*- coding: utf-8 -*-
"""Scaler functions."""


def scale_data(X, strategy="minmax"):
    """Scale data by minmax or standard normalization strategy.

    Parameters
    ----------
    X : pandas DataFrame, must have same columns as X in `fit`
        data to transform
    strategy : string, one of "minmax" or "std"
        "std" = each column of X is subtracted mean, divided standard deviation
        "minmax" = each column is linearly scaled to minimum = 0, maximum = 1

    Returns
    -------
    Xt : pandas DataFrame, same length as `X`
        transformed version of `X`
    """
    if strategy not in ["std", "minmax"]:
        raise ValueError(
            "in Scaler, strategy must be one of the strings "
            f'"std" or "minmax", but found {strategy}'
        )

    if strategy == "minmax":
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_span = X_max - X_min

        Xt = (X - X_min) / X_span

    elif strategy == "std":
        X_std = X.std(axis=0)
        X_mean = X.mean(axis=0)

        Xt = (X - X_mean) / X_std

    return Xt
