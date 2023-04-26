# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
from .base import BaseTransformer


class Scaler(BaseTransformer):
    """Base class for supervised regressors."""

    def __init__(self, strategy="std"):
        self.strategy = strategy

        super(Scaler, self).__init__()

        # this is fine here because of how set_params works!
        if strategy not in ["std", "minmax"]:
            raise ValueError(
                "in Scaler, strategy must be one of the strings "
                f'"std" or "minmax", but found {strategy}'
            )

    def fit(self, X):
        """Fit transformer to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            data to fit transformer to

        Returns
        -------
        self : reference to self
        """
        # min, max, max-min as fitted params, for minmax
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_span = X_max - X_min

        # mean, std as fitted params, for std
        X_std = X.std(axis=0)
        X_mean = X.mean(axis=0)

        self.X_min_ = X_min
        self.X_max_ = X_max
        self.X_span_ = X_span
        self.X_std_ = X_std
        self.X_mean_ = X_mean

        return self

    def transform(self, X):
        """Transform data with fitted transformer.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to transform

        Returns
        -------
        Xt : pandas DataFrame, same length as `X`
            transformed version of `X`
        """
        strategy = self.strategy

        if strategy == "minmax":
            X_min = self.X_min_
            X_span = self.X_span_

            Xt = (X - X_min) / X_span

        elif strategy == "std":
            X_mean = self.X_mean_
            X_std = self.X_std_

            Xt = (X - X_mean) / X_std

        return Xt
