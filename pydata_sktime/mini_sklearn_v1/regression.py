# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
import numpy as np
import pandas as pd

from .base import BaseRegressor


class LinReg(BaseRegressor):
    """Base class for supervised regressors."""

    def __init__(self, shrink=0.0):
        self.shrink = shrink

        super(LinReg, self).__init__()

    def fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        # store columns in self for later check with transform
        self._X_columns = X.columns

        beta = np.matmul(np.linalg.pinv(X.values), y.values)
        beta = pd.DataFrame(beta, index=X.columns, columns=y.columns)
        self.beta_ = beta

        return self

    def predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        if not (X.columns == self._X_columns).all():
            raise ValueError(
                "X in transform must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in transform found X.columns = {X.columns}"
            )

        beta = self.beta_
        return X.dot(beta)
