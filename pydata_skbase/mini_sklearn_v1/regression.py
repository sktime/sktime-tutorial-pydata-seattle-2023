# -*- coding: utf-8 -*-
"""Example regressors."""
import numpy as np
import pandas as pd

from .base import BaseRegressor


class LinReg(BaseRegressor):
    """Linear (ridge) regression without intercept.

    Parameters
    ----------
    shrink : float, must be 0 or larger
        Tikhonov regularization parameter

    Attributes
    ----------
    beta_ : pd.DataFrame, present after fitting
        coefficients of the linear regressor fit
        rows = variables in X
        columns = variables in y
    """

    def __init__(self, shrink=0.0):
        self.shrink = shrink

        super(LinReg, self).__init__()

    def fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

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

        # input check X vs y
        if not len(X) == len(y):
            raise ValueError(
                f"X and y in fit of {self} must have same number of rows, "
                f"but X had {len(X)} rows, and y had {len(y)} rows"
            )

        beta = np.matmul(np.linalg.pinv(X.values), y.values)
        beta = pd.DataFrame(beta, index=X.columns, columns=y.columns)
        self.beta_ = beta

        # this should be the underscore tag
        # the skbase BaseEstimator handles is_fitted in dependency of this
        self._is_fitted = True

        return self

    def predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        if not (X.columns == self._X_columns).all():
            raise ValueError(
                "X in transform must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in transform found X.columns = {X.columns}"
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"error in estimator {self}, attempt to call predict, "
                "but fit has not been called yet"
            )

        beta = self.beta_
        return X.dot(beta)
