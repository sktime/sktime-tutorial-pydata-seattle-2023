# -*- coding: utf-8 -*-
"""Extension template for transformers."""
from .base import BaseTransformer


class ClassName(BaseTransformer):
    """Base class for supervised regressors."""

    def __init__(self, paramname, paramname2="paramname2default"):
        self.paramname = paramname
        self.paramname2 = "paramname2default"

        super(ClassName, self).__init__()

        # any parameter checks go here

    def fit(self, X):
        """Fit transformer to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

        Parameters
        ----------
        X : pandas DataFrame
            data to fit transformer to

        Returns
        -------
        self : reference to self
        """
        # store columns in self for later check with transform
        self._X_columns = X.columns

        # insert logic for estimator here
        # fitted parameters should be written to parameters ending in underscore

        # this should be the underscore tag
        # the skbase BaseEstimator handles is_fitted in dependency of this
        self._is_fitted = True

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
        if not (X.columns == self._X_columns).all():
            raise ValueError(
                "X in transform must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in transform found X.columns = {X.columns}"
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"error in estimator {self}, attempt to call transform, "
                "but fit has not been called yet"
            )

        # implement logic for transformation here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self
        Xt = X  # do something

        return Xt
