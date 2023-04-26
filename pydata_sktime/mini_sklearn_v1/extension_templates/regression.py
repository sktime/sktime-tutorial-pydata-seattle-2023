# -*- coding: utf-8 -*-
"""Extension template for regressors."""
from .base import BaseRegressor


# todo: change class name and write docstring
class ClassName(BaseRegressor):
    """Base class for supervised regressors."""

    # todo: fill init
    # params should be written to self and never changed
    # super call must not be removed, change class name
    # parameter checks can go after super call
    def __init__(self, paramname, paramname2="paramname2default"):
        self.paramname = paramname
        self.paramname2 = "paramname2default"

        super(ClassName, self).__init__()

        # any parameter checks go here

    # todo: implement logic
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

        # insert logic for estimator here
        # fitted parameters should be written to parameters ending in underscore

        # this should be the underscore tag
        # the skbase BaseEstimator handles is_fitted in dependency of this
        self._is_fitted = True

        # self must be returned at the end
        return self

    # todo: implement logic
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

        # implement logic for prediction here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self

        y_pred = "placeholder"
        # returned object should be pd.DataFrame
        # same length as X, same columns as y in fit
        return y_pred
