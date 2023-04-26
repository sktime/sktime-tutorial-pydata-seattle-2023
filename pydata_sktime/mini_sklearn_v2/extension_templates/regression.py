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
    def _fit(self, X, y):
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
        # insert logic for estimator here
        # fitted parameters should be written to parameters ending in underscore

        # self must be returned at the end
        return self

    # todo: implement logic
    def _predict(self, X):
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
        # implement logic for prediction here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self

        y_pred = "placeholder"
        # returned object should be pd.DataFrame
        # same length as X, same columns as y in fit
        return y_pred
