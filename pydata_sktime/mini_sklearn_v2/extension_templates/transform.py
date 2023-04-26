# -*- coding: utf-8 -*-
"""Extension template for transformers."""
from .base import BaseTransformer


# todo: change class name and write docstring
class ClassName(BaseTransformer):
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
    def _fit(self, X):
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
        # insert logic for estimator here
        # fitted parameters should be written to parameters ending in underscore

        # self must be returned at the end
        return self

    # todo: implement logic
    def _transform(self, X):
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
        # implement logic for transformation here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self
        Xt = X  # do something

        return Xt
