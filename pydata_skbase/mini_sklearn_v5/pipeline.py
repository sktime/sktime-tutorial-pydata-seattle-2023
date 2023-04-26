# -*- coding: utf-8 -*-
"""Regressor pipeline."""
from .base import BaseMetaEstimator, BaseRegressor, BaseTransformer


class RegressorPipeline(BaseRegressor, BaseMetaEstimator):
    """Scale data by minmax or standard normalization strategy.

    Parameters
    ----------
    steps :

    Attributes
    ----------
    steps_ : clone of steps, coerced to (name, estimator) tuples
    """

    _tags = {
        "regressor_type": "compositor",
        "named_object_parameters": "steps",
        "fitted_named_object_parameters": "steps_",
    }

    def __init__(self, steps):
        self.steps = steps

        super(RegressorPipeline, self).__init__()

        # check and coerce to name/estimator tuples in separate parameter
        # the hyperparameters (e.g., steps) should never be mutated!
        # in steps_, the estimators are clones, so no side effect mutation takes place
        self.steps_ = self._check_objects(
            self.steps,
            cls_type=(BaseRegressor, BaseTransformer),
        )

        # determine the unique regressor in the pipeline and save index to self._reg_ix
        # also check that there ix exactly one regressor
        names, ests = self._get_names_and_objects(self.steps_)
        n = len(ests)
        reg_ix = [i for i in range(n) if isinstance(ests[i], BaseRegressor)]
        reg_ix_names = [names[i] for i in reg_ix]
        if len(reg_ix) != 1:
            raise ValueError(
                "in RegressorPipeline, exactly one of the estimators must be"
                f"a regressor, but found {len(reg_ix)} many. Names of all regressors "
                f"in steps are {reg_ix_names}."
            )
        self._reg_ix = reg_ix[0]

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
        reg_ix = self._reg_ix
        _, ests = self._get_names_and_objects(self.steps_)

        for i in range(reg_ix - 1):
            X = ests[i].fit(X).transform(X)

        ests[reg_ix].fit(X, y)

        return self

    def _predict(self, X):
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
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        reg_ix = self._reg_ix
        _, ests = self._get_names_and_objects(self.steps_)

        for i in range(reg_ix - 1):
            X = ests[i].transform(X)

        y_pred = ests[reg_ix].predict(X)

        for i in range(reg_ix + 1, len(ests)):
            y_pred = ests[i].transform(y_pred)

        return y_pred
