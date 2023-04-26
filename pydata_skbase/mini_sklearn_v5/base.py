# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
from skbase.base import (
    BaseEstimator as _BaseEstimator,
    BaseMetaEstimator as _BaseMetaEstimator,
)


class _CommonTags():
    """Mixin for common tag definitions to all estimator base classes."""

    # config common to all estimators
    _config = {
        "return_type": "pandas",
        # determines return of predict or transform type methods
        # "pandas" - pd.DataFrame
        # "numpy" - 2D np.ndarray
    }

    _tags = {"estimator_type": "estimator"}


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""


class BaseRegressor(BaseEstimator):
    """Base class for supervised regressors."""

    _tags = {"estimator_type": "regressor"}

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

        # set fitted flag to True
        self._is_fitted = True

        return self._fit(X, y)

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
        raise NotImplementedError

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
                "X in predict must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in predict found X.columns = {X.columns}"
            )

        y_pred = self._predict(X)

        return_config = self.get_config()["return_type"]
        if return_config == "numpy":
            return y_pred.values
        elif return_config == "pandas":
            return y_pred
        else:
            raise ValueError(
                f"unexpected value in config return_type of {self}, "
                f'must be one of strings "numpy", "pandas", but found {return_config}'
            )

    def _predict(self, X):
        """Predict labels for feature data frae.

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
        raise NotImplementedError


class BaseTransformer(BaseEstimator):
    """Base class for transformers."""

    _tags = {"estimator_type": "transformer"}

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

        # set fitted flag to True
        self._is_fitted = True

        return self._fit(X)

    def _fit(self, X):
        """Fit transformer to training data.

        Private _fit called from fit.

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
        raise NotImplementedError

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

        Xt = self._transform(X)

        return_config = self.get_config()["return_type"]
        if return_config == "numpy":
            return Xt.values
        elif return_config == "pandas":
            return Xt
        else:
            raise ValueError(
                f"unexpected value in config return_type of {self}, "
                f'must be one of strings "numpy", "pandas", but found {return_config}'
            )

    def _transform(self, X):
        """Transform data with fitted transformer.

        Private _transform called from transform.

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
        raise NotImplementedError
