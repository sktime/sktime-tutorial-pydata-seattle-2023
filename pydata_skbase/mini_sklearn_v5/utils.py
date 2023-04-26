# -*- coding: utf-8 -*-
"""Estimator checking utility."""
from .tests.test_estimators import TestAllObjects, TestAllRegressors


def check_estimator(
    estimator,
    raise_exceptions=False,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
):
    """Run all tests on one single estimator.

    Tests that are run on estimator:
        all tests in test_all_estimators
        all interface compatibility tests from the module of estimator's scitype
            for example, test_all_forecasters if estimator is a forecaster

    Parameters
    ----------
    estimator : estimator class or estimator instance
    raise_exceptions : bool, optional, default=False
        whether to return exceptions/failures in the results dict, or raise them

        * if False: returns exceptions in returned `results` dict
        * if True: raises exceptions as they occur

    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (test/function name string) of tests to run.
        sub-sets tests that are run to the tests given here.
    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest test-fixture combination codes, which test-fixture combinations to run.
        sub-sets tests and fixtures to run to the list given here.
        If both tests_to_run and fixtures_to_run are provided, runs the *union*,
        i.e., all test-fixture combinations for tests in tests_to_run,
            plus all test-fixture combinations in fixtures_to_run.
    verbose : str, optional, default=True.
        whether to print out informative summary of tests run.
    tests_to_exclude : str or list of str, names of tests to exclude. default = None
        removes tests that should not be run, after subsetting via tests_to_run.
    fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
        removes test-fixture combinations that should not be run.
        This is done after subsetting via fixtures_to_run.

    Returns
    -------
    results : dict of results of the tests in self
        keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
        entries are the string "PASSED" if the test passed,
        or the exception raised if the test did not pass
        returned only if all tests pass, or raise_exceptions=False

    Raises
    ------
    if raise_exceptions=True,
    raises any exception produced by the tests directly
    """
    results = TestAllObjects().run_tests(
        obj=estimator,
        raise_exceptions=raise_exceptions,
        tests_to_run=tests_to_run,
        fixtures_to_run=fixtures_to_run,
        tests_to_exclude=tests_to_exclude,
        fixtures_to_exclude=fixtures_to_exclude,
    )

    if estimator.get_class_tags()["estimator_type"] == "regressor":
        results_add = TestAllRegressors().run_tests(
            obj=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
        )
        results.update(results_add)

    return results
