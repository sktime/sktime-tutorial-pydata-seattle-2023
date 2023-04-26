<a href="https://sktime.net"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo-no-text.jpg?raw=true)" width="175" align="right" /></a>

`skbase` - a workbench for creating scikit-learn like parametric objects and libraries
======================================================================================

**Welcome to the skbase tutorial at PyData Seattle 2023**

`skbase` is a meta-toolkit that makes it easy to build your own python package that follows `scikit-learn` design patterns, e.g., parametric composable objects, and fittable objects. It contains standalone `BaseObject` and `BaseEstimator` base classes, that is, base class templates to write your own base classes, templateable test classes and object checks, object retrieval and inspection, and more.

`skbase` is the base component of `sktime` turned workbench, and is an evolution on the base class framework of `sklearn`.

[sktime]: https://sktime.net

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime-tutorial-pydata-seattle-2023/main?filepath=notebooks)

If you are unfamiliar with `sklearn` or `sktime`, consider working through **general sklearn and sktime introduction tutorials** first:

:movie_camera: **[general sktime intro tutorial](https://github.com/sktime/sktime-tutorial-pydata-glboal-2021) from PyData Global 2021**\
:tv: [youtube video of sktime intro at PyData Global 2021](https://www.youtube.com/watch?v=ODspi8-uWgo)

## :bulb: Description

The workshop will walk the audience through an example of creating their own package with parametric objects, custom base classes and objects inheriting from these, and a full testing framework.

This will also showcase skbase's (https://github.com/sktime/skbase) core functionality which is contained in submodules:

* `skbase.base` provides: `BaseObject` - parameteric object with get/set_params, tag system, etc; `BaseEstimator`, for objects with fit, with `is_fitted`, `get_fitted_params`; mixin classes such as BaseMetaObject for homogenous and heterogeneous composites (e.g., ensembles, pipelines, graph objects).
* `skbase.lookup` provides search tools such as all_objects that retrieves all `BaseObject`-s with certain tags from a project.
* `skbase.validate` provides tools for validating and comparing `BaseObject`-s and collections of `BaseObject`-s
* `skbase.testing` provides tools for testing `BaseObject`-s, and for setting up testing frameworks and object checkers, for dependent base classes.

## sktime tutorials

`skbase` has been consolidated from the base class frameworks of `sklearn` and `sktime`.

You may find the following tutorials for `sktime` useful:

- [PyData 2021 - Intro to sktime](https://www.youtube.com/watch?v=ODspi8-uWgo)

- [Pydata 2022 - How to implement your own estimator in sktime](https://www.youtube.com/watch?v=S_3ewcvs_pg)

- [Pydata 2022 - Advanced Forecasting Tutorial](https://www.youtube.com/watch?v=4Rf9euAhjNc)

## :rocket: How to get started

You have different options how to run the tutorial notebooks:

* Run the notebooks in the cloud on [Binder] - for this you don't have to install anything!
* Run the notebooks on your machine. [Clone] this repository, get [conda], install the required packages (`skbase`, `sktime`, `pytest`, `seaborn`, `jupyter`) in an environment, and open the notebooks with that environment. For detail instructions, see below. For troubleshooting, see sktime's more detailed [installation instructions].
* or, use python venv, and/or an editable install of this repo as a package. Instructions below.

[Binder]: https://mybinder.org/v2/gh/sktime/sktime-tutorial-pydata-seattle-2023/main?filepath=notebooks
[clone]: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
[conda]: https://docs.conda.io/en/latest/
[installation instructions]: https://www.sktime.net/en/latest/installation.html

## :wave: How to contribute

If you're interested in contributing to sktime, you can find out more how to get involved [here](https://www.sktime.net/en/stable/get_involved.html).

Any contributions are welcome, not just code!

## Installation instructions in detail

### Cloning the repository

To clone the repository locally:

`git clone https://github.com/sktime/sktime-tutorial-pydata-seattle-2023.git`


### Using conda env

#### option 1: installing requirements manually

1. Create a python virtual environment:  
`conda create -y -n pydata_skbase python=3.9`
2. Install required packages:  
`conda install -y -n pydata_skbase pip scikit-base sktime pytest seaborn jupyter`
3. Activate your environment:  
`conda activate pydata_skbase`
4. If using jupyter: make the environment available in jupyter:  
`python -m ipykernel install --user --name=pydata_skbase`

#### option 2: installing repo as package

1. Create a python virtual environment:  
`conda create -y -n pydata_skbase python=3.9`
2. Make sure the environment has pip:  
`conda install -y -n pydata_skbase pip`
3. Activate your environment:  
`conda activate pydata_skbase`
4. Install the package in development mode:  
`pip install -e .`
5. If using jupyter: make the environment available in jupyter:  
`python -m ipykernel install --user --name=pydata_skbase`

### Using python venv

#### option 1: installing requirements manually

1. Create a python virtual environment:  
`python -m venv .venv`
2. Activate your environment:  
`source .venv/bin/activate`
3. Install the requirements:  
`pip install scikit-base sktime pytest seaborn jupyter`
4. If using jupyter: make the environment available in jupyter:  
`python -m ipykernel install --user --name=pydata_skbase`

#### option 2: installing repo as package

1. Create a python virtual environment:  
`python -m venv .venv`
2. Activate your environment:  
`source .venv/bin/activate`
3. Install the package in development mode:  
`pip install -e .`
4. If using jupyter: make the environment available in jupyter:  
`python -m ipykernel install --user --name=pydata_skbase`
