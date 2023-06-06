LibRecommender
==============

.. image:: https://img.shields.io/github/actions/workflow/status/massquantity/LibRecommender/wheels.yml?branch=master
   :target: https://github.com/massquantity/LibRecommender/actions/workflows/wheels.yml
   :alt: Build status

.. image:: https://readthedocs.org/projects/librecommender/badge/?version=stable
    :target: https://librecommender.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. image:: https://github.com/massquantity/LibRecommender/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/massquantity/LibRecommender/actions/workflows/ci.yml
   :alt: CI status

.. image:: https://codecov.io/gh/massquantity/LibRecommender/branch/master/graph/badge.svg?token=BYOYFBUJRL
   :target: https://codecov.io/gh/massquantity/LibRecommender
   :alt: Codecov status

.. image:: https://img.shields.io/pypi/v/LibRecommender?color=blue
   :target: https://pypi.org/project/LibRecommender/
   :alt: Pypi version

.. image:: https://static.pepy.tech/personalized-badge/librecommender?period=total&units=international_system&left_color=grey&right_color=lightgrey&left_text=Downloads
   :target: https://pepy.tech/project/librecommender
   :alt: Downloads

.. image:: https://app.codacy.com/project/badge/Grade/860f0cb5339c41fba9bee5770d09be47
   :target: https://www.codacy.com/gh/massquantity/LibRecommender/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=massquantity/LibRecommender&amp;utm_campaign=Badge_Grade
   :alt: Codacy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json
   :target: https://github.com/charliermarsh/ruff
   :alt: Ruff

.. image:: https://img.shields.io/github/license/massquantity/LibRecommender?color=ff69b4
   :target: https://github.com/massquantity/LibRecommender/blob/master/LICENSE
   :alt: License

------------------

**LibRecommender** is an easy-to-use recommender system focused on end-to-end recommendation process.
It contains a training(`libreco <https://github.com/massquantity/LibRecommender/tree/master/libreco>`_) and serving(`libserving <https://github.com/massquantity/LibRecommender/tree/master/libserving>`_)
module to let users quickly train and deploy different kinds of recommendation models.

**The main features are:**

+ Implements a number of popular recommendation algorithms such as FM, DIN, LightGCN etc. See `full algorithm list <https://github.com/massquantity/LibRecommender#references>`_.
+ A hybrid recommender system, which allows users to use either collaborative-filtering or content-based features. New features can be added on the fly.
+ Low memory usage, automatically convert categorical and multi-value categorical features to sparse representation.
+ Supports training for both explicit and implicit datasets, as well as negative sampling on implicit data.
+ Provides end-to-end workflow, i.e. data handling / preprocessing -> model training -> evaluate -> save/load -> serving.
+ Supports cold-start prediction and recommendation.
+ Supports dynamic feature and sequence recommendation.
+ Provides unified and friendly API for all algorithms.
+ Easy to retrain model with new users/items from new data.

Quick Start
-----------

The two tabs below demonstrate the process of train, evaluate, predict, recommend and cold-start.

1. **Pure** example(collaborative filtering), which uses ``LightGCN`` model.

2. **Feat** example(use features), which uses ``YouTubeRanking`` model.

.. tab:: pure_example

    .. literalinclude:: ../../examples/pure_example.py
       :caption: From file `examples/pure_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/pure_example.py>`_
       :name: pure_example.py
       :lines: 15-

.. tab:: feat_example

    .. literalinclude:: ../../examples/feat_example.py
       :caption: From file `examples/feat_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/feat_example.py>`_
       :name: feat_example.py
       :lines: 10-


.. toctree::
   :maxdepth: 1
   :caption: Intro
   :hidden:

   installation
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/data_processing
   user_guide/feature_engineering
   user_guide/model_train
   user_guide/evaluation_save_load
   user_guide/recommendation
   user_guide/embedding
   user_guide/model_retrain

.. toctree::
   :maxdepth: 1
   :caption: Deploy
   :hidden:

   serving_guide/python
   serving_guide/rust
   serving_guide/online

.. toctree::
   :maxdepth: 1
   :caption: Internal
   :hidden:

   internal/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/data/index
   api/algorithms/index
   api/evaluation
   api/serialization

.. toctree::
   :maxdepth: 1
   :caption: Outro
   :hidden:

   Docker <https://github.com/massquantity/LibRecommender/tree/master/docker>
   Github <https://github.com/massquantity/LibRecommender>

Indices and tables
..................

* :ref:`genindex`
