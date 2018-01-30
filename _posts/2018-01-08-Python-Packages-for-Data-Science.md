---
layout: post
title: Python Packages for Data Science 
key: 20180108
tags:
  - info as list
lang: en
---

This blog is created to record the Python packages of data science found in daily practice or reading, covering the whole process of machine learning from visualization and pre-processing to model training and deployment.

This post is kept updating.

## Visualization

**[Scikit-plot](http://scikit-plot.readthedocs.io/en/stable/index.html)**
- The quickest and easiest way to plot machine learning result, built upon scikit-learn and matplotlib
- *Metrics Module* -- evaluation metrics, e.g. confusion matrix, ROC, etc.
- *Estimators Module* -- learning curve and features importance
- *Clusterer Module* -- elbow plot
- *Decomposition Module* -- PCA 2D projection and PCA component explained variances

**[Altair](https://altair-viz.github.io/)**
- Declarative statistical visualization, just like JMP but in Python
- Example: 
``` python
 # only need to define x, y and legend
alt.Chart(cars).mark_circle().encode(x='Horsepower',
									 y='Miles_per_Gallon',
									 color='Origin')
```
![altair](https://raw.githubusercontent.com/YestinYang/YestinYang.github.io/master/screenshots/2018-01-08_altair.png)

**[Visdom](https://github.com/facebookresearch/visdom)**
- alive data visualization dashboard
![Visdom](https://camo.githubusercontent.com/2b1b3f8ceb9b9379d59183352f1ca3f2e6bbf064/68747470733a2f2f6c68332e676f6f676c6575736572636f6e74656e742e636f6d2f2d6833487576625532563053666771675847694b334c50676845357671765330707a704f6253305967475f4c41424d466b36324a4361334b56755f324e565f344c4a4b614161352d74673d7330)

## Data Cleansing

**[datacleaner](https://github.com/rhiever/datacleaner)**
- Work with Pandas DataFrame
- Automatically complete the basic cleansing as below
	- Optionally drops any row with a missing value
	- Replaces missing values with the mode (for categorical variables) or median (for continuous variables) on a column-by-column basis
	- Encodes non-numerical variables (e.g., categorical variables with strings) with numerical equivalents

## Feature Selection

**[Facets](https://pair-code.github.io/facets/)**
- A visualization tool for descriptive statistical information of features, and relationship between any pair of features
- To help you fast understanding the features and make decision for feature selection and engineering

**[sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic)**
- Using genetic algorithm ([explanation on KD](https://www.kdnuggets.com/2017/11/rapidminer-evolutionary-algorithms-feature-selection.html))

## Auto Machine Learning

**[DEAP](https://github.com/DEAP/deap)**
- Genetic Algorithm / Evolutionary Computation Framework

**[MLBox](http://mlbox.readthedocs.io/en/latest/index.html)**
- [Tutorial on Automated Machine Learning using MLBox](https://www.analyticsvidhya.com/blog/2017/07/mlbox-library-automated-machine-learning/)

**[auto-sklearn](http://automl.github.io/auto-sklearn/stable/)**
- An automated machine learning toolkit and a drop-in replacement for a specific scikit-learn estimator
- Supporting classifers / regressors / preprocessers
```python
import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()  ## search the best one among all classifier
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
```

## Specific Data Types

### Image Recognition

[Detectron from Facebook](https://github.com/facebookresearch/Detectron)
- Pre-trained object detection with object masking, instead of object bounding

### Time Series Data

[Working with Time Series Data in Python](https://github.com/MaxBenChrist/awesome_time_series_in_python)

### Nature Language Processing

**[SpaCy](https://spacy.io/)**
- Industrial-Strength Natural Language Processing
- [Natural Language Processing Made Easy – using SpaCy (​in Python)](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/)

**[DeepSpeech](https://github.com/mozilla/DeepSpeech)**
- Tensorflow implementation of Speech-to-Text synthesis from Baidu

### Spatial Data

**[Rasterio](https://mapbox.github.io/rasterio/)**
- [My Favorite Tool: Rasterio](http://www.datacarpentry.org/blog/sare-favorite/)

## Deployment

**[Flask](http://flask.pocoo.org/)**
- [Tutorial to deploy Machine Learning model in Production as API with Flask](https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/)

**Binder 2.0**
- Interactive online Jupyter notebook
- [Introducing Binder 2.0 — share your interactive research environment](https://elifesciences.org/labs/8653a61d/introducing-binder-2-0-share-your-interactive-research-environment?utm_content=buffer44b80&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)
- [Binder 2.0, a Tech Guide](https://blog.jupyter.org/binder-2-0-a-tech-guide-2017-fd40515a3a84)

## Mobile Machine Learning

**[TuriCreate](https://github.com/apple/turicreate)**
- A Simplified Machine Learning Library for iOS
- [How to build your first Machine Learning model on iPhone (Intro to Apple’s CoreML)](https://www.analyticsvidhya.com/blog/2017/09/build-machine-learning-iphone-apple-coreml/)

**[Baidu Mobile ML](https://github.com/baidu/mobile-deep-learning)**

## Others

**[tqdm](https://pypi.python.org/pypi/tqdm)**
- progression monitor
76%|████████████████████████████        | 7568/10000 [00:33<00:10, 229.00it/s]

**[Docker](https://www.docker.com/)**
- Portable environment for your code
- [How Docker Can Help You Become A More Effective Data Scientist](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5)

## Commercial Tools

**[KNIMI](https://www.knime.com/)**
- GUI interface for data science, no coding required
- [Building your first machine learning model using KNIME (no coding required!)](https://www.analyticsvidhya.com/blog/2017/08/knime-machine-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)

**[Orange](https://orange.biolab.si/)**
- GUI interface for data science, like Klarity ACE

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDE2MTA1MDkyXX0=
-->