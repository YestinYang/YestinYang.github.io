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

## Features Selection

**[sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic)**
- Using genetic algorithm ([explanation on KD](https://www.kdnuggets.com/2017/11/rapidminer-evolutionary-algorithms-feature-selection.html))

## Specific Data Types

### Time Series Data

Working Time Series Data in Python https://github.com/MaxBenChrist/awesome_time_series_in_python

### Nature Language Processing

Natural Language Processing Made Easy - using SpaCy (in Python)

**[DeepSpeech](https://github.com/mozilla/DeepSpeech)**
- Tensorflow implementation of Speech-to-Text synthesis from Baidu

### Spatial Data

**[Rasterio](https://mapbox.github.io/rasterio/)**
- [Introduction](http://www.datacarpentry.org/blog/sare-favorite/)

### 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk3MDk4OTEzN119
-->