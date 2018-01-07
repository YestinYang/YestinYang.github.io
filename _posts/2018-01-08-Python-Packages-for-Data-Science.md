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

##  Visualization

**[Altair](https://altair-viz.github.io/)**
- Declarative statistical visualization, just like JMP but in Python
- Example: 
``` python
 # only need to define x, y and legend
alt.Chart(cars).mark_circle().encode(x='Horsepower', y='Miles_per_Gallon', color='Origin')
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI2NjAzNzYwM119
-->