---
layout: post
title: Deep Learning Specialization 3 -- Structuring Machine Learning Projects
key: 20171206
tags:
  - Coursera
  - Notes
  - Study
  - Deep Learning
  - Data Science
lang: en
mathjax: true
mathjax_autoNumber: true
---

deeplearning.ai by Andrew Ng on Coursera

## W1: ML Strategy 1

### Introduction to ML strategy

**V1: course target -- how to effectively tune your ML**

**V2: orthogonalization of the tuning**
- each operation only tunes one factor, like width or length of a TV
- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511865881114.png){:.border}

### Setting up your goal

**V1: single real number evaluation metric**
- speed up your iteration by easier evaluating whether certain tuning is working
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511872231128.png){:.border}

**V2: satisficing and optimizing metric, for the case cannot use single real number**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511872797115.png){:.border}

**V3: train/dev/test sets**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512089953122.png){:.border}

**V4: size of dev/test sets**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511878635393.png){:.border}

**V5: when to change "dev/test sets+metric"**
- Doing orthogonalization for "dev/test+metric" setup
	- **Node 1: Plan Target** -- Define a "dev/test+metric", even it is not perfect
	- **Node 2: Shoot Target** -- separated from Node 1, evaluate how well the metric works, and optimize metric if needed


### Comparing to human-level performance

**V1: human-level performance**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511917923998.png){:.border}

**V2: avoidable bias**
- human-level performance can tell the expectation on how well your ML does on training sets, but not too well
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511941965152.png){:.border}

**V3: define human-level performance and error analysis**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511921842800.png){:.border}

**V4: ML surpasses human-level performance**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511938756249.png){:.border}

**V5: improving your model performance**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511939220791.png){:.border}


### Machine Learning flight simulator

Decision making senario
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1511942582576.png){:.border}


## W2: ML Strategy 2

### Error Analysis

**V1: error analysis saving huge effort on wrong way**
> Decide next step by wisely balancing effort and output (best input-output ratio)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512088732750.png){:.border}

**V2: clean up incorrectly labeled data in dev/test set**
- For training set, normally no need to correct the incorrect label, because
	1. DL is robust to random errors (but sensitive to systematic errors)
	2. The mislabel error (false positive and false negative) is already predicted correctly by current model from current training set. So if training set is updated, the outcome will even hard to control (even more different from dev/test set)
	3. It is impossible to clean up all labels in training set
- To be caution, after correcting label in dev/test set, training set and dev/test set have slightly different distribution. So need to add in train-dev set
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512090644335.png){:.border}

**V3: build first system quickly, then iterate**
> Learn from mistake or error or failure

- for new topic, the general rule of working is
	1. set up dev/test set and metrics
		- set a target
	2. build an initial system quickly and dirty without overthinking
		- train training set quickly: fit the parameters
		- dev set: tune the parameters
		- test set: assess the performance
	3. use bias-variance analysis & error analysis to prioritize next steps

- for those problems having strong prior experience or academic literatures, go for the existing knowledge directly

### Mismatched Training and Dev/Test Set

**V1: training and testing on different distributions**
- build dev/test set only with the data your final product actually cares about 
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512116047609.png){:.border}

**V2: bias-variance with mismatched distribution**
- add in training-dev set to evaluate variance while training set has different distribution from dev/test set
- *refer to pdf for case example*
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512182726250.png){:.border}

**V3: addressing data mismatch**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512184735471.png){:.border}

### Learning from Multiple Tasks

**V1: transfer learning**
- used when you have very little target data but a large dataset of related object, like in image recognition and speech recognition area
- train NN with large dataset and then fix all parameters but re-initialize those of last hidden layer, and finally re-train the model with brand new small dataset
	- there is another case that if the brand new dataset is not so small, then can use the pre-learning parameters as initialization of fine-tuning, then re-train the whole NN with new dataset
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512200618795.png){:.border}

**V2: multi-task learning**
- analogize to multiple outputs in traditional ML
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512203141600.png){:.border}

### End-to-end Deep Learning

**V1: examples of end-to-end DL**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512227231753.png){:.border}

**V2: when end-to-end is better**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-06_1512227310789.png){:.border}
