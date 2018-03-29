---
layout: post
title: Improving Deep Neural Networks - Deep Learning Specialization 2
key: 20171204
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

Full Course Name: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

## W1: Practical aspects of Deep Learning

### Setting up your Machine Learning Application

Applied ML is a highly iterative process, need a lot cycles as below
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511400839878.png){:.border}

#### Train / Development / Test sets

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511402240566.png){:.border}

#### Bias and Variance

- In DL era, less discussion about trade-off, but bias and variance themselves
	- (show in V3) Trade-off only happens in pre-DL era, because there is no tool can only reduce bias without increasing variance
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511405106210.png){:.border}

#### Systematic Way of Improve DL

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511406405256.png){:.border}

### Regularizing your neural network

#### L2 Regularization

**V1: L2 regularization**
- add a regularization part in cost function
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511408128174.png){:.border}

**V2: Why can prevent overfitting**
- will slightly increase bias but significantly decrease variance
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511421395750.png){:.border}

#### Dropout Regularization

**V3&V4:** 
- Used when training only, and do not use when testing
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511423874207.png){:.border}

#### Others

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511424935674.png){:.border}

### Setting up your optimization problem (to speed up and debug)

#### Normalizing Inputs $A^{[0]}$

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511602072164.png){:.border}

#### Better Initialization $W^{[L]}$

- extreme deep NN accumulates $Z^{[L]}$ values and generate extreme large or small outputs, which leads to study difficulties
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511604381201.png){:.border}

#### Debug of Gradient Checking

to verify the back propagation is correctly implemented

**V4: numerical approx of gradients**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511616204623.png){:.border}

**V5: gradient checking**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511616783269.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511795777569.png){:.border}

**V6: gradient check for improvement**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1511617012480.png){:.border}


## W2: Optimization algorithms (to speed up training)

### Mini-batch Gradient Descent

**V1&V2: mini-batch gradient descent**
- split huge training set into small batches
- compared to batch gradient descent, mini-batch can make progress without training the whole training set. So each descent step is faster but oscillating

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512002838033.png){:.border}

### Gradient Descent with Momentum

**V3&V4: exponentially weighted moving averages**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512005572720.png){:.border}

**V5: bias correction in early phase of weighted average**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512006401556.png){:.border}

**V6: gradient descent with momentum**
- using weighted average to reduce oscillation especially in mini-batch gradient descent (batch GD has much weaker oscillation)
- it is an idea of weighted average, and can be explained as physical movement affected by friction and acceleration
- has negligible help on small learning rate and simplistic dataset 
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512008165207.png){:.border}

### RMSprop

**V7: RMSprop**

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512012576865.png){:.border}

### Adam Optimization

**V8: Adam optimization = momentum + RMSprop**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512013255708.png){:.border}

### Learning Rate Decay

**V9: learning rate decay in training process**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512022485728.png){:.border}

### Understanding Local Optima

**V10: problems of local optima**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512023150624.png){:.border}


## W3: Hyperparameter tuning, Batch Normalization and Programming Frameworks

### Hyperparameter Tuning

**V1: tuning process**
- use random search for higher efficiency, not grid search
- from coarse to fine

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512307158812.png){:.border}

**V2: appropriate scale for searching**
- choose the scale to search efficiently

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512307400927.png){:.border}

**V3: tuning practice**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512307498391.png){:.border}

### Batch Normalization

**V1: normalizing activations in NN**
- it uses the same logic of normalizing input partially, but use on each activation functions by normalizing Z (or A in rarer case)
- however, we don't want all activation functions are distributed at $N(0,1)$ as input $X$, so we use beta and gamma as learn parameters to tune the mean and variance
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512356740610.png){:.border}

**V2: add in batch norm**

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512356674989.png){:.border}

**V3: why works**
- to get faster gradient descent as same as normalized input
- to make weights of deeper layer more robust to the weight changes of earlier layers，and let each layer train more relied on itself
- slightly regularization effect when combine with mini-batch
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512357853700.png){:.border}

**V4: batch norm at test time**
-  estimate mean and variance by weighted average across mini-batches --> so need to define mini-batch size during testing

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512357992489.png){:.border}

### Multi-class Classification

**V1: softmax regression**
- use softmax as last layer
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512359766490.png){:.border}

**V2: training a softmax classifier**

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512360819247.png){:.border}

### Introduction to Programming Frameworks

**V1: Deep learning frameworks**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512371062512.png){:.border}

**V2: TensorFlow**
- only define cost function (forward prop) by hand, and the backward part will be automatically done
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_1512372216189.png){:.border}

------

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-04_cert.png){:.border}