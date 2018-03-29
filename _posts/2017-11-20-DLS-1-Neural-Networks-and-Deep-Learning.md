---
layout: post
title: Neural Networks and Deep Learning - Deep Learning Specialization 1
key: 20171120
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

## W1: Introduction to Deep Learning

### Supervised Neural Network

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510368169137.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510381658099.png){:.border}


### Why Neural Network is Taking Off

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510379424629.png){:.border}


## W2: Neural Network Basic illustrated by Logistic Regression

### Notation and Matrix

logistic regression weight stack in rows, and samples stack in columns (transposed matrix compared to traditional ML)

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510714157165.png){:.border}


### Logistic Regression

V2:
- weight and intercept is better to be separated in NN
- use sigmoid function to transform the result into [0,1] so that $\hat{y}$ is the probability of 1 given x

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510714518762.png){:.border}

V3: 
- special loss function for logistic regression, to make the optimization problem as convex so that gradient descent works to find global optima (not local optima)
- loss function is for single sample, while cost function is for entire training set (cost of your parameters; average of total loss function)

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510715039949.png){:.border}


### Gradient Descent and Computation Graph

V4: gradient descent
- step along the steepest direction on a convex surface started at the initial point
- step size is the learning rate

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510715588776.png){:.border}

V7&V8: computation graph
- forward propagation (blue): yield the output 
- backward propagation (red): compute gradients or derivative (use $dvar$ as notation for derivative of final output respected to intermediate variable)

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510710645200.png){:.border}

#### In Logistic Regression Context

V9&10: gradient descent of logistic regression
- use vectorization to avoid large for loop

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510718616019.png){:.border}


### Vectorization

V1&V2: use built-in function to calculate(for example, np.dot = dot multiplication of two matrix), instead of using for loops

V3&V4: vectoring logistic regression
- forward propagation
- backward propagation and gradient descent

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510838206633.png){:.border}

V5: broadcasting rule of python
- how python transform matrix when the size is unmatched during matrix calculation

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510735954470.png){:.border}

V6: bug-free numpy coding tips
- avoid using rank 1 array, but commit shape when create the matrix

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510753398450.png){:.border}


### Programing Practice

- Implemented each function separately: initialize(), propagate(), optimize(). Then you built a model()
- use assert to verify shape or types of matrix


## W3: Shallow Neural Network

Introduction of real structure of neural network, converting from computation graph of logistic regression

### Neural Network Representation
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510996642678.png){:.border}


### Vectorization

V3: with single sample
- NN version is doing logistic regression multiple times
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510890294210.png){:.border}

V4&V5: with multiple samples
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510891818391.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510970374372.png){:.border}

### Activation Functions and Gradient Descent

V6: different choices of activation functions
- activation function: determine the output
- for logistic regression, we normally use sigmoid as activation function, but some other functions are working better

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510971902507.png){:.border}

V7&V8: why we need non-linear activation functions
- if use linear ones, no matter how many hidden layers it has, there is no difference for the output
- special case like regression model, may use linear activation function in output layer, but in hidden layers still need to use non-linear one
> Neural network is like a system of non-linear superposition
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510975496058.png){:.border}

V9&V10: gradient descent
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510981566094.png){:.border}

### Random Initialization

V11:
- if set 0 for all weight values, all hidden units/nodes/neurons will be calculated symmetrically, and the outputs are same for each units which is meaningless for NN
- we use random initialization to get small values to avoid stuck when using tanh or sigmoid
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1510994738769.png){:.border}


## W4: Deep Neural Network

### L-layer Neural Network Notation

V1: L-layer neural network notation
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1511098179388.png){:.border}

### Forward and Backward Building Blocks

V2: forward propagation
- similar to 2 layer NN, but using for loop for multiple layers calculation

V3: matrix dimensions verification
- (like dimensional analysis in physics)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1511142728497.png){:.border}

V4: why deep can work better
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1511143855036.png){:.border}

V5&V6: building blocks of deep NN
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1511147435676.png){:.border}

### Hyperparameters

V7: hyperparameters
- hyperparameters determine the parameters
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-11-20_1511148154707.png){:.border}











