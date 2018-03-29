---
layout: post
title: Machine Learning - Andrew Ng @ Coursera
key: 20180129
tags:
  - Coursera
  - Notes
  - Study
  - Machine Learning
  - Data Science
lang: en
mathjax: true
mathjax_autoNumber: true
---

## Week 1: Introduction

### Application of ML
- Database mining
	- large dataset growth of automation/web
- Application can't program by hand
	- handwriting recognition, NLP, computer vision
- Self-customizing programs
	- recommendations
- Understanding human learning (brain, real AI)

### Definition of ML
> Tom Mitchell: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T as measured by P, improves with experience E

**Supervised Learning:** "right answers" is given
- Regression
- Classification

**Unsupervised Learning:** no idea about the result
- Social network analysis / news grouping / ..
- Cocktail party problem: separate the overlapping sounds
	- easily implement in Octave: [W,s,v] = svd((repmat(sum(x.\*x,1),size(x,1),1).\*x)\*x');
	- Octave is commonly used for prototyping machine learning algorithm before going to industry

### Model and Cost Function

**Model Representation**
![@h: hypothesis that mapping X to Y](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515465654321.png)

**Training a Model**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515466808544.png)

**Cost Function:** the target we are going to minimize
- Square error function for regression (also used in DL)
$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$

### Parameter Learning

**Gradient Descent:** a general algorithm used all over the place in ML
- gradient $$\frac{d}{d\theta} J(\theta)$$ represents that when $$\theta$$ $$+$$ or $$-$$ a unit, how much $$J$$ will $$+$$ or $$-$$ correspondingly
- then because we want $$J$$ to decrease, we update $$\theta$$ with partial gradient (partial unit of $$\theta$$ change) controlled by learning rate
- update all parameters at the same time, so that the cost function keeps same for all parameters in single round 

**Gradient Descent of Linear Regression**
- it is always a convex function without local optima
- Batch Gradient Descent: consider all training examples when doing gradient descent, corresponding to minibatch
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515484236489.png)![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515484175265.png)

### Linear Algebra Review

**Matrix:** $$m\times n$$ (2d matrix)
**Vector:** $$n\times 1$$ (n-dimensional vector)
**Scalar:** an single value object, not a matrix or vector

**Properties:**
- not commutative $$A∗B \neq B∗A$$
- associative $$(A∗B)∗C =  A∗(B∗C)$$
- Identity matrix $$I_{3\times 3} = \begin{bmatrix} 1 & 0 & 0 \newline 0 & 1 & 0 \newline 0 & 0 & 1 \newline \end{bmatrix}$$ has $$A∗I = I∗A = A$$

**Inverse:** $$A*A^{-1} = A^{-1}*A = I$$
- Matrix without inverse matrix is called singular or degenerate matrix

**Transpose:** $$A_{ij} = A^T_{ji}$$

## Week 2: Linear Regression with Multiple Variables

### Multivariate Linear Regression

**Expression:** $$\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}$$
- (n+1)-dimensional `vector` where keep $$x_0 = 1$$ (case with only one sample)

**Gradient Descent Optimization 1: Features Scaling**
- If all features are on similar scale, gradient descent can converge more quickly
- Mean normalization is commonly used
	- $$x_i := \dfrac{x_i - \mu_i}{s_i}$$ where $$s_i$$ can be standard deviation or range
	- target is to make `approximately` $$-1\le x_i\le 1$$ or $$-0.5\le x_i\le 0.5$$

**Gradient Descent Optimization 2: Learning Rate**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515562969081.png)

**Feature Engineering:**
- Combine features, such as using square instead of length + width
- Transfer to polynomial regression
	- created new features $$x_2$$ and $$x_3$$ where $$x_2 = x_1^2$$ and $$x_3 = x_1^3$$
	- `Normalize` $$x_1$$, $$x_2$$ and $$x_3$$ to similar scalar
	- Train model as multivariate linear regression

### Computing Parameters Analytically

**Normal Equation:** alternative analytical method of gradient descent in linear regression
- use $$m \times n$$ `matrix`
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515567296175.png)
- pros: no need to consider scalar / faster
- if $$X^TX$$ is non-invertible
	- pseudo inverse function can handle
	- reasons
		- redundant features with linearly dependency: delete
		- too many features (like $$m \le n$$): delete or use regularization

## Week 3: Logistic Regression

### Classification and Representation

**Logistic Regression Representation:**
- exactly a node in DL --> logistics regression + sigmoid function (or ReLu in DL)
- output the `probability` of positive class $$h_\theta(x)=g(\theta^Tx)=P(y=1\vert x;\theta)$$

**Decision Boundary:**
- property of hypothesis, not of dataset
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515664737110.png)

### Logistic Regression Model

**Cost Function:**
$$\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}$$
$$\text{or}$$
$$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515666797283.png)![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515666801993.png)

**Gradient Descent:**
$$\theta := \theta - \alpha\frac{1}{m}\sum_{i=1}^{m}[(h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}]$$
- same as linear regression but with the hypothesis changed
- features scaling also helps on faster descending

**Advanced algorithms other than gradient descent:**
- Conjugate gradient / BFGS / L-BFGS
	- pros: auto pick $$\alpha$$ / faster descent
	- cons: complex
- Call integrated tools to apply advanced algorithms
```matlab
% Create function for cost and gradient
function [J, grad] = costFunction(theta, X, y)
  J = [...code to compute J(theta)...];
  for i = 1:size(X)
	  grad(i) = [...code to compute derivative of J(theta)...];
  end
end

% Set your algorithm
% 'GradObj'=on tells that input function has both cost and gradient
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);

% Call fminunc to find the local minimum of unconstrained multivariable function
% @(t) tells that theta is the updating input in this function
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% For constrained multivariable function use fmincon
```

### Multiclass Classification

**One-vs-All:**
- Train multiple hypothesis returning probability of belonging to each calss
- Run `max` to output the class with highest prob
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515726957477.png)

**PS:** also can use softmax here, but need to replace all sigmoid activate functions with a single softmax activate function

### Solving the Problem of Overfitting

**Overfitting:** high variance
- Caused by a complicated function made by too many features
- Solution: reduce features manually or automatically (may lose useful information) / regularization

**Regularization on Cost Function:**
- the smaller parameters correspond to simpler hypothesis
$$\displaystyle J(\theta) = \frac{1}{2m}\left[\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^n\theta_j^2\right]$$
- conventionally does not include $$\theta_0$$ where $$j=0$$

**Gradient Descent with Regularization:**
- Change the content of $$h_\theta(x^{(i)})$$ corresponding to linear regression or logistic regression
$$\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}$$

- For linear regression, can also use Normal Equation
$$\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\text{, making 1st parentheses always invertible}\end{align*}$$

## Week 4: Neural Networks: Representation

### Non-linear Model

**Why we need non-linear model:** non-linear decision boundary is more flexible to fit various cases/datasets

**Why we need neural network:** logistic regression with polynomial features can provide non-linear result but it is a costly approach *--> we need another way to build non-linear model*
- because polynomial LR needs to add in all possible combinations between features with different order/degree, which will explode when there are already a lot of features

### Neural Networks

(refer to Deep Learning Specialization)

### Applications

**Building XNOR logic:** non-linear decision boundary
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1515827261588.png)

## Week 5: Neural Networks: Learning

### Cost Function and Backpropagation

**Cost function:**
$$\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}$$

**Backpropagation algorithm:**
(refer to better explanation in deep learning specialization)

### Backpropagation in Practice

**Unroll and reshape theta and gradient:**
- due to `fminunc` only handle theta as vector
```matlab
% Unroll to vector
thetaVector = [ Theta1(:); Theta2(:); Theta3(:) ]
deltaVector = [ D1(:); D2(:); D3(:) ]

% Reshape to matrix
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

**Gradient checking:**
- Use 2-sided difference to estimate gradient, then compare it with backprop gradient
$$\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}$$

**Random initialization:** symmetry breaking

**Flow of training:**
1. Randomly initialize the weights
2. Implement forward propagation to get hΘ(x(i)) for any x(i)
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

## Week 6: Advice for Applying Machine Learning

### Evaluating a Learning Algorithm

**Evaluating a Hypothesis:** 0.7 train / 0.3 test split
- Linear regression: cost
- Logistic regression: test error (not cost)



**Model Selection & Hyperparameter Tuning:** 0.6 train / 0.2 validation / 0.2 test split
- Parameters are fitting training set
- Hyperparameters are fitting validation set for predicting new data
- Finally, use test set to report the generalization ability of selected model with hyperparameters and trained parameters

### Bias vs. Variance

**Regularization and Bias-Variance:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516162002799.png) ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516175002063.png)

**Machine Learning Diagnostic with Learning Curves:** to understand bias-variance of your model
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516175634900.png)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516175696214.png)

**What to Do Next:** based on the diagnostic result of bias-variance
- Fixed high variance (overfitting)
	- getting more training examples
	- trying smaller sets of features
	- increasing lambda of regularization
	- smaller neural network
- Fixed high bias (underfitting)
	- adding features
	- adding polynomial features
	- decreasing lambda of regularization
	- larger neural network

### Case Study: Building a Spam Classifier

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516328393933.png)

> *Brainstorming* before starting helps you saving plenty of time

**Recommended Approach:**
1. Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
2. Plot learning curves to decide if more data, more features, etc. are likely to help.
3. `Error Analysis` on CV set -- Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516347837584.png)
4. `Single numerical evaluation` to test whether an improvement approach works or not, such as accuracy
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516348007010.png)

### Handling Skewed Data

**Skewed Data:** positive and negative samples are extremely imbalance, like 99.5% w/o cancer vs 0.5% with cancer

**Error Metrics Analysis for Skewed Classes:**
- Set y=1 as rare class, and if both precision and recall are high, then the model performs well
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516349596835.png)

**Precision-Recall Trade Off:** using F score instead
- By tuning threshold of logistic regression $$h_\Theta(x) \ge \text{threshold}$$, precision and recall has reverse correlation
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516350887616.png)
- `F Score` ($$F_1 \ \text{Score}$$)takes both precision and recall into consideration as $$2 \frac{PR}{P+R}$$

### Using Large Data Sets

**When does more data help?**
- *Step 1:* (`Human Performance Level`) Given the input $$X$$, can a human expert confidently predict $$y$$?
	- If YES --> go to *Step 2*
	- If NO --> $$X$$ has not enough information for prediction --> add in more features
- *Step 2:* If the model is complex enough to achieve low training error (low bias)?
	- If YES --> more data will help (it is hard to overfit very large dataset)
	- If NO --> add in more parameters

## Week 7: SVM

### Large Margin Classification

**From logistic regression to SVM:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516498715220.png)

- Compared to LR:
	- SVM use hinge loss as activation function `in cost function`, which considers margin between 2 groups, instead of considering the `probability difference` between prediction $$h_\theta(x)$$ and true label $$y$$ in LR
	- Hypothesis of SVM uses discriminant function, while that of LR uses sigmoid function with specific threshold
$$h_\theta(x) =\begin{cases}    1 & \text{if} \ \Theta^Tx \geq 0 \\    0 & \text{otherwise}\end{cases}$$

- If C is very large, then algorithm tends to choose theta that the cost part equal to zero, and leave the regularization part only
	- If we have outlier examples that we don't want to affect the decision boundary, then we can reduce C

**SVM controls the decision margin between classes:**
- From the definition of vector inner product, $$\theta^Tx^{(i)} = |\vec \theta|\cdot|\vec {x^{(i)}}|\cdot cos$$ represents the projection of vector $$\vec {x^{(i)}}$$ along the vector $$\vec \theta$$, times the length of vector $$\vec \theta$$
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516501008482.png)
- Therefore, by SVM cost function, $$\theta$$ is trained to guarantee the projection of all samples upon vector $$\vec \theta$$ is at least 1 or -1, with shortest $$\vec \theta$$ (regularization term)
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516501443270.png)

### Kernels

> Use SVM as non-linear classifier with kernel tricks

**Gaussian kernel:** combine all features into relative distance between samples
- Replaces all features with $$m$$ distance between any pair of samples (using each sample as a landmark), where the distance is normalized by Gaussian distribution
$$f = similarity(x, l) = \exp(-\dfrac{||x - l||^2}{2\sigma^2})$$
$$x^{(i)} \rightarrow \begin{bmatrix}f_1^{(i)} = similarity(x^{(i)}, l^{(1)}) \newline f_2^{(i)} = similarity(x^{(i)}, l^{(2)}) \newline\vdots \newline f_m^{(i)} = similarity(x^{(i)}, l^{(m)}) \newline\end{bmatrix}$$
- Hyperparameters: $$\sigma^2$$ the higher the further sample prone to be in a similar group of this landmark
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516505558251.png)

### SVMs in Practice

**LR or SVM:**
- $$n$$ larger than $$m$$ (10k vs 10~1k)
	- LR or SVM w/o kernel
- $$n$$ small, $$m$$ intermediate (1~1k vs 10~10k)
	- SVM with Gaussian kernel
- $$n$$ small, $$m$$ large (1~1k vs 50k+)
	- add more features, then LR or SVM w/o kernel

**Hyperparameters:**
- $$C = \frac{1}{\lambda}$$: larger C, less regularization
- **Kernel (similarity function):**
	- **No kernel (linear kernel):** when $$n$$ is large, $$m$$ is small
	- **Gaussian kernel:** when $$n$$ is small, $$m$$ is large
		- $$\sigma^2$$: larger $$\sigma$$, feature $$f_i$$ vary more smoothly --> higher bias, lower variance
		- need feature scaling before performing kernel transformation
	- Others rarely used

**Multi-class classification:** one-vs-all, training K classifier for K classes

## Week 8: Unsupervised Learning

### Clustering 

**K-means:**
- Cost function $$J(c^{(i)},\dots,c^{(m)},\mu_1,\dots,\mu_K) = \dfrac{1}{m}\sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$
	- also called the distortion of the training examples

``` matlab
% Minimize J by assigning c(1)..c(m), holding mu(1)..mu(K) fixed
for i = 1 to m:
      c(i):= index (from 1 to K) of cluster centroid closest to x(i)

% Minimize J by moving centroid mu(1)..mu(K)
for k = 1 to K:
      mu(k):= average (mean) of points assigned to cluster k
```

- 1st loop: $$c^{(i)} = argmin_k\ ||x^{(i)} - \mu_k||^2$$
- 2nd loop: $$\mu_k = \dfrac{1}{n}[x^{(k_1)} + x^{(k_2)} + \dots + x^{(k_n)}] \in \mathbb{R}^n$$

**Random initialization:** 50-1000 iterations to avoid local optima
- Randomly pick $$K<m$$ distinct training examples
- Set $$\mu_1,\dots,\mu_K$$ equal to these $$K$$ examples

**Choosing number of clusters $$K$$:**
- Follow the application purpose of running clustering
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516674482631.png)
- Elbow method (less used)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516674550032.png)

**Application:**
- Compress image from 24-bit to 4-bit
	- find out the major 16 colors in a image (centroids)
	- store these colors with 24*16 bit
	- replace color of each pixel with corresponding centroid color 1~16 (4-bit)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516765068277.png)

**Bonus:** [Drawbacks of K-Means](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)
- k-means assumes the variance of the distribution of each attribute (variable) is spherical
- all variables have the same variance
- the prior probability for all k clusters is the same, i.e., each cluster has roughly equal number of observations

### PCA

> Try model without PCA first, and use PCA only when pre-train model cannot satisfy you

**Motivation:**
- Data Compression
	- combine highly correlated variables
	- remove features with less ability of distinguish samples
- Visualization

**Optimization target:** find a direction onto which to project the data so as to minimize the projection error $$\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x^{(i)}_\text{approx}\|^2$$
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516715214198.png)

**Procedure of PCA:**
1. Feature scaling with zero mean and comparable scale, for example, $$x_j^{(i)} = \dfrac{x_j^{(i)} - \mu_j}{s_j}$$
2. Compute covariance matrix `Sigma = (1/m) * X' * X`
3. Compute eigenvectors of covariance matrix `[U,S,V] = svd(Sigma)`
4. Take the first k columns of U matrix `Ureduce = U(:,1:k)`
5. Compute Z `Z = X * Ureduce`

**Choosing K:** how much variance is retained
$$\dfrac{\dfrac{1}{m}\sum^m_{i=1}||x^{(i)} - x_{approx}^{(i)}||^2}{\dfrac{1}{m}\sum^m_{i=1}||x^{(i)}||^2} \leq 0.01 \ \text{or} \ \dfrac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^nS_{ii}} \geq 0.99$$
where $$S_{ii}$$ is from diagonal matrix `S` in `[U,S,V] = svd(Sigma)`

**Applications:**
- Reconstruction from compressed representation
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516753636234.png)
- Speedup supervised learning
	- PCA should be defined by training set, and then apply to validation and test set
	- useful for image dataset

> Do NOT use PCA to prevent overfitting
> - Overfitting is a problem of model, not of dataset
> - Regularization is the way to improve model, and it takes labels into account
> - PCA necessarily remove some information from dataset, and it only considers features, dropping out labels

## Week 9

### Anomaly Detection (Semi-supervised)

The ET drift annotation system in GF is a kind of anomaly detection, which uses last 50 samples as training set of a single feature, and highlights anomaly if that new sample is out of 3 sigma (probability density threshold as 3 sigma). Precisely, this detection is a unsupervised version because it uses unlabeled data as training set. 

> Supervised vs. Semi-supervised
> - Supervised learning uses large number of both positive and negative samples
>    - algorithm is able to get a sense of what positive samples look like
>    - the future positive samples are similar to the ones in training set
> - Semi-supervised learning uses very small number of positive samples with large number of negative samples
>    - it is impossible for algorithm to learn from positive samples about what positive samples look like
>    - the future anomalies may look nothing like any samples in training set

**Target:** return the probability density from Gaussian distribution, where $$p(x) = \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2) \le \varepsilon$$ indicates $$x$$ is anomaly $$y=1$$

**Train / Validation / Test split:**
- Full set example: 10k normal samples + 20 anomalous samples
	- Train set: 6k normal
	- Validation set: 2k normal + 10 anomalous
	- Test set: 2k normal + 10 anomalous

**Procedure and Metrics:**
1. Fit parameters with train set $$\mu_1,\dots,\mu_n,\sigma_1^2,\dots,\sigma_n^2$$
2. Calculate $$\mu_j$$ and $$\sigma^2_j$$
3. Using validation set, calculate $$p(x) = \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2) = \prod\limits^n_{j=1} \dfrac{1}{\sqrt{2\pi}\sigma_j}exp(-\dfrac{(x_j - \mu_j)^2}{2\sigma^2_j})$$
4. Use Precision/Recall/$$F_1$$ score to evaluate model and tune $$\varepsilon$$
5. Using test set, evaluate generalization ability

**Feature Engineering:** for model improvement
- Transform features to Gaussian distribution, by using $$log(x+c)$$, $$\sqrt{x}$$, $$x^{1/3}$$ etc.
- Manually choose / add / create features that might take on unusually large or small values in the event of an anomaly (could be based on your understanding of the application)
	- Otherwise, use multivariate Gaussian distribution version
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516801213223.png)

**Anomaly Detection with Multivariate Gaussian Distribution:**
- Original AD treats each features as `independent`, modeling them separately, which means only a feature with extreme low probability density, then the sample is detected as anomaly
	- Graphically, original algorithm (pink circles) generate ellipse with both axis paralleling to the direction of features, while multivariate version (blue circles) covers more general cases by generating ellipse with rotated axis, considering `correlation between features`
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516845153919.png)
- Instead of original algorithm, $$p(x;\mu,\Sigma) = \dfrac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} exp(-1/2(x-\mu)^T\Sigma^{-1}(x-\mu))$$

**Comparison between Two Algorithm:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516960721200.png)

### Recommendation

> Special category of ML: feature learning (automatically select good features)

**Content Based Recommendation:** w/o feature learning because features are known already
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1516970355937.png)

**Collaborative Filtering Recommendation:**
> Repeat learning features (contents) from all user rates, and then learning parameters (user preference) based on contents.
>> All user helps to learn movie features, and then by these features, algorithm predicts better result for every user.
>> However, the feature learnt is hard to interpret but can be used to find the related movies by calculate $$||x^{(i)}-x^{(j)}||$$

- Given $$x^{(1)},...,x^{(n_m)}$$, estimate $$\theta^{(1)},...,\theta^{(n_u)}$$
$$min_{\theta^{(1)},\dots,\theta^{(n_u)}} = \dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u}  \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2$$
- Given $$\theta^{(1)},...,\theta^{(n_u)}$$, estimate $$x^{(1)},...,x^{(n_m)}$$
$$min_{x^{(1)},\dots,x^{(n_m)}} \dfrac{1}{2} \displaystyle \sum_{i=1}^{n_m}  \sum_{j:r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$$
- Combine both and run simultaneously (initial random values + gradient descent)
$$J(x,\theta) = \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} (\theta_k^{(j)})^2$$

**Vectorization with Low Rank Matrix Factorization:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1517049502383.png)

**Mean Normalization for User w/o Any Rating:**
- Normalize rates with their mean before running algorithm
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1517056406628.png)

## Week 10: Large Scale Machine Learning

### Gradient Descent with Large Datasets

> With larger m trained by a model, the variance will be smaller.
> But when m is extreme large, like 100M, each epoch of gradient descent is computational expensive due to calculating and averaging over 100M values.

**Stochastic Gradient Descent:** 1 sample per update
1. Randomly "shuffle" the dataset
2. Iterate $$i=1\dots m$$, $$\Theta_j := \Theta_j - \alpha (h_{\Theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}_j$$
3. Repeat above 1-10 times (epoch), resulting NEAR global minimum

**Mini-batch Gradient Descent:** $$b=2\sim 100$$ samples per update (fastest with vectorization)
1. Randomly "shuffle" the dataset
2. Set $$b=10$$, iterate $$i = 1,11,21,31,\dots,991$$, $$\theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) - y^{(k)})x_j^{(k)}$$
3. Repeat above 1-10 times (epoch)

**Gradient Check:** for Stochastic GD
- With every $$i$$ iterations, plot the average cost of last $$i$$ samples with current $$\theta$$
- $$i$$ normally sets as 500-2000

**Choosing Learning Rate:** normally keep constant, otherwise can decrease over iterations

### Online Learning

> Handle unlimited continuous stream data, so only perform train-and-drop way of training, where no need to re-use the data. And this method can adapt to the changing of user preference due to forever training with fresh data.

**Predicted Click Through Rate (CTR):** return 10 results in 100 phones when user search
- $$x = $$ features of phone, how many words in user query match name of phone, how many words in query match description of phone, etc.
- $$y=1$$ is click on link.
- Generate 10 samples per search action.
- Learn $$p(y=1|x;\theta)$$.
- Drop these 10 samples and continue gradient descent with new 10 samples

### Map Reduce and Data Parallelism

> Great technique to handle even larger machine learning problems: split jobs onto more than one core or computer

**MapReduceable:** the model can be expressed as computing sums of functions over the training set
- Linear regression / logistic regression
- Neural Network (each core computes the forward and backward propagation of a subset, then combine)

## Week 11: Case Study -- Photo OCR

### Pipeline of Photo OCR

> Divide a large project into stages and assign manpower corresponding to the workload

1. Text detection (2D sliding windows)
2. Character segmentation (1D sliding windows for character split)
3. Character classification (LR or SVM)

### Artificial Data Synthesis

**Artificially Create More Data:** Distortion of Data
- Should represent types of noise / distortion in test set or real applications
	- background noise
	- bad cellphone connection
- Purely random / meaningless noise has no help

**How much work would it be to get 10x as much data as we currently have?**
- Quantitatively understand the effort of getting more data, and then make better decision
- Approaches:
	- Artificial data synthesis
	- Collect/label it yourself
	- "Crowd Source" like Amazon Mechanical Turk

### Ceiling Analysis for Pipeline

> Prioritize your work in a pipeline

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1517218598529.png)

----
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-29_1517220172327.png)
