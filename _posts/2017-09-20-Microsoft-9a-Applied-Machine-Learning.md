---
layout: post
title: Applied Machine Learning - Microsoft Certificate in Data Science 9a
key: 20170928
tags:
  - Microsoft
  - Notes
  - Study
  - Machine Learning
  - Data Science
lang: en
mathjax: true
mathjax_autoNumber: true
---

## 1. Time Series and Forecasting

### Introduction to Time Series

Finance / stock / currency exchange rate / sales forecast / temperature / heartrate / Semicon ET and inline long-term trend / ...

#### The Nature of Time Series Data

Time Series vs. Random or Independent Noise
![@Time Series vs. Random or Independent Noise](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504050649521.png){:.border}

**Autocorrelation**: value at $$t=0$$ has correlation with the value at following $$t$$

Autocorrelation                                
![@Autocorrelation](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504050427726.png){:.border}

**Regular Reporting**: Some algorithms can only work with regular reporting

Regular vs. Irregular Reporting
![@Regular vs. Irregular Reporting](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504050906721.png){:.border}

#### Decomposition -- STL Package (Investigating)

![Components of Time Series Data](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504100756257.png){:.border}

##### STL Package Procedure

1. Start with time series data $$X$$
2. Use Loess to find a general trend $$T$$
3. Use Moving Average Smoothing $$X-T$$ to find fine-grained trend $$C$$
4. Get seasonal / periodic component $$S = X-T-C$$
5. Get final trend $$V$$ by smoothing the nonseasonal trend $$X-S$$ with Loess
6. Get remainder $$R = X-S-V$$

##### Lowess / Loess Regression

General trend $$T$$ for smoothing time series data

> **Idea:** fit local polynomial models and merge them together
> *local* for flexible
> *polynomial* for smooth

**Step 1:** Define the window width m, and do **local** regression with m nearest neighbors
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504102705116.png){:.border}

**Step 2:** Choose a weight function giving higher weights to nearer points to center
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504102692688.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504102609678.png){:.border}

**Step 3:** Do quadratic Taylor's polynomial regression considering the weights from Step 2
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504103390681.png){:.border}

**Step 4:** Substitute $$x_{0}$$ with $$\widehat{x_{0}}$$, which is calculated from regression when $$t_{i}=t_{0}$$

**Step 5:** Repeat above for each $$\widehat{x}$$ of $$t$$, then connect points to get the general trend

![Adjusting Window](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504105995941.png){:.border}

##### Moving Average Smoothing

Fine-grained trend $$C$$ for smoothing time series data with **clearly periodicity** (after extract general trend)

![Procedure of Moving Average Smoothing](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504233307082.png){:.border}

#### Stationary Remainder / Time Series

Second-order stationarity conditions: 
1. Constant Mean
2. Constant Variance
3. An autocovariance that does not depend on time

**Technique 1:** Boxplots with binned data point into upper hierarchy
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504234520365.png){:.border}

**Technique 2:** Boxplots with binned data point into upper hierarchy
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504234648362.png){:.border}

#### Autocorrelation and Autocovariance

>Same as Correlation (normalized Covariance) used to describe (linear) relationship between Feature X and Feature Y

Auto = self
ACF = Autocorrelation Function

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504237222122.png){:.border}

### Working with Time Series

Introduction of models for modeling different types of time series data so that we can do forecasting

> Remainder can also have time series pattern which need to be carefully modeled and removed, then the left residue (prediction error) should be normal distributed along time
>> Noticed successful STL should have below appearance
>> 1. histogram of remainder is close to normal distribution
>> 2. boxplot of remainder at seasonal level (like month) is stable

#### Moving Average Models MA(q)

> Microsoft announces one news everyday, and its stock will be affected by today's and last 2-days news
> ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504324039241.png){:.border}

A model has only short memory of the previous noise
![Moving Average Model](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504324300066.png){:.border}

**ACF:** sharp cut off after order q; can identify whether you data can be modeled as $$MA(q)$$ with what order $$q$$
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504326187335.png){:.border}

#### Autoregression Models AR(p)

Today's value is slightly different from a combination of the last $$p$$ day's values
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504343835929.png){:.border}

**ACF:** Exponential decay; can not identify order $$p$$

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504344131782.png){:.border}

##### Partial Autocorrelation

>The correlation that is not accounted for all of the lags in between

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504345731662.png){:.border}

![@Comparison between ACF and PACF of AR(1)](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504355947462.png){:.border}

#### Auto-Regressive Moving Average Model ARMA(p,q)

Used when both ACF and PACF shows slow decay

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504356507431.png){:.border}

#### Auto-Regressive Integrated Moving Average Model ARIMA(d,p,q)

##### Differencing

>Non-stationary time series can have stationary differences

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504358325135.png){:.border}

Higher order trends can be turned into stationary models through repeated differencing 
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504358681370.png){:.border}

##### Model Details

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504359151123.png){:.border}

#### Exponentially Weighted Moving Averages Model EWMA / Simple Exponential Smoothing Model SES

Most widely used for business applications / forecasting

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504370455425.png){:.border}

### Forecasting in Context

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504419046425.png){:.border}

### Reference

[Time Series Analysis (TSA) in Python - Linear Models to GARCH](http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016)

## 2. Spatial Data Analysis

Mobile marketing / smart watch data / oil exploration / real estate pricing / transportation network / crimes data ...

### Introduction to Spatial Data

Types of Spatial Data
- Points (location only)
- Polygons
- Pixels / Raster (location + count/density shown as colors)

Types of Distance
- Euclidean distance (physical distance; use built-in tool to calculate since earth is round)
- Driving / Walking distance
- Adapted to the local area (like same building)
Distance Matrix
![@Distance Matrix](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504622881694.png){:.border}

Visualize relationship of different features and overlay multiple features in one plot by various way like bubble size or filled color

#### Kernel Density Estimation KDE

- Go-to method for density / event rate $$\lambda$$ estimation
- "Nonparametric", meaning that there is a bump on each point

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504655253027.png){:.border}

#### K-Nearest Neighbour

> **Localized** technique of probability estimation

- Classification by majority vote
- Regression by average vote
- Take care
	- scale sensitive: consider normalization
	- selection of K and weight of distance

### Working with Spatial Data

#### Spatial Poisson Processes

> Probability estimation of occurrence count in an area in a period, based on Poisson distribution, which is a discrete probability distribution

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504704710467.png){:.border}

#### Variogram

> Estimate the (label) covariance between samples with spatial changes in units, which is just like the ACF and PACF for time series
> Input data is labeled
> Consider overall data in dataset
> ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504741135367.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505046512171.png){:.border}

- Reference
	- [Semi-Variogram: Nugget, Range and Sill](http://gisgeography.com/semi-variogram-nugget-range-sill/)
	- [Estimation and Modeling of Spatial Correlations](http://www4.ncsu.edu/~ykao/docs/Lab%203/Estimation%20and%20Modeling%20of%20Sptial%20Correlation.pdf) (about the second-order stationary assumption)

#### Kriging / Gaussian Process / Spatial Regression

> **Overall** technique of probability estimation based on Variogram providing the covariance k
>> k can be modelled by arbitrary covariance function in Variogram stage

> Interpolation method for estimating the property of unsampled location, so can get the complete map

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504971854754.png){:.border}

### Spatial Data in Context

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1504974993741.png){:.border}

## 3. Text Analytics

Summary of text / compare between text or classification 

### Introduction to Text Analytics

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505087678525.png){:.border}

#### Word Frequency

- Frequency plot
- Cumulative plot to examine the cleaned up dataset

#### Stemming

Only for English

> connection, connected, connective, connecting --> connect

- Porter's Algorithm
	- V is one or more vowels (A, E, I, O, U)
	- C is one or more consonants
	- All words are of the following form
		- [C]VC{m}[V], optional in brackets and stack times in parentheses
	- For each words, we check whether it obeys a condition, and shorten or lengthen it accordingly

#### Feature Hashing (Dimensionality Reduction)

Fast and space-efficient way of vectorizing features, by applying a hash function to the features and using their hash values as indices directly.

Also called hashing (kernel) trick.

[Wiki: Feature hashing](https://en.wikipedia.org/wiki/Feature_hashing)

### Working with Text

#### Calculating Word Importance by TF-IDF

> TF = Term Frequency (the number of times you see a word)
> IDF = Inverse Document Frequency
> $$TF\cdot log(\frac{\#Documents}{\#Number\ of\ Documents\ Word\ Appears})$$

TF-IDF is the key factor used in search engines

- TF-IDF is high when
	- the term appears many times in few documents
- TF-IDF is low when
	- the term appears in almost all documents
	- the term does not appear often

#### Introduction to Natural Language Processing

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505229757541.png){:.border}

### Text Analytics in Context

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505259624308.png){:.border}

## 4. Image Analysis

Photographs / Security cameras / Check reader / Medical images / Art work analysis ...

### Introduction to Image Analysis

#### Read / Plot Image

- misc function from scipy (output a numpy array with rows and columns as the image size)
- imshow from matplotlib.pyplot
- glob.glob for multiple images reading
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505526258599.png){:.border}

#### Image Properties

- Examine the distribution of gray scale
	- Histogram (ideal image has nearly uniform distribution)
	- CDF (ideal image has a straight line)
- Adaptive Histogram Equalization to improve contrast
	- The histogram equalization algorithm attempts to adjust the pixel values in the image to create a more uniform distribution
	- [exposure.equalize_adapthist](http://scikit-image.org/docs/dev/api/skimage.exposure.html?highlight=exposure#skimage.exposure.equalize_adapthist) from skimage
		- Before and After Equalization
		- ![@Before and After Equalization](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505527870968.png){:.border}

#### Image Manipulation

- Resize by [misc.imresize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html) from scipy
- Rotate by [interpolation.rotate](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html) from scipy.ndimage

#### Blurring and Denoising

> Pre-whitening together with Denoising can improve the sobel edge detection result --> clearer edge
>> The reason may be that it covers and removes the unnecessary / meaningless portion of image, which also happens in time series analysis when doing cross-correlation function for two series case

- Pre-whitening to add noise
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505534646497.png){:.border}
- Denoising by gaussian_filter / median_filter from [scipy.ndimage.filters](https://docs.scipy.org/doc/scipy/reference/ndimage.html#filters)

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505533877786.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505532904685.png){:.border}

### Working with Images

#### Feature Extraction

##### Sobel Edge Detection

> Detecting edge by looking for single direction gradients within selected area
> Viola and Jones Method in Course 7
> ![@Viola and Jones Method in Course 7](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505648783364.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505650122952.png){:.border}

##### Segmentation

Remove noise or unwanted portion

Simplest way --> threshold (move out the points under or over threshold)

##### Harris Corner Detection

> Compute Q matrix in E function representing a ellipse, and detect a corner when Q has 2 large eigenvalues which illustrates smaller principal axes of ellipse

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505658464478.png){:.border}

#### Introduction to Mathematical Morphology

##### Dilation and Erosion

> Fill or remove the center pixel of specific shape to do dilation and erosion

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505837315985.png){:.border}

##### Opening and Closing

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505692781412.png){:.border}

### Image Analysis In Context

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-09-20_1505777992871.png){:.border}
