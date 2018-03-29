---
layout: post
title: Microsoft Certificate in Data Science -- Course 8b Programming with Python for Data Science
key: 20170828
tags:
  - Microsoft
  - Notes
  - Study
  - Python
  - Data Science
lang: en
---



![@Structure of This Course](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1502668708572.png){:.border}

## 1. The Big Picture

- Data Science and Analysis
	- Keep your goal in mind and be careful when collecting data
		- Good data is the more important than algorithm itself (TensorFlow by Google)

- Machine Learning
	- Data has pattern, so we can learn from it (A leads B, like poking balloon --> scared)
	- Data-driven question before starting ML
		- If easy through simple means --> no need ML
		- If very complex, or multiple tuning parameters --> ML

- Algorithm is generic, compared to hard coding

- Unsupervised learning attempts to extract patterns; supervised learning tries to fit rules and equations

- The Possibilities of ML
	- Classification / Regression / Clustering / Dimensionality reduction / Reinforcement learning (how to play a video game)

- [Reinforcement Learning](evernote:///view/8001933/s33/6ee01abf-fffe-4364-beb2-f8b1e649cb83/6ee01abf-fffe-4364-beb2-f8b1e649cb83/)

## 2. Data and Features

Collect as many samples and features as you can, and use your intuitive to understand your question so can choose best feature structures

- ![Data Structure](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1502724390739.png){:.border}

- ![Types of Features](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1502724434736.png){:.border}

### Determining Features

- Adding additional features or collecting more samples?
	- depend on your data-driven question
	- bottom limit -- higher dimensions of sample than of feature

- Should assume certain feature is more important?
	- weak features can combine as a strong feature
	- machine learning can define the importance by itself, sometimes even find out the hidden relationship

- Garbage In, Garbage Out

### Manipulating Data

- Loading data
	- **pd.read_csv** and **pd.read_html**
		- *sep / delimiter / header / names / index_col / skipinitialspace / skiprows / na_values / thousands / decimal*
		- ![Axis Setting](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1502724512428.png){:.border}

- Quick peek of your data
	- **df.head(num) / df.describe() / df.columns / df.index / df.dtypes / df.unique() / df.value_counts()**

- Slicing
	- For column -- by name or index
		- Return series
			- **df.recency  ==  df['recency']  ==  df.loc[:, 'recency']  ==  df.iloc[:, 0]  ==  df.ix[:, 0]**
	- Return dataframe with >= 1 columns
		- **df[['recency']]  ==  df.loc[:, ['recency']]  ==  df.iloc[:, [0]]**
	- For row -- only by index
		- **df[0:2]  ==  df.iloc[0:2, :]**
		- iloc is not inclusive but loc and ix is inclusive
			- iloc[0:1] returns 1st row only, but loc[0:1] and ix[0:1] returns 1st and 2nd rows

- Dicing
	- Create Boolean index series
		- **df.recency < 7**
	- Using Boolean index series for slicing
		- **df[ df.recency < 7 ]**
		- **df[ (df.recency < 7) & (df.newbie == 0) ]**
		- **df[ (df.recency < 7) \| (df.newbie == 0) ]**
	- Replacing with Boolean
		- **df[df.recency < 7] = -100**
		- ![Example of Replacing with Boolean](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1502721957958.png){:.border}

### Feature Representation

Computer only speaks numbers

- Continuous data -- no worry
- Categorical data

```python
# for Ordinal data
>>> ordered_satisfaction = ['Very Unhappy', 'Unhappy', 'Neutral', 'Happy', 'Very Happy']
>>> df = pd.DataFrame({'satisfaction':['Mad', 'Happy', 'Unhappy', 'Neutral']})
>>> df.satisfaction = df.satisfaction.astpe("category",ordered=True, categories=ordered_satisfaction).cat.codes

>>> df
  satisfaction
0            -1
1            3
2            1
3            2

# for Nominal data
>>> df = pd.DataFrame({'vertebrates':['Bird', 'Bird', 'Mammal', 'Fish', 'Amphibian', 'Reptile', 'Mammal']})

# Method 1) for 1st test run of your model because simple and fast
>>> df['vertebrates'] = df.vertebrates.astype("category").cat.codes

>>> df
  vertebrates  vertebrates
0        Bird            1
1        Bird            1
2      Mammal            3
3        Fish            2
4   Amphibian            0
5     Reptile            4
6      Mammal            3

-----
# Method 2) for more precise
>>> df = pd.get_dummies(df,columns=['vertebrates'])

>>> df
  vertebrates_Amphibian  vertebrates_Bird  vertebrates_Fish  \
0                    0.0              1.0              0.0 
1                    0.0              1.0              0.0 
2                    0.0              0.0              0.0 
3                    0.0              0.0              1.0 
4                    1.0              0.0              0.0 
5                    0.0              0.0              0.0 
6                    0.0              0.0              0.0 

  vertebrates_Mammal  vertebrates_Reptile 
0                0.0                  0.0 
1                0.0                  0.0 
2                1.0                  0.0 
3                0.0                  0.0 
4                0.0                  0.0 
5                0.0                  1.0 
6                1.0                  0.0
```

- Pure Textual Features -- **CountVectorizer()** in SciKit-Learn (Bag-of-Words model)

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [ "Authman ran faster than Harry because he is an athlete.",  "Authman and Harry ran faster and faster." ]
>>> bow = CountVectorizer() # Define a analyzer with default setting
>>> X = bow.fit_transform(corpus) # Sparse Matrix storing result, which is the matrix with more 0 than 1 (from SciPy)
>>> bow.get_feature_names()
['an', 'and', 'athlete', 'authman', 'because', 'faster', 'harry', 'he', 'is', 'ran', 'than']
>>> X.toarray()
[[1 0 1 1 1 1 1 1 1 1 1] # 1st observation (sentence)
[0 2 0 1 0 2 1 0 0 1 0]] # 2nd observation (sentence)
```

- Graphical Features -- **misc** in SciPy

```python
from scipy import misc
img = misc.imread('image.png') # X by Y color image --> X rows, Y columns, 3 value per cell --> (X, Y, 3) NDArray
# Shrink the image down by resampling it
img = img[::2, ::2]
# Scale colors from (0-255) to (0-1), then reshape to 1D array per pixel, e.g. grayscale
X = (img / 255.0).reshape(-1) # ((X/2)*(Y/2), 1) NDArray
# If it is a color image and you want to preserve all color channels, use .reshape(-1,3)
X = (img / 255.0).reshape(-1,3) # ((X/2)*(Y/2), 3) NDArray
# Covert color image into gray scale if needed
red   = img[:,0]
green = img[:,1]
blue  = img[:,2]
gray = (0.299*red + 0.587*green + 0.114*blue)

-----
# Multiple images import and transform to Pandas dataframe for machine learning

# Load the image up
dset = []
for fname in files:
  img = misc.imread(fname)
  # resize to 1/2 and then reshape to one row as one sample
  dset.append( (img[::2, ::2] / 255.0).reshape(-1) ) 

dset = pd.DataFrame( dset ) 
```

- Audio Features -- **wavefile** in SciPy

```python
import scipy.io.wavfile as wavfile
sample_rate, audio_data = wavfile.read('sound.wav') # sample size must be identical
print audio_data
```

### Wrangling Your Data

- Fill missing data NaN (**np.nan**)

```python
df.fillna(0)  # fill all NaN with 0
df.my_feature.fillna( df.my_feature.mean() )  # fill all NaN in column 'my_feature' with the mean of 'my_feature'
df.fillna(method='ffill')  # forward fill (with last valid value)
df.fillna(method='bfill')  # back fill (with next valid value)
df.fillna(limit=5)  # maximum number of consecutive NaN filled
df.interpolate(method='polynomial', order=2)  # fill NaN with interpolate value calculated by different method (default: linear)
```

- Remove NaN

```python
df = df.dropna(axis=0)  # remove any row with NaNs
df = df.dropna(axis=1)  # remove any column with NaNs

df = df.dropna(axis=0, thresh=4)  # drop any row with more than 4 NaN values
```

- Remove duplicate rows / samples

```python
df = df.drop_duplicates(subset=['Feature_1', 'Feature_2'])
df = df.reset_index(drop=True)  # because original index is not continuous after removing duplicate; drop=True is to avoid Pandas keep a backup copy

# Chain methods for Dataframe wrangling
df= df.dropna(axis=0, thresh=2).drop(labels=['ColA', axis=1]).drop_duplicates(subset=['ColB', 'ColC']).reset_index(inplace=True)
```

- Data type conversion (especially for read_html which will set all columns as Pandas object)

```python
df.Date = pd.to_datetime(df.Date, errors='coerce')  # 'coerce' will replace invalid parsing with NaN 
df.Height = pd.to_numeric(df.Height, errors='coerce')
```

- More techniques are in [Dive Deeper][2]

## 3. Exploring Data

MatPlotLib as method, or MatPlotLib.pyplot as package

### Histogram

Knowing data distribution

```python
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot') # Look Pretty; If the above line throws an error, use plt.style.use('ggplot') instead

student_dataset = pd.read_csv("/Datasets/students.data", index_col=0)
my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)

plt.show()
```

![@G3](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503928707234.png){:.border} ![@G1+G2+G3](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503928710002.png){:.border}

### 2D Scatter Plots

```python
student_dataset = pd.read_csv("/Datasets/students.data", index_col=0)
student_dataset.plot.scatter(x='G1', y='G3')
```

### 3D Scatter Plots

Can only be created from MatPlotLib directly (not supported by Pandas dataframe)

```python
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
matplotlib.style.use('ggplot') 

student_dataset = pd.read_csv("Datasets/students.data", index_col=0)

# Plot setting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

# Select feature to plot
ax.scatter(student_dataset.G1, student_dataset.G3, student_dataset['Dalc'], c='r', marker='.')

plt.show()

# Save as .py file and run in terminal to utilize the interactive 3D rotating result
```

### Higher Dimensionality Visualizations

#### Parallel Coordinates

Each vertical axis is one dimension, for less than 10 features

- In MatPlotLib, unique y scale for all features --> consider normalization or log scale

![Parallel Coordinates](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503932447227.png){:.border}

```python
from sklearn.datasets import load_iris  # a classification sample dataset
from pandas.tools.plotting import parallel_coordinates

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target_names'] = [data.target_names[i] for i in data.target]  # name target (0,1,2) to corresponding name

# Parallel Coordinates Start Here
plt.figure()
parallel_coordinates(df, 'target_names')

plt.show()
```

#### Andrew's Curve

Each feature is set as the coefficient of a Fourier-series curve *(可理解为 feature vector 在 curve vector 上的映射)*

- Can easily detect outlier
- No normalization needed while defining groups

![@Andrew's Curve in Python](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503932214245.png){:.border}![@Andrew's Curve in Matlab Plotting Mean and Cut-off](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503932246680.png){:.border}

```python
from sklearn.datasets import load_iris
from pandas.tools.plotting import andrews_curves

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target_names'] = [data.target_names[i] for i in data.target]

# Andrews Curves Start Here
plt.figure()
andrews_curves(df, 'target_names')

plt.show()
```

#### Imshow

Visualize covariance matrix for correlation coefficient

![Matrix Plot by Imshow](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503932895621.png){:.border}

```python
df = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
df.corr()

# output covariance matrix showing correlation coefficient between every two features
          a        b        c        d        e
a  1.000000  0.007568  0.014746  0.027275 -0.029043
b  0.007568  1.000000 -0.039130 -0.011612  0.082062
c  0.014746 -0.039130  1.000000  0.025330 -0.028471
d  0.027275 -0.011612  0.025330  1.000000 -0.002215
e -0.029043  0.082062 -0.028471 -0.002215  1.000000

import matplotlib.pyplot as plt

plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')  # command for plotting array as image
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)

plt.show()
```

## 4. Transforming Data

Create a new space includes only the feature with the most distinguish power based on relationship among samples

![@Best View](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1504020663963.png){:.border}![@Top View - less effective](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1504020683800.png){:.border}![@Post Transforming](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1504020702868.png){:.border}

- We collect as much data as we can while doing data collection step
- But it will include a lot of 'duplicate' information because we may examine the object in similar view angle
- Transformation is to reduce redundant information and build up better features for machine learning

### Principle Component Analysis (PCA)

Linear unsupervised dimensionality reduction algorithm

> Calculate the best "view angles" for observation
> "view" = feature under machine learning context

- Linear transformation of the old feature space -- convert possibly correlated features into a set of linearly uncorrelated ones
	- start from center
	- 1st direction towards widest spread of values ***-- assuming more variance, more important the feature is***
	- 2nd direction orthogonal to 1st, and towards widest spread of values
	- 3rd, 4th ... drop the least important feature listed in the end of the space to reduce dimensionality
	- transform old coordinate into new orthogonal space -- projection only on principle component/axis/view/feature
	- PS: in actual calculation, it is done by covariance and eigenvector calculation

![@How PCA Works](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1504020951651.png){:.border}

- Limitation
	- Linear transformation only
	- Sensitive to scale --> need to standardize data before PCA using **only Z-score** (StandardScaler), because assumption is that greatest variance direction is the most important direction
	- Slow for very large datasets --> use randomized PCA by **svd_solver = 'randomized'**

```python
>>> from sklearn.decomposition import PCA
>>> pca = PCA(n_components=2, svd_solver='full')  # create a PCA instance for training later; n_components is dimension to keep

>>> pca.fit(df)  # train your PCA with dataset
PCA(copy=True, n_components=2, whiten=False)

>>> T = pca.transform(df)  # transform original dataset with newly trained PCA model

>>> df.shape
(430, 6) # 430 Student survey responses, 6 questions
>>> T.shape
(430, 2) # 430 Student survey responses, 2 principal components
```

```python
# Code for projecting the original feature axis on the new principle space, to understand the change of PCA

def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax
```
![Output of Projecting Code](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1504021527949.png){:.border}

```python
# Preprocessing with StandardScaler and add in column names

def scaleFeatures(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)  # fit StandardScaler model (calculate mean and sigma) then transform original dataframe into a new NDArray without column names
    scaled = pd.DataFrame(scaled, columns=df.columns)  # transform NDArray into DataFrame with column names

    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled
```

### Isomap

Non-linear unsupervised dimensionality reduction algorithm

- Usage
	- Whichever PCA does not work well
	- bending or curve
	- naturally generated dataset or high level of correlation between samples
		- vision recognition problems / hand written digits / identifying similar objects / speech problems / tracking movement of an object
		- for 3D rotating chair, if every pixel is a feature, at the end of the day, the manifold surface is parametrizable by just the angle of the chair --> a single feature! -- *Because the difference between samples is only angle*

- Essentially a node distance map that has been fed into a special type of PCA -- *the direction with greatest distance variance is the one has most important information*
	- calculate the distance from each sample to every other sample
	- only keep the K-nearest samples per sample in the neighborhood list --> like a map of  the cities
	- estimate the lower dimension embedded in dataset, also as manifold representation of dataset -- *as a special type of PCA targeting on distance*
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1506216538106.png){:.border}

```python
>>> from sklearn import manifold
>>> iso = manifold.Isomap(n_neighbors=4, n_components=2)  # neighborhood size can be 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, and 64 is almost close to PCA result
>>> iso.fit(df)
>>> manifold = iso.transform(df)

>>> df.shape
(430, 6)
>>> manifold.shape
(430, 2)
```
[Output with Various 'n-neighbor' Value](https://courses.edx.org/asset-v1:Microsoft+DAT210x+4T2016+type@asset+block@animation.mp4)

- Limitation
	- slower than PCA and irreversible
	- sensitive to scale -- Z-score standardize
	- sensitive to noise -- noise is like a short-circuit path which Isomap prefer

### Data Cleansing

- While gathering data, set proper process control and identify issues that might cause inconsistencies, and capture additional features that'll help you rectify them
- Retrospectively adjust your data to account for discovered problems
- If you encounter errors in your data, deleting the affected rows, or by cooking your data

## 5. Data Modeling I

### Splitting Data

To split out training and test dataset for scoring the model built up by training data

```python
>>> from sklearn.model_selection import train_test_split

>>> data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.5, random_state=7)
```

```python
# Model evaluation using test_data; can also use methods .score in respective ML model

from sklearn.metrics import accuracy_score

>>> predictions = my_model.predict(data_test)  # Returns an array of predictions
>>> predictions
[0, 0, 0, 1, 0]
>>> label_test  # The actual answers
[1, 1, 0, 0, 0]

>>> accuracy_score(label_test, predictions)
0.4000000000000000
>>> accuracy_score(label_test, predictions, normalize=False)
2
```

### Clustering

Unsupervised; K-Means as example

- **Input Types:** numerical features only, due to using Euclidean distance

- **Procedure:** minimizing within-cluster inertia -- within-cluster sum of squared errors
	- the lower overall inertia, the better cluster

- **Application:** focus on cluster or centroid
	- cluster --> grouping
	- centroid --> compress data from samples to centroid -- use centroid to represent samples
		- e.g. select the best location for service center

- **Limit:**
	- initial centroid has much effect on result
		- refer to *An Examination of Procedures for Determining the Number of Clusters in a Data Set* in [Microsoft Professional Program Certificate in Data Science](https://app.yinxiang.com/shard/s33/nl/8001933/9f04d712-23b6-469e-9813-4e395bf5c9ca)
	- scale sensitive by assuming samples are length normalized, or whose length encode a specific meaning
	- assuming cluster size is spherical and similar

[More clustering technique and performance](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py)

```python
>>> from sklearn.cluster import KMeans
>>> kmeans = KMeans(n_clusters=5)
>>> kmeans.fit(df)

>>> labels = kmeans.predict(df)
>>> centroids = kmeans.cluster_centers_
```

### Classification

Supervised; categorical output; K-Neighbor as example

- **Input Types:** numerical features only, including binary features

- **Procedure:**
	- just store training dataset together with label -- new labeled data can just top up without retrain the model
	- for each test sample, we get its nearest k neighbors, and do a mode vote to assign a label
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1506382982459.png){:.border}

- **Setting:** focus on cluster or centroid
	- **n_neighbors:**
		- higher K, less jittery decision boundaries
			- but less sensitive to local fluctuations and worse calculation burden
			- will worsen the result with imbalance group size -- such as 70% A and 30% B, result may be A but actually it is in group B
		- lower K, more sensitive to perturbation and local structure
		- set as odd for binary feature
	- **weights:** consider distance when doing voting -- if group size is imbalance, better use uniform
	- **algorithm:** method to find the nearest neighbors

- **Limit:**
	- calculation burden depends on training samples, not classes or groups
	- sensitive to feature scaling because based on distance
	- choose K value

```python
# Process:# Load a dataset into a dataframe
X = pd.read_csv('data.set', index_col=0)

# Do basic wrangling, but no transformations
# drop / to_numeric / fillna / dropna

# Immediately copy out the classification / label / class / answer column
y = X['classification'].copy()
X.drop(labels=['classification'], inplace=True, axis=1)

# Split data as necessary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7)

# Feature scaling and dimensionality reduction (fit X_train, then transform both X_train and X_test)
# preprocessing
# PCA or IsoMap

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3, weights='uniform')  # weights can be 'distance' for 1/d as weight
model.fit(X_train, y_train) 

# Evaluation
# Scoring different K
for i in range(15):
    knn = KNeighborsClassifier(n_neighbors=i+1, weights='uniform')
    knn.fit(X_train, y_train)
    print(str(i+1) + ":" + str(knn.score(X_test, y_test)))

# Plotting decision boundary
# refer to code in 'Module5-Lab7'

# TP/TN/FP/FN
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model.predict(X_test), labels= #label by sequence)
```
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1506383981004.png){:.border}

### Regression

Supervised; continuous output; linear regression as example

- Predict with continuous output by interpolation and extrapolation

- **Procedure:**
	- to find best fitting dimensional hyper plane
	- minimize the sum of square error / ordinary least squares
	- ![@Comparison between linear correlation (y~x and x~y) and PCA](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1506427993194.png){:.border}

- **Setting:**
	- data input
		- have to be selected subset of the most promising features -- more features results in lower R^2 value **BUT** errant overfitting
		- categorical data should be numerical encoded
	- **normalize:** integrated preprocessing function; only valid when fit_intercept = True
	- for multiple outputs: y_train and y_test set as [n_samples, n_targets]

- **Limit:**
	- Only for insight purpose when predicting outside the training data -- less accurate for extrapolation
		- the further you are predicting outside the training data, the more uncertainty there is
	- Linear assumption is only accurate on small and localized scales
		- assumption is that the result is completely linearly dependent upon the independent features
	- Samples are assumed as independent between each -- any feature of one sample has no effect or correlation on that of another sample
		- e.g. score of students in one class has linear correlation to -- may have certain additional correlation because of the effectiveness of teacher --> students are not independent samples
		- e.g. score of students in Class A and Class B has linear correlation to the effectiveness of teachers --> score of classes are independent samples
	- Only examine the the mean value between output and input

## 6. Data Modeling II

### SVM and SVC

- *Support Vector Classifier* -- compute the smallest change of features that can differentiate between classes
	- *Support Vector* -- coordinates of the sample closest to the boundary
	- Maximum the distance between frontier/hyperplane and closest sample
		- Calculate distance by projecting support vector on normal vector of the hyperplane
	- [One-class SVC][1] -- decide whether new testing data is the same class as training data

- *Kernel Trick / Table-flipping* -- nifty geometric shortcut for easy distance calculation just based on original coordinates, after applying kernel transformation
	- *Kernel* -- a similarity function
	- Using certain kernel to transform/map data from original space into higher-dimension space, in which it is easier for placing frontier

- Characteristic
	- Like decision tree, support non-linear decision-making by use of a linear decision surface, but by table-flipping
	- Only work on support vector, while the other samples far away from boundary will not be considered
		- Therefore, can work on dataset having more features than samples
	- Sensitive to scale / not scale invariant --> need preprocessing
	- May be sensitive to irrelevant features, while decision tree will not
	- Non-probabilistic, compared to decision tree
		- Only with cross validation can provide probability of SVC prediction
	- Consume time only at train stage, compared to K-neighbor

- Setting
	- **kernel**
	- **C** : penalty parameter; the higher value the more tendency of overfitting (decrease **C** when dataset has a lot of noise)
	- **gamma** : the higher value the more localized effect of each training sample; the lower value the larger area each sample affecting
	- **random_state** : if need deterministic execution, set a fixed seed
	- **class_weight** : set as 'balanced' when dataset is unbalanced
		- recall the example of minimizing the misclassification malignant tumor as benign

```python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X, y)
```


### Decision Tree

Intuitive and simulating the procedure of decision making in our life (example: 3 patients in emergency room and who should be treated first)

- Purify the decision by asking one YES/NO question per time
	- Considering one features per time -- one node
	- Splitting criteria of upper node is to minimize the impurity/entropy (or maximize information gain) in lower node
	- The more **unique** and **related** questions you ask, the more accurate you predict. Improper questions will render overfitting.
	- ![Example of Decision Tree](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503068527741.png){:.border}

- Characteristic
	- Like SVM, support non-linear decision-making by use of a linear decision surface, but by dividing up feature set into sections and boxes (add up multiple linear surfaces)
		- data need to be multivariate linearly separable
		- each node has a flat decision surface, and then can intersect others at angles
	- Unlike SVM, accuracy of a DTree will not decrease when dataset includes irrelevant features
	- Unlike PCA/IsoMap, DTree is insensitive / indifferent to monotonic feature scaling or transformations
		- because only one feature is considered per time
	- Work with both categorical and continuous features
	- Sensitive to small, local fluctuation / outlier in training set resulting in overfitting
		- use max setting to solve
		- Random Forest

- Setting
	- **criterion** : default is "gini" for Gini impurity, alternatively "entropy" for information gain
	- **splitter** : default is "best" allowing algorithm to find the best split among features
	- **max_features** : max number of features to consider when looking for the best split
	- **max_depth** / **min_sample_split** / **min_sample_leaf** : end-point controlling the overfitting
	- **class_weight** : false positive and false negative control
	- [attribute] **feature_importances_** : return array of features sorted by importance from high to low
	- [attribute] **predict_proba** / **predict_log_proba** : telling the probability of each class that an input may belong to

```python
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=9, criterion="entropy")
model.fit(X,y)

# .DOT files can be rendered to .PNGs through graphviz
# Windows 10 can output .DOT and use .exe installed by Graphviz.mis to view the tree
tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
```

### Random Forest

Solve the overfitting problem due to outlier in DTree, by creating a number of weaker DTrees which are working together and averaging the result for classification

- Procedure
	1. Randomly sample the training set, to get de-correlated subsets for training
		- *Tree Bagging* or *Bootstrap Aggregating*
		- each decision tree will be not correlated --> insensitive to outliers
	2. Training a decision tree HARD with the de-correlated subset
		- HARD = *Features Bagging*, to select a random sampling of features at every split of each individual decision tree
		- some features are high correlated to the 'y'-label, so we need *Features Bagging* to avoid over-examination of this kind of feature and increase the diversity of features examined
	3. Repeat step 1 & 2 until hit the desired number of iterations
	4. Make the final classification by taking the mode of the results

- Setting -- nearly interchangeable with DTree
	- **n_estimators** : density of the forest
	- **bootstrap** : training with tree bagging
	- **obb_score** : whether to score forest by using out-of-bag samples
		- *out-of-bag* : training samples not used for training a particular tree are considered out-of-bag for that particular tree
		- use a particular tree to predict its *out-of-bag* samples, and repeat this on every trees, then average the predictions from all trees to get the overall score of the forest 
	- [attribute] **.estimators_** : examining the structure of the individual decision trees

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, oob_score=True)
model.fit(X, y)
           
print model.oob_score_
0.789925345
```

- Characteristic
	- Unlike DTree, it is hard to be interpreted as IF...THEN blocks
	- Longer training time

## 7. Evaluating Data

### Confusion

- Choosing Right Estimator
	- [Advice from SciKit-Learn][3]
	- [Advice from AzureML][4]

- The Confusion Matrix
	- Comparing Predicted Value @ X-axis against Actual Value @ Y-axis
	- ![Confusion Matrix](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503497640055.png){:.border}
	- ![imshow of Confusion Matrix](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-08-28_1503497654640.png){:.border}
	
```python
>>> import sklearn.metrics as metrics
>>> y_true = [1, 1, 2, 2, 3, 3]  # Actual, observed testing dataset values
>>> y_pred = [1, 1, 1, 3, 2, 3]  # Predicted values from your model

>>> metrics.confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [1, 0, 1],
       [0, 1, 1]])

>>> import matplotlib.pyplot as plt
>>> columns = ['Cat', 'Dog', 'Monkey']
>>> confusion = metrics.confusion_matrix(y_true, y_pred)
>>> plt.imshow(confusion, cmap=plt.cm.Blues, interpolation='nearest')
>>> plt.xticks([0,1,2], columns, rotation='vertical')
>>> plt.yticks([0,1,2], columns)
>>> plt.colorbar()

>>> plt.show()
```

- Scoring Metrics
	- Evaluate the predictor while predicting each class
	- True Positive / True Negative / False Positive / False Negative

```python
# Accuracy
>>> metrics.accuracy_score(y_true, y_pred)
0.5
>>> metrics.accuracy_score(y_true, y_pred, normalize=False)
3

# Precision
>>> metrics.precision_score(y_true, y_pred, average='weighted')
0.38888888888888884
>>> metrics.precision_score(y_true, y_pred, average=None)
array([ 0.66666667,  0. ,  0.5])

# F1 score
>>> metrics.f1_score(y_true, y_pred, average='weighted')
0.43333333333333335
>>> metrics.f1_score(y_true, y_pred, average=None)
array([ 0.8,  0. ,  0.5])

# Full report
>>> target_names = ['Fruit 1', 'Fruit 2', 'Fruit 3']
>>> metrics.classification_report(y_true, y_pred, target_names=target_names)
precision    recall  f1-score   support
Fruit 1       0.67      1.00      0.80         2
Fruit 2       0.00      0.00      0.00         2
Fruit 3       0.50      0.50      0.50         2
avg / total       0.39      0.50      0.43         6
```

### Cross Validation

Overcome overfitting and fully utilize known data set

- Why?
	- Reason 1 -- one-time split results in untrustworthy accuracy score due to various quality of training/test set, like sampling bias
	- Reason 2 -- only based on training set will lose the information in testing set which can also be good material for training
	- Reason 3 -- tuning hyper-parameters for more configurable estimator with same 'train / test' split is actually stealing some information from testing set then put into training set

- Advantage / Disadvantage
	- Simplicity of coding process
	- Information leakage which still has possibility to happen even before fitting model, when tuning hyper-parameters at PCA / Isomap stage

- [Cross Validator Iterators][5]
	- For Independent and Identically Distributed (i.i.d.) data, which is stem from same generative process (same distribution)and this process has no memory of pass generated samples (idependent)
		- K-Fold / Repeated K-Fold / Leave One Out (LOO) / Leave P Out (LPO) / ShuffleSplit
	- For data with imbalance distribution of class, such as more negative samples than positive samples
		- Stratified K-Fold / Stratified Shuffle Split
	- For grouped data with groups of dependent samples, such as different subjects with several samples per-subject
		- Group K-Fold / Leave One Group Out / Leave P Groups Out / Group Shuffle Split
	- For time series data / autocorrelation data
		- Time Series Split

- Setting -- [Cross Validate][6]
	- **cv**: how to generate cross-validation set using different iterator

```python
>>> from sklearn import cross_validation as cval
>>> cval.cross_val_score(model, X_train, y_train, cv=10)
array([ 0.93513514,  0.99453552,  0.97237569,  0.98888889,  0.96089385,  0.98882682,  0.99441341,  0.98876404,  0.97175141,  0.96590909])

>>> cval.cross_val_score(model, X_train, y_train, cv=10).mean()
0.97614938602520218
```

Model + hyper-parameters + Cross-Validation

- **GridSearchCV**

```python
>>> from sklearn import svm, grid_search, datasets

>>> iris = datasets.load_iris()
>>> parameters = {
  'kernel':('linear', 'rbf'),
  'C':[1, 5, 10],
}
>>> model = svm.SVC()

>>> classifier = grid_search.GridSearchCV(model, parameters)
>>> classifier.fit(iris.data, iris.target)

	# classifier now stores the best result for .predict() / .score() etc. as normal
```

- **RandomizedSearchCV**

```python
	# Input a distribution for the parameter instead of a list of value (grid object) to try

>>> parameter_dist = {
  'C': scipy.stats.expon(scale=100), 
  'kernel': ['linear'],
  'gamma': scipy.stats.expon(scale=.1),
}

>>> classifier = grid_search.RandomizedSearchCV(model, parameter_dist)
>>> classifier.fit(iris.data, iris.target)
```

### Pipelining

Chaining estimators

- Scenario 1 -- Convenience and Encapsulation
	- for fixed sequence steps in processing the data
	- tune hyper-parameters of all estimators at one shot

```python
>>> from sklearn.pipeline import Pipeline

>>> svc = svm.SVC(kernel='linear')
>>> pca = RandomizedPCA()

>>> pipeline = Pipeline([
  ('pca', pca),
  ('svc', svc)
])
>>> pipeline.set_params(pca__n_components=5, svc__C=1, svc__gamma=0.0001)
>>> pipeline.fit(X, y)
```

- Scenario 2 -- Joint Parameter Selection by using Power Tuning
	- [Pipeline(PCA + LogisticRegression) + GridSearchCV][8]

- Safty -- help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors

- (Requirement) All estimators in a pipeline must have transform method, except the last one

```python
	# Script to add in transform method for a estimator without it
from sklearn.base import TransformerMixin

class ModelTransformer(TransformerMixin):
  def __init__(self, model):
    self.model = model

  def fit(self, *args, **kwargs):
    self.model.fit(*args, **kwargs)
    return self

  def transform(self, X, **transform_params):
    # This is the magic =)
    return DataFrame(self.model.predict(X))
```

### Best Two Processes of ML

Always reserve a seperate testing set to conduct final scoring until the study is complete, so that can avoid information leakage from test to train set ([Answers on CrossValidated][7])

- Without CV
	1. Split your data into training, validation, and testing sets.
	2. Setup a pipeline, and fit it with your training set
	3. Access the accuracy of its output using your validation set
	4. Fine tune this accuracy by adjusting the hyperparamters of your pipeline
	5. When you're comfortable with its accuracy, finally evaluate your pipeline with the testing set

- With CV
	1. Split your data into training and testing sets.
	2. Setup a pipeline with CV and fit / score it with your training set
	3. Fine tune this accuracy by adjusting the hyperparamters of your pipeline
	4. When you're comfortable with its accuracy, finally evaluate your pipeline with the testing set

----
## Reference

- [Overview of Normalization Technique][9]
- Document Clustering / Sparse Dataset (K-means / K-medoids / [Cosine Similarity][10])
	- [What and Why K-medoids][11]
	- [Why K-means cannot fit sparse dataset][12]
	- [Overview of Clustering][13]
- [Choosing preprocessing techniques][14]
	- StandardScaler -- assuming Gaussian distribution @ each feature
	- Normalizer -- see each sample as an vector, run l1 or l2 @ each sample
	- MinMaxScaler -- for non-Gaussian or tiny sigma @ each feature
- [Plotting Higher Dimensionality Boundaries][15] by finding keypoints of p~0.5 then projecting into 2D
- Audio Machine Learning [Voice Activity Detection in Python][16] / [Simple Minded Audio Classifier for Python][17]

----
[1]: https://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/)
[2]: https://courses.edx.org/courses/course-v1:Microsoft+DAT210x+3T2017/courseware/12621a4064aa4d92874a9d8a953734c5/e43e044ae9c045298766ece7d3881386/?child=first
[3]: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
[4]: https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-algorithm-choice
[5]: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
[6]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
[7]: https://stats.stackexchange.com/questions/20010/how-can-i-help-ensure-testing-data-does-not-leak-into-training-data
[8]: http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py
[9]: https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm
[10]: https://zh.wikipedia.org/wiki/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7
[11]: http://blog.pluskid.org/?p=40
[12]: https://stackoverflow.com/questions/12497252/how-can-i-cluster-document-using-k-means-flann-with-python
[13]: http://scikit-learn.sourceforge.net/dev/modules/clustering.html
[14]: https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm#_=_
[15]: https://github.com/tmadl/highdimensional-decision-boundary-plot
[16]: https://github.com/netankit/AudioMLProject1
[17]: https://github.com/danstowell/smacpy/blob/master/smacpy.py
















