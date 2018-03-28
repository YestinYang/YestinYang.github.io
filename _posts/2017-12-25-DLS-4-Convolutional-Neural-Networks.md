---
layout: post
title: Deep Learning Specialization 4 -- Convolutional Neural Networks
key: 20171225
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

## W1: Foundations of Convolutional Neural Networks

V1: computer vision problem
- types: classification / object detection / style transfer
- full-connected (FC) NN cannot handle high resolution pictures due to huge matrix after reshape an image as one dimension

### Convolution in DL
V2&V3: edge detection example
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512824960357.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512827143831.png){:.border}

V4: padding
- add zeros around the images
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512830120566.png){:.border}

V5: strided convolutions
> 'Convolution' used in ML is actually cross-correlation in math, which is without slipping the filter (response function)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512964015025.png){:.border}

V6: convolution over volume
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512978375479.png){:.border}

### Convolution Neural Network
V7: one layer of convolutional network
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512981133427.png){:.border}

V8: simple CNN example (only conv layer)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1512986349571.png){:.border}

V9: pooling layer
> Reduce the size of the representation / Speed up the computation / Make feature detection more robust

- even pooling has no parameters to be tuned, but it will affect the backpropagation calculation 
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513065304077.png){:.border}

V10: full CNN example
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513066496574.png){:.border}

### Why CNN works
V11: why CNN
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513067682808.png){:.border}


## W2: Deep Convolutional Models: Case Studies

> For engineering work, the most efficient way is to do case study and read literature to learn from other's CNN architecture, then apply on your own task

### Classic CNN
V2: classic networks
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513251533159.png){:.border}

### ResNets
V3&V4: ResNets and why (compared to plain networks)
> ResNet is to solve the problem of vanishing and exploding gradient in training very deep neural networks, and ResNet blocks with the shortcut makes it very easy for sandwiched blocks to learn an identity function (weight and bias)
>> However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values).
>> During training, you might therefore see the magnitude (or norm) of the gradient for the earlier layers descrease to zero very rapidly as training proceeds. (cited from coding exercise)![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513412417635.png){:.border}

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513254479327.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513254643613.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513494273499.png){:.border}

### Inception network
V5: 1x1 convolution
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513298890863.png){:.border}

V6&V7: inception network
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513301168940.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513303024777.png){:.border}

### Transfer learning from open-source implementation
V8: search open-source implementation in Github
- Starting from other's architecture is a common path of starting your own works
- Only reading paper is hard to replicate its architecture, so it is better to find out the shared implementations of that particular paper to start with
- Some open-source implementations also include pre-trained data, so that you can use it to do transfer learning making your progress even faster

V9: transfer learning
- almost always do transfer learning because it works very well on image recognition, but only do yourself training when you have extreme large dataset and enough computational budget
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513384859223.png){:.border}

### Data augmentation
 V10: data augmentation
 - commonly used in image recognition because it is always lack of data for this kind of task
 - the best way is still starting from other's implementation
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513386418113.png){:.border}

### State of computer vision and advises
V11: state of computer vision
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513391646039.png){:.border}

## W3: Object Detection

### Convolution Implementation of Sliding
V1: object localization
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513498611409.png){:.border}

V2: landmark detection
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513519065133.png){:.border}

V3: object detection with sliding windows
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513520975835.png){:.border}

V4: convolution **X** sliding windows
> the way of reducing computation cost of sliding with CNN
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513561804499.png){:.border}

### YOLO Algorithm
V5: bounding box prediction
> Utilize the idea of "FC-->convolution-->add sliding" **(each $1\times1\times c_{output}$ in output represents one portion of the whole image)**, but use grid instead of sliding, then add bounding box into label
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513562739330.png){:.border}

V6: intersection over union (IoU)
> to evaluate the performance of object detector
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513602640782.png){:.border}

V7: non-max suppression
> to make sure each object is detected only once
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513603998952.png){:.border}

V8: anchor boxes
> to solve the rare case that multiple objects are assigned to single grid
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513692580507.png){:.border}

V9: full YOLO algorithm
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513753509545.png){:.border}

### R-CNN
V10: R-CNN introduction (different from YOLO)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1513732294465.png){:.border}

## W4: Special Application: Face Recognition & Neural Style Transfer

### Face Recognition
V1: face verification and recognition
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514012685899.png){:.border}

V2: one-shot learning
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514013447708.png){:.border}

V3: Siamese network
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514014207759.png){:.border}

V4: triplet loss
> How to train the "encoding" network above (loss function and training set)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514036954861.png){:.border}

V5: alternative training method -- binary classification
> Instead of triplet loss, the Siamese network can also be trained as a binary classification
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514039482662.png){:.border}

### Neural Style Transfer

V1: neural style transfer
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514122432163.png){:.border}

V2: what is deep ConvNets learning
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514123004640.png){:.border}

V3&V4&V5: cost function
> Loss function can be defined to achieve the target, and it can grasp input from any layers of model (not just from last layer)
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514187490941.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514184684350.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514185566075.png){:.border}

### 1D and 3D Convolution
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-12-25_1514186727566.png){:.border}
