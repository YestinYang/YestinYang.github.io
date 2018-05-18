---
layout: post
title: WeRateDogs @ Twitter 数据探索 
key: 20180518
tags:
  - Udacity
  - Data Analyst
  - Data Science
lang: en
---



# WeRateDogs @ Twitter 数据探索

该项目来源于Udacity Data Analyst Advanced课程的第二个项目，目标是对不同数据来源（既有数据、http数据和API json数据）进行收集评估清洗，通过可视化方法发掘数据有价值的信息。收集的数据包括推特的基本信息，以及利用神经网络针对推特图片进行的内容预测（预测图片中是什么品种的狗）。

对于数据可视化一些有意思的结果，特此在这里与大家分享。

## WeRateDogs发推数量变化

![](https://raw.githubusercontent.com/YestinYang/Learning-Path/master/Projects/WeRateDogs/Screen%20Shot%202018-05-08%20at%2021.29.40.png)

从图中可以看出2015年11月该推主开始经营该推特账号，第二个月的发推数量达到了顶峰，平均每天超过了12条原创推特（开荒阶段着实辛苦）。在此后逐渐下降，在2016年第二季度开始就趋于平稳，平均每天约2～3条，并一直维持。

## WeRateDogs推特热度变化

对比于上图，我们可以再来看看WeRateDogs是因为没有关注而发推减少，还是因为热度极高，不需要再依赖发推数量来吸引眼球了。

![](https://raw.githubusercontent.com/YestinYang/Learning-Path/master/Projects/WeRateDogs/Screen%20Shot%202018-05-08%20at%2021.50.50.png)

这里我使用每条推特的平均转发和点赞量作为衡量WeRateDogs热度的标准。从上图可以看出，转发量呈现线性增长，而点赞量呈现指数性增长。更加细节的一点在于，推主从2016年第二季度开始减少了发推数量，但热度的提升确实在第三季度才出现—— 这说明推主减少发推数量并非由于热度提高。

## 最受欢迎的15种狗狗

![](https://raw.githubusercontent.com/YestinYang/Learning-Path/master/Projects/WeRateDogs/Screen%20Shot%202018-05-08%20at%2021.29.58.png)

此处依然使用了每条推特的平均转发和点赞量作为衡量受欢迎程度的标准。从上图可以看出，前15名的犬种在转发数量上没有太大的差距，差距主要体现在点赞数量上。萨路基猎犬和贝灵顿梗是最受欢迎的两个犬种，这两种狗在中国都很少见，依次来一波图感受一下。

![萨路基猎犬](http://img.58cdn.com.cn/ds/ershou/hangqingImg/3-150415103Z7.jpg)

![贝灵顿梗](http://img1.goumin.com/cms/petschool/day_151021/20151021_6ee2d7d.jpg)

