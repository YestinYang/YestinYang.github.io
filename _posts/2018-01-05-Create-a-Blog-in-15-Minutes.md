---
layout: post
title: 15分钟创建免费个人博客 @ GitHub Pages
key: 20180105
tags:
  - tutorial
lang: en
---

作为这个博客的第一篇文章，先写写我是如何创建这个博客的。与标题不同，我花了N倍于15分钟的时间来开启这个博客，而秉持着“解决核心问题，避免额外认知负担”的思路，最终采取了这一套简单稳定的方案。这也符合接触新事物时“粗浅--深入--精炼”的认知过程。

##  基本思路和准备条件

利用GitHub Pages项目博客生成系统，在GitHub Repository中建立必要的网站文件结构，最终通过StackEdit以Markdown语言撰写博文。

建立并使用整个博客，我们需要完成下列几项：

-  一个GitHub账号
-  从Jekyll Themes中选择喜欢的主题，并复制到自己的GitHub中
-  设置网站的基本信息
-  用StackEdit撰写博文

这里面提到了Markdown语言，后面会提供一些别人教程的链接。这是一种非常简单快捷的写作方式，推荐所有用电脑写文章的朋友用额外的10分钟学习，一定物超所值（从高效写作的角度完全超越Word）。

##  1. 建立GitHub账号

简单来说，GitHub Repository是存放整个网站信息的地方，包括了你的网站设置和博客文章。

访问[GitHub](https://github.com/)主页申请一个账号。完成申请后登陆，点击右上角的加号选择`New repository`（如下图）。

![GitHub](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_GitHub.png)

接着建立Repository，注意`Repository name`一定要以你的用户名（也就是前面的`Owner`) 开头，后面加上`.github.io`。勾上`Initialize this repository with a README`，点绿色按钮完成创建。

![Repo](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Repo.png)

##  2. 选择博客主题

创建GitHub Repository后，就到了最为愉悦 ~~纠结~~ 的审美阶段。

打开Jekyll Themes，绝大部分主题都是支持电脑、平板、手机三平台不同排版的。看到合眼缘的可以点进去，选择`Demo`可以进一步查看样板网站。你会发现很多平时看到的高大上的博客页面，都能在这里找到。

![Jekyll](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Jekyll.png)

选好后，点击主题页面中的`Homepage`（上图中`Demo`左边），来到这个主题的GitHub页面。选择右上角的`Fork`，此时GitHub就把这个网站的模板放到你的账号中，以一个Repository的方式保存。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQxODIwMDg3OV19
-->