---
layout: post
title: 15分钟创建个人博客 @ GitHub Pages
key: 20180105
tags:
  - Tutorial
lang: en
---

作为新博客的第一篇文章，先写写我是如何创建这个博客的。与标题不同，我花了N倍于15分钟的时间来开启这个博客，而秉持着“解决核心问题，避免额外认知负担”的思路，最终采取了这一套简单稳定的方案。这也符合接触新事物时“粗浅--深入--精炼”的认知过程。

##  基本思路和准备条件

利用GitHub Pages项目免费的博客生成系统，在GitHub Repository中建立必要的网站文件结构，最终通过StackEdit以Markdown语言撰写博文。

建立并使用整个博客，我们需要完成下列几项：

1.  一个GitHub账号
2.  从Jekyll Themes中选择喜欢的样式，并复制到自己的GitHub中
3.  设置网站的基本信息
4.  创建博文
5. 用StackEdit撰写博文

这里面提到了Markdown语言，后面会提供一些别人教程的链接。这是一种非常简单快捷的写作方式，推荐所有用电脑写文章的朋友用额外的10分钟学习，一定物超所值（从高效写作的角度完全超越Word）。

##  1. 建立GitHub账号

简单来说，GitHub Repository是存放整个网站信息的地方，包括了你的网站设置和博客文章。因此我们需要在[GitHub](https://github.com/)上申请一个账号。

##  2. 选择博客样式

创建完账号后，就到了最为愉悦 ~~纠结~~ 的审美阶段。

打开[Jekyll Themes](http://jekyllthemes.org/)，绝大部分主题都是支持电脑、平板、手机三平台不同排版的。看到合眼缘的可以点进去，选择`Demo`可以进一步查看样板网站。你会发现很多平时看到的高大上的博客页面，都能在这里找到。

![Jekyll](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Jekyll.png)

选好后，点击主题页面中的`Homepage`（上图中`Demo`左边），来到这个主题的GitHub页面。选择右上角的`Fork`，此时GitHub就把这个网站的模板放到你的账号中，以一个Repository的方式保存。

![Fork](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Fork.png)

接着，点击下图中的`Settings`。

![Settings](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Settings.png)

将`Repository name`改为以你的用户名开头，后面加上`.github.io`。点`Rename`完成修改。

![Rename](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Rename.png)

此时去泡杯茶，跟家人聊聊天，10分钟后回来，登陆`https://你的用户名.github.io/`（也就是`https:// + Repository name`）。怎么样？你的博客已经以你喜欢的样子上线了！

##  3. 设置博客的基本信息

在此处仅以我的博客样式TeXt为例说明。[^1] 打开Repository根目录下的`_config.yml`（GitHub中点下图的铅笔按钮`Edit`）。

![YML](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_yml.png)

依次修改下列信息后，点最下方的绿色按钮保存：

-  `title`改成你的博客名
-  `description`是鼠标移到你的博客名上悬停时，显示的信息
-  `timezone`你所在的时区（默认亚洲/上海）
-  `author`中的信息是你的个人信息。比如你想在网站的下方显示你的Linkedin链接，就去掉`linkedin`项前面的`#`号，在冒号后填入你的Linkedin链接最后一段即可。

此外，你可以选择在`about.md`中写一些关于你自己的信息。[^2]

##  4. 创建博文

按照下图创建博文文件，在`Name your file...`处填入`_posts/YYYY-MM-DD-Your-Document-Name.md`，比如我这篇文章`_posts/2018-01-05-Create-a-Blog-in-15-Minutes.md`。

![Create1](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Create1.png)
![Create2](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_Create2.png)

在文档开头设置下列文章的信息，最后按最下方绿色按钮保存。

{% highlight Markdown %}
---
layout: post  # 文章样式，不建议修改
title: Markdown Example - Definition  # 文章的标题
tags: 
- tag1  # 文章标签1
- tag2  # 文章标签2
---
{% endhighlight %}

##  5. 用StackEdit撰写博文

（2018-03-23 更新：也可以使用[Dillinger](https://dillinger.io/)替代，如果你更喜欢在自己电脑上写，可以下载[Typora](https://typora.io/)）

GitHub Pages使用Markdown语言作为博文的编写语言。听到“语言”两个字先别慌，其实一篇Markdown文档中99%都是你自己写的内容，1%是控制文章格式的符号。花10分钟阅读一下[简书上的教程](https://www.jianshu.com/p/q81RER)就可以上手🏆。

回到正题，StackEdit是一个免费的Markdown在线编辑器，其最重要的特性是可以与GitHub Repository中的文件进行双向同步。也就是说，我们可以在GitHub中建立我们的文档后，在StackEdit中进行文章的撰写。~~其实最吸引人的特性是可以spell check~~

打开[StackEdit](https://stackedit.io/app)主页面，点击右上角的#按钮，选择`Synchronize`--`Add GitHub account`，点`OK`授权StackEdit访问你的GitHub信息。

![SE](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_SE1.png)

接着回到StackEdit，选择`Open from GitHub`，接着填入下列信息。

![SE](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-01-05_SE2.png)

大功告成！现在你可以在StackEdit开始写博文了。文章会自动同步到GitHub中，直接展现在你的博客上。

小贴士：StackEdit右上角的按钮里有`Markdown cheat sheet`可以随时查看Markdown的语法。

---
[^1]: 每个博客样板间有细微的差别，但基本都是通过对根目录下的`_config.yml`文件进行修改来完成设置。关于设置的说明可以在每个样板的GitHub Repository（第二步中点`Homepage`）的README.md中找到。
[^2]: `about.md`需使用Markdown语言编写。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU4OTYwODgxOCw1NzU1ODQ4ODZdfQ==
-->