# Lecture Plan

- The course
- Human language and word meaning 
- Word2vec introduction
- Word2vec objective function gradients 
- Optimization basics
- Looking at word vectors

# Human language and word meaning

⼈类之所以⽐类⼈猿更“聪明”，是因为我们有语⾔，因此是⼀个⼈机⽹络，其中⼈类语⾔作为⽹络语⾔。⼈类语⾔具有 **信息功能** 和 **社会功能** 。

据估计，⼈类语⾔只有⼤约5000年的短暂历。语⾔是⼈类变得强⼤的主要原因。写作是另⼀件让⼈类变 得强⼤的事情。它是使知识能够在空间上传送到世界各地，并在时间上传送的⼀种⼯具。

但是，相较于如今的互联⽹的传播速度⽽⾔，⼈类语⾔是⼀种缓慢的语⾔。然⽽，只需⼈类语⾔形式的⼏百位信息，就可以构建整个视觉场景。这就是⾃然语⾔如此迷⼈的原因。

## How do we represent the meaning of a word?

 ***meaning***
- ⽤⼀个词、词组等表示的概念。
- ⼀个⼈想⽤语⾔、符号等来表达的想法。
- 表达在作品、艺术等⽅⾯的思想。

理解意义的最普遍的语⾔⽅式(linguistic way) : 语⾔符号与语⾔符号的意义的转化


![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021020173886.png#pic_center)


> denotational semantics 指称语义

## How do we have usable meaning in a computer?

***WordNet***, ⼀个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms**的列表的辞典![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210201931391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Problems with resources like WordNet

- 作为⼀个资源很好，但忽略了细微差别
  - 例如“proﬁcient”被列为“good”的同义词。这只在某些上下⽂中是正确的。
- 缺少单词的新含义
  - 难以持续更新
  - 例如 wicked, badass, nifty, wizard, genius, ninja, bombest
- 主观的
-  需要⼈类劳动来创造和调整
- ⽆法计算单词相似度

## Representing words as discrete symbols

在传统的⾃然语⾔处理中，我们把词语看作离散的符号: hotel, conference, motel - a **localist** representation。单词可以通过独热向量(one-hot vectors，只有⼀个1，其余均为0的稀疏向量) 。向量维度=词汇量(如500,000)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210202156798.png#pic_center)



## Problem with words as discrete symbols**

所有向量是正交的。对于独热向量，没有关于相似性概念，并且向量维度过⼤。

## Solutions

- 使⽤类似 ***WordNet*** 的⼯具中的列表，获得相似度，但会因不够完整⽽失败
- 学习在向量本身中编码相似性

## Representing words by their context

- ***Distributional semantics*** ：⼀个单词的意思是由经常出现在它附近的单词给出的
  - *“You shall know a word by the company it keeps”* ( J. R. Firth 1957: 11)
  - 现代统计NLP最成功的理念之⼀
  - 有点物以类聚，⼈以群分的感觉
- 当⼀个单词 出现在⽂本中时，它的上下⽂是出现在其附近的⼀组单词(在⼀个固定⼤⼩的窗口中)。
- 使⽤$w$的许多上下⽂来构建$w$的表示

![在这里插入图片描述](https://img-blog.csdnimg.cn/202102102025126.png#pic_center)


# Word2vec introduction

我们为每个单词构建⼀个 **密集** 的向量，使其与出现在相似上下⽂中的单词向量相似词向量 ***word vectors*** 有时被称为词嵌⼊ ***word embeddings*** 或词表示 ***word representations***
它们是分布式表示 ***distributed representation***

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210202546966.png#pic_center)


***Word2vec*** (Mikolov et al. 2013)是⼀个学习单词向量的 **框架**

**IDEA**：

- 我们有⼤量的⽂本 (corpus means 'body' in Latin. 复数为corpora) 
- 固定词汇表中的每个单词都由⼀个向量表示
- ⽂本中的每个位置$t$，其中有⼀个中⼼词$c$和上下⽂(“外部”)单词$o$
- 使⽤ $c$和 $o$的 **词向量的相似性** 来计算给定 的 的 **概率** (反之亦然) 
- **不断调整词向量** 来最⼤化这个概率

下图为窗⼝⼤⼩$j=2$时的$P(w_{t+j}|w_t)$ 计算过程，center word分别为 into 和 banking

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210202926686.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


# Word2vec objective function

对于每个位置 t=1,…,T ，在⼤⼩为 m 的固定窗⼝内预测上下⽂单词，给定中⼼词$w_j$

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021020304695.png#pic_center)


- 其中， 为所有需要优化的变量

⽬标函数$J(\theta)$(有时被称为代价函数或损失函数) 是(平均)负对数似然

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210203318604.png#pic_center)

其中log形式是⽅便将连乘转化为求和，负号是希望将极⼤化似然率转化为极⼩化损失函数的等价问题。

> 在连乘之前使⽤log转化为求和⾮常有效，特别是在做优化时
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210203416424.png#pic_center)

-  **最⼩化⽬标函数 最⼤化预测精度**
- *问题*：如何计算$P(w_{t+j}|w_t; \theta)$ ？ 
- *回答*：对于每个单词都是⽤两个向量
  - $v_w$当$w$是中⼼词时
  - $u_w$当$w$是上下⽂词时
- 于是对于⼀个中⼼词$c$ 和⼀个上下⽂词$o$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210203911435.png#pic_center)

> 公式中，向量$u_o$和向量$v_c$进⾏点乘。向量之间越相似，点乘结果越⼤，从⽽归⼀化后得到的概率值也越⼤。模型的训练正是为了使得具有相似上下⽂的单词，具有相似的向量。

# Word2vec prediction function

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204203591.png#pic_center)


- 取幂使任何数都为正
- 点积⽐较o和c的相似性$u^Tv=u·v=\sum_{i=1}^nv_iu_i$ ，点积越⼤则概率越⼤
- 分⺟：对整个词汇表进⾏标准化，从⽽给出概率分布

**softmax function**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204401624.png#pic_center)


将任意值$x_i$映射到概率分布$p_i$

- **max** ：因为放⼤了最⼤的概率
- **soft** ：因为仍然为较⼩的$x_i$赋予了⼀定概率
- 深度学习中常⽤

⾸先我们随机初始化$u_w,\ v_w$ ，⽽后使⽤梯度下降法进⾏更新

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204557181.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


> 偏导数可以移进求和中，对应上⽅公式的最后两⾏的推导
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204606998.png#pic_center)

我们可以对上述结果重新排列如下，第⼀项是真正的上下⽂单词，第⼆项是预测的上下⽂单词。使⽤梯 度下降法，模型的预测上下⽂将逐步接近真正的上下⽂。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204704599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

再对$u_o$进⾏偏微分计算，注意这⾥的$u_o$是$u_{w=o}$的简写，故可知

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210204844369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

可以理解，当$P(o|c)\rightarrow 1$即通过中⼼词$c$我们可以正确预测上下⽂词$o$，此时我们不需要调整$u_o$，反之，则相应调整$u_o$ 。