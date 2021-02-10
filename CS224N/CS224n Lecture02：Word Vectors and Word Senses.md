# Lecture Plan
- Finish looking at word vectors and word2vec
- Optimization basics
- Can we capture this essence more effectively by counting?
- The GloVe model of word vectors
- Evaluating word vectors
- Word senses
- The course
**Goal**: be able to read word embeddings papers by the end of class.
# Review: Main idea of word2vec
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210210356328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021021040759.png#pic_center)


- 遍历整个语料库中的每个单词
- 使⽤单词向量预测周围的单词
- 更新向量以便更好地预测
## Word2vec parameters and computations
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210210457508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210210525803.png#pic_center)

- 每⾏代表⼀个单词的词向量，点乘后得到的分数通过softmax映射为概率分布，并且我们得到的概率分布是对于该中⼼词⽽⾔的上下⽂中单词的概率分布，该分布于上下⽂所在的具体位置⽆关，所以在每个位置的预测都是⼀样的
- 我们希望模型对上下⽂中(相当频繁)出现的所有单词给出⼀个合理的⾼概率估计
- the, and, that, of 这样的停⽤词，是每个单词点乘后得到的较⼤概率的单词
  - 去掉这⼀部分可以使词向量效果更好
# Optimization: Gradient Descent
Gradient Descent 每次使⽤全部样本进⾏更新
Stochastic Gradient Descent 每次只是⽤单个样本进⾏更新
Mini-batch具有以下优点
- 通过平均值，减少梯度估计的噪⾳
- 在GPU上并⾏化运算，加快运算速度
## Stochastic gradients with word vectors
$\nabla_\theta J_t(\theta)$将会⾮常稀疏，所以我们可能只更新实际出现的向量
解决⽅案
- 需要稀疏矩阵更新操作来只更新矩阵U和V中的特定⾏
- 需要保留单词向量的散列
如果有数百万个单词向量，并且进⾏分布式计算，那么重要的是不必到处发送巨⼤的更新
## Word2vec: More details
为什么两个向量？
- 更容易优化，最后都取平均值
- 可以每个单词只⽤⼀个向量
两个模型变体
- Skip-grams (SG)
  - 输⼊中⼼词并预测上下⽂中的单词
- Continuous Bag of Words (CBOW)
  - 输⼊上下⽂中的单词并预测中⼼词
之前⼀直使⽤naive的softmax(简单但代价很⾼的训练⽅法)，接下来使⽤负采样⽅法加快训练速率
## The skip-gram model with negative sampling (HW2)
softmax中⽤于归⼀化的分⺟的计算代价太⾼
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210211602335.png#pic_center)

我们将在作业2中实现使⽤ **negative sampling** 负采样⽅法的 skip-gram 模型
- 使⽤⼀个 true pair (中⼼词及其上下⽂窗⼝中的词)与⼏个 noise pair (中⼼词与随机词搭配) 形成的样本，训练⼆元逻辑回归
原⽂中的(最⼤化)⽬标函数是
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021021164762.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

本课以及作业中的⽬标函数是
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210211702801.png#pic_center)

- 我们希望中⼼词与真实上下⽂单词的向量点积更⼤，中⼼词与随机单词的点积更⼩
- k是我们负采样的样本数⽬
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210211732824.png#pic_center)

使⽤上式作为抽样的分布，$U(m)$ 是 unigram 分布，通过$\frac{3}{4}$次⽅，相对减少常⻅单词的频率，增⼤稀有词的概率。 ⽤于⽣成概率分布。
# But why not capture co-occurrence counts directly?
共现矩阵 X
- 两个选项：windows vs. full document
- Window ：与word2vec类似，在每个单词周围都使⽤Window，包括语法(POS)和语义信息
- Word-document 共现矩阵的基本假设是在同⼀篇⽂章中出现的单词更有可能相互关联。假设单词 i 出现在⽂章 j  中，则矩阵元素$X_{ij}$加⼀，当我们处理完数据库中的所有⽂章后，就得到了矩阵 X，其⼤⼩为$|V|\times M$ ，其中 |V| 为词汇量，⽽ |M| 为⽂章数。这⼀构建单词⽂章co-occurrence matrix的⽅法也是经典的Latent Semantic Analysis所采⽤的。{>>潜在语义分析<<}

利⽤某个定⻓窗⼝中单词与单词同时出现的次数来产⽣window-based (word-word) co-occurrence matrix。下⾯以窗⼝⻓度为1来举例，假设我们的数据包含以下⼏个句⼦：
- I like deep learning.
- I like NLP.
- I enjoy flying.
则我们可以得到如下的word-word co-occurrence matrix:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210212127919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

使⽤共现次数衡量单词的相似性，但是会随着词汇量的增加⽽增⼤矩阵的⼤⼩，并且需要很多空间来存储这⼀⾼维矩阵，后续的分类模型也会由于矩阵的稀疏性⽽存在稀疏性问题，使得效果不佳。我们需要对这⼀矩阵进⾏降维，获得低维(25-1000)的稠密向量。
## Method 1: Dimensionality Reduction on X (HW1)
使⽤SVD⽅法将共现矩阵 X 分解为$U\Sigma V^T$ ， $\Sigma$是对⻆线矩阵，对⻆线上的值是矩阵的奇异值。U,V 是对应于⾏和列的正交基。
为了减少尺度同时尽量保存有效信息，可保留对⻆矩阵的最⼤的k个值，并将矩阵U,V 的相应的⾏列保留。这是经典的线性代数算法，对于⼤型矩阵⽽⾔，计算代价昂贵。
## Hacks to X (several used in Rohde et al. 2005)
按⽐例调整 counts 会很有效
- 对⾼频词进⾏缩放(语法有太多的影响)
  - 使⽤log进⾏缩放
  - $min(X,t), t\approx 100$
  - 直接全部忽视
- 在基于window的计数中，提⾼更加接近的单词的计数
- 使⽤Person相关系数
> Conclusion：对计数进⾏处理是可以得到有效的词向量的

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210212533466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

在向量中出现的有趣的句法模式：语义向量基本上是线性组件，虽然有⼀些摆动，但是基本是存在动词和动词实施者的⽅向。
**基于计数**：使⽤整个矩阵的全局统计数据来直接估计
- 优点
  - 训练快速
  - 统计数据⾼效利⽤
- 缺点
  - 主要⽤于捕捉单词相似性
  - 对⼤量数据给予⽐例失调的重视
**转换计数**：定义概率分布并试图预测单词
- 优点
  - 提⾼其他任务的性能
  - 能捕获除了单词相似性以外的复杂的模式
- 缺点
  - 与语料库⼤⼩有关的量表
  - 统计数据的低效使⽤（采样是对统计数据的低效使⽤）
## Encoding meaning in vector differences
将两个流派的想法结合起来，在神经⽹络中使⽤计数矩阵
**关键思想**：共现概率的⽐值可以对meaning component进⾏编码
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210212712860.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

重点不是单⼀的概率⼤⼩，重点是他们之间的⽐值，其中蕴含着meaning component。
> 例如我们想区分热⼒学上两种不同状态ice冰与蒸汽steam，它们之间的关系可通过与不同的单词 x 的co-occurrence probability 的⽐值来描述。
例如对于solid固态，虽然$P(solid|ice)$ 与 $P(solid|steam)$ 本身很⼩，不能透露有效的信息，但是它们的⽐值$\frac{P(solid|ice) }{P(solid|steam)}$却较⼤，因为solid更常⽤来描述ice的状态⽽不是steam的状态，所
以在ice的上下⽂中出现⼏率较⼤
对于gas则恰恰相反，⽽对于water这种描述ice与steam均可或者fashion这种与两者都没什么联系的单词，则⽐值接近于1。所以相较于单纯的co-occurrence probability，实际上co-occurrence probability的相对⽐值更有意义

我们如何在词向量空间中以线性meaning component的形式捕获共现概率的⽐值？
log-bilinear 模型 : $w_i·w_j=logP(i|j)$
向量差异 : $w_x(w_a-w_b)=log\frac{P(x|a)}{P(x|b)}$
- 如果使向量点积等于共现概率的对数，那么向量差异变成了共现概率的⽐率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213236180.png#pic_center)

- 使⽤平⽅误差促使点积尽可能得接近共现概率的对数
- 使⽤ 对常⻅单词进⾏限制

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213258842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 优点
  - 训练快速
  - 可以扩展到⼤型语料库
  - 即使是⼩语料库和⼩向量，性能也很好
# How to evaluate word vectors?
与NLP的⼀般评估相关：内在与外在
- 内在
  - 对特定/中间⼦任务进⾏评估
  - 计算速度快
  - 有助于理解这个系统
  - 不清楚是否真的有⽤，除⾮与实际任务建⽴了相关性
- 外在
  - 对真实任务的评估
  - 计算精确度可能需要很⻓时间
  - 不清楚⼦系统是问题所在，是交互问题，还是其他⼦系统
  - 如果⽤另⼀个⼦系统替换⼀个⼦系统可以提⾼精确度
## Intrinsic word vector evaluation
词向量类⽐ a:b :: c:?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213358797.png#pic_center)

- 通过加法后的余弦距离是否能很好地捕捉到直观的语义和句法类⽐问题来评估单词向量
- 从搜索中丢弃输⼊的单词
- 问题:如果有信息但不是线性的怎么办？
Glove可视化效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213423981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213435395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213452420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> 可以使⽤数据集评估语法和语义上的效果

## Analogy evaluation and hyperparameters
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213517944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 300是⼀个很好的词向量维度
- 不对称上下⽂(只使⽤单侧的单词)不是很好，但是这在下游任务重可能不同
- window size 设为 8 对 Glove向量来说⽐较好
- 分析
  - window size设为2的时候实际上有效的，并且对于句法分析是更好的，因为句法效果⾮常局部
## On the Dimensionality of Word Embedding
利⽤矩阵摄动理论，揭示了词嵌⼊维数选择的基本的偏差与⽅法的权衡
当持续增⼤词向量维度的时候，词向量的效果不会⼀直变差并且会保持平稳
## Analogy evaluation and hyperparameters
- 训练时间越⻓越好
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021021360536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 数据集越⼤越好，并且维基百科数据集⽐新闻⽂本数据集要好
  - 因为维基百科就是在解释概念以及他们之间的相互关联，更多的说明性⽂本显示了事物之间的所有联系
  - ⽽新闻并不去解释，⽽只是去阐述⼀些事件

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213630172.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Another intrinsic word vector evaluation
使⽤ cosine similarity 衡量词向量之间的相似程度
# Word senses and word sense ambiguity
⼤多数单词都是多义的
- 特别是常⻅单词
- 特别是存在已久的单词
## Improving Word Representations Via Global Context And Multiple Word Prototypes (Huang et al. 2012)
将常⽤词的所有上下⽂进⾏聚类，通过该词得到⼀些清晰的簇，从⽽将这个常⽤词分解为多个单词，例如 bank_1, bank_2, bank_3
虽然这很粗糙，并且有时sensors之间的划分也不是很明确甚⾄相互重叠
## Linear Algebraic Structure of Word Senses, with Applications to Polysemy、
- 单词在标准单词嵌⼊(如word2vec)中的不同含义以线性叠加(加权和)的形式存在， $f$指频率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210213742282.png#pic_center)

令⼈惊讶的结果，只是加权平均值就已经可以获得很好的效果
- 由于从稀疏编码中得到的概念，你实际上可以将感官分离出来(前提是它们相对⽐较常⻅)
- 可以理解为由于单词存在于⾼维的向量空间之中，不同的纬度所包含的含义是不同的，所以加权平均值并不会损害单词在不同含义所属的纬度上存储的信息
## Extrinsic word vector evaluation
单词向量的外部评估：词向量可以应⽤于NLP的很多任务