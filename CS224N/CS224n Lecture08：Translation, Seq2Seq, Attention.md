# Lecture Plan
- 引⼊新任务：机器翻译
- 引⼊⼀种新的神经结构：sequence-to-sequence
  - 机器翻译是sequence-to-sequence的⼀个主要⽤例
- 引⼊⼀种新的神经技术：注意⼒
  - sequence-to-sequence通过attention得到提升
# Section 1: Pre-Neural Machine Translation
## Machine Translation
机器翻译(MT)是将⼀个句⼦ x 从⼀种语⾔( 源语⾔ )转换为另⼀种语⾔( ⽬标语⾔ )的句⼦ y 的任务。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212113539858.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## 1950s: Early Machine Translation
机器翻译研究始于20世纪50年代初。
- 俄语 $rightarrow$英语(冷战的推动)
- 系统主要是基于规则的，使⽤双语词典来讲俄语单词映射为对应的英语部分
## 1990s-2010s: Statistical Machine Translation
- 核⼼想法：从数据中学习概率模型。
- 假设我们正在翻译法语 $rightarrow$ 英语。
- 我们想要找到最好的英语句⼦ y ，给定法语句⼦ x
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212113719401.png#pic_center)

- 使⽤Bayes规则将其分解为两个组件从⽽分别学习

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212113728312.png#pic_center)
- $P(x | y)$
  - 翻译模型
  - 分析单词和短语应该如何翻译(逼真)
  - 从并⾏数据中学习
- $P(y)$
  - 语⾔模型
  - 模型如何写出好英语(流利)
  - 从单语数据中学习
- 问题：如何学习翻译模型$P(x | y)$
- ⾸先，需要⼤量的并⾏数据(例如成对的⼈⼯翻译的法语/英语句⼦)
## Learning alignment for SMT
- 问题：如何从并⾏语料库中学习翻译模型$P(x | y)$
- 进⼀步分解:我们实际上想要考虑
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212113900408.png#pic_center)


- $a$是对⻬，即法语句⼦ x 和英语句⼦ y 之间的单词级对应
## What is alignment?
对⻬是翻译句⼦中特定词语之间的对应关系。
- 注意：有些词没有对应词

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212113940699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Alignment is complex
对⻬可以是多对⼀的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114002756.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

对⻬可以是⼀对多的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114024290.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

有些词很丰富

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114040908.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114051179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

对⻬可以是多对多(短语级)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114109951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Learning alignment for SMT
- 我们学习很多因素的组合，包括
  - 特定单词对⻬的概率(也取决于发送位置)
  - 特定单词具有特定⽣育率的概率(对应单词的数量)
  - 等等
## Decoding for SMT
问题 ：如何计算argmax
- 我们可以列举所有可能的 y 并计算概率？ $rightarrow$太贵了
- 使⽤启发式搜索算法搜索最佳翻译，丢弃概率过低的假设
- 这个过程称为解码

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114230238.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114239792.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114248394.png#pic_center)

- SMT是⼀个巨⼤的研究领域
- 最好的系统⾮常复杂
  - 数以百计的重要细节我们还没有提到
  - 系统有许多分别设计⼦组件⼯程
    - 很多功能需要设计特性来获取特定的语⾔现象
  - 需要编译和维护额外的资源
    - ⽐如等价短语表
  - 需要⼤量的⼈⼒来维护
    - 对于每⼀对语⾔都需要重复操作
# Section 2: Neural Machine Translation
## What is Neural Machine Translation?
- 神经机器翻译是利⽤单个神经⽹络进⾏机器翻译的⼀种⽅法
- 神经⽹络架构称为sequence-to-sequence (⼜名seq2seq)，它包含两个RNNs
## Neural Machine Translation (NMT)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021211440027.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 编码器RNN⽣成源语句的编码
  - 源语句的编码为解码器RNN提供初始隐藏状态
- 解码器RNN是⼀种以编码为条件⽣成⽬标句的语⾔模型
- 注意：此图显示了测试时⾏为 $rightarrow$ 解码器输出作为下⼀步的输⼊
## Sequence-to-sequence is versatile!
- 序列到序列不仅仅对MT有⽤
- 许多NLP任务可以按照顺序进⾏表达
  - 摘要(⻓⽂本 $rightarrow$ 短⽂本)
  - 对话(前⼀句话 $rightarrow$ 下⼀句话)
  - 解析(输⼊⽂本 $rightarrow$ 输出解析为序列)
  - 代码⽣成(⾃然语⾔ $rightarrow$ Python代码)
- sequence-to-sequence 模型是 Conditional Language Model 条件语⾔模型的⼀个例⼦
  - 语⾔模型，因为解码器正在预测⽬标句的下⼀个单词 y
  - 有条件的，因为它的预测也取决于源句 x
- NMT直接计算 $P(y|x)$
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102121145435.png#pic_center)

- 上式中最后⼀项为，给定到⽬前为⽌的⽬标词和源句 x ，下⼀个⽬标词的概率
- 问题 ：如何培训NMT系统？
- 回答 ：找⼀个⼤的平⾏语料库
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114614116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Seq2seq被优化为⼀个单⼀的系统。反向传播运⾏在“端到端”中
## Greedy decoding
- 我们了解了如何⽣成(或“解码”)⽬标句，通过对解码器的每个步骤使⽤ argmax
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114638652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 这是贪婪解码(每⼀步都取最可能的单词)
- 这种⽅法有问题吗？
## Problems with greedy decoding
- 贪婪解码没有办法撤销决定
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114701447.png#pic_center)

- 如何修复？
## Exhaustive search decoding
- 理想情况下，我们想要找到⼀个(⻓度为 T )的翻译 y 使其最⼤化
- 我们可以尝试计算所有可能的序列 y
  - 这意味着在解码器的每⼀步 t ，我们跟踪 $V^t$个可能的部分翻译，其中$V$ 是 vocab ⼤⼩
  - 这种$O(V^T)$ 的复杂性太昂贵了！
## Beam search decoding
- 核⼼思想 ：在解码器的每⼀步，跟踪 k 个最可能的部分翻译(我们称之为 hypotheses 假设 )
  - k是Beam的⼤⼩(实际中⼤约是5到10)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212114855788.png#pic_center)

- 假设 $y_1,…,y_t$有⼀个分数，即它的对数概率
  - 分数都是负数，分数越⾼越好
  - 我们寻找得分较⾼的假设，跟踪每⼀步的 top k 个部分翻译
- 波束搜索 不⼀定能 找到最优解
- 但⽐穷举搜索效率⾼得多
## Beam search decoding: example
Beam size = k = 2
蓝⾊的数字是$score(y_1,…,y_t)=\sum_{i=1}^tlogP_{LM}(y_i| y_1,…,y_{i-1},x )$ 的结果
- 计算下⼀个单词的概率分布
- 取前k个单词并计算分数
- 对于每⼀次的 k 个假设，找出最前⾯的 k 个单词并计算分数
- 在 $k^2$ 的假设中，保留 k 个最⾼的分值，如 t = 2 时，保留分数最⾼的 hit 和 was
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121216694.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Beam search decoding: stopping criterion
- 在贪⼼解码中，我们通常解码到模型产⽣⼀个 $<END>$ 令牌
  - 例如: $<START>$ he hit me with a pie $<END>$
- 在 Beam Search 解码中，不同的假设可能在不同的时间步⻓上产⽣ $<END>$ 令牌
  - 当⼀个假设⽣成了 $<END>$ 令牌，该假设完成
  - 把它放在⼀边，通过 Beam Search 继续探索其他假设
- 通常我们继续进⾏ Beam Search ，直到
  - 我们到达时间步⻓ T (其中 T 是预定义截⽌点)
  - 我们⾄少有 n 个已完成的假设(其中 n 是预定义截⽌点)
## Beam search decoding: finishing up
- 我们有完整的假设列表
- 如何选择得分最⾼的？
- 我们清单上的每个假设 $y_1,…,y_t$ 都有⼀个分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121454321.png#pic_center)
- 问题在于 ：较⻓的假设得分较低
- 修正 ：按⻓度标准化。⽤下式来选择top one
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121509117.png#pic_center)

## Advantages of NMT
与SMT相⽐，NMT有很多优点
- 更好的性能
  - 更流利
  - 更好地使⽤上下⽂
  - 更好地使⽤短语相似性
- 单个神经⽹络端到端优化
  - 没有⼦组件需要单独优化
- 对所有语⾔对使⽤相同的⽅法
## Disadvantages of NMT?
SMT相⽐
- NMT的可解释性较差
  - 难以调试
- NMT很难控制
  - 例如，不能轻松指定翻译规则或指南
  - 安全问题
## How do we evaluate Machine Translation?
## BLEU (Bilingual Evaluation Understudy)
- 你将会在 Assignment 4 中看到BLEU的细节
- BLEU将机器翻译和⼈⼯翻译(⼀个或多个)，并计算⼀个相似的分数
  - n-gram 精度 (通常为1-4)
  - 对过于短的机器翻译的加上惩罚
- BLEU很有⽤,但不完美
  - 有很多有效的⽅法来翻译⼀个句⼦
  - 所以⼀个好的翻译可以得到⼀个糟糕的BLEU score，因为它与⼈⼯翻译的n-gram重叠较低
## MT progress over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121646637.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## NMT: the biggest success story of NLP Deep Learning
神经机器翻译于2014年从边缘研究活动到2016年成为领先标准⽅法
- 2014：第⼀篇 seq2seq 的⽂章发布
- 2016：⾕歌翻译从 SMT 换成了 NMT
- 这是惊⼈的
  - 由数百名⼯程师历经多年打造的SMT系统，在短短⼏个⽉内就被少数⼯程师训练过的NMT系统超越
## So is Machine Translation solved?
- 不！
- 许多困难仍然存在
  - 词表外的单词处理
  - 训练和测试数据之间的 领域不匹配
  - 在较⻓⽂本上维护上下⽂
  - 资源较低的语⾔对
- 使⽤常识仍然很难
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121745598.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- NMT在训练数据中发现偏差
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121801288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⽆法解释的系统会做⼀些奇怪的事情
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102121218135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## NMT research continues
NMT是NLP深度学习的核⼼任务
- NMT研究引领了NLP深度学习的许多最新创新
- 2019年：NMT研究将继续蓬勃发展
  - 研究⼈员发现，对于我们今天介绍的普通seq2seq NMT系统，有很多、很多的改进。
  - 但有⼀个改进是如此不可或缺
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121838963.png#pic_center)

# Section 3: Attention
## Sequence-to-sequence: the bottleneck problem
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121900859.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 源语句的编码需要捕获关于源语句的所有信息
- 信息瓶颈！
## Attention
- 注意⼒为瓶颈问题提供了⼀个解决⽅案
- 核⼼理念 ：在解码器的每⼀步，使⽤ 与编码器的直接连接 来专注于源序列的特定部分
- ⾸先我们将通过图表展示(没有⽅程)，然后我们将⽤⽅程展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212121935770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 将解码器部分的第⼀个token $<START>$与源语句中的每⼀个时间步的隐藏状态进⾏ Dot Product 得到每⼀时间步的分数
- 通过softmax将分数转化为概率分布
  - 在这个解码器时间步⻓上，我们主要关注第⼀个编码器隐藏状态(“he”)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122022455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 利⽤注意⼒分布对编码器的隐藏状态进⾏加权求和
- 注意⼒输出主要包含来⾃于受到⾼度关注的隐藏状态的信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122049401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 连接的 注意⼒输出 与 解码器隐藏状态 ，然后⽤来计算$y_1^*$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122122116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 有时，我们从前⾯的步骤中提取注意⼒输出，并将其输⼊解码器(连同通常的解码器输⼊)。我们在作业4中做这个。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122146400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Attention: in equations
- 我们有编码器隐藏状态$h_1,…,h_N\ \in \ R^h$
- 在时间步 t 上，我们有解码器隐藏状态$s_t \ \in \ R^h$
- 我们得到这⼀步的注意分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122302394.png#pic_center)

- 我们使⽤softmax得到这⼀步的注意分布 $\alpha^t$ (这是⼀个概率分布，和为1)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021212240554.png#pic_center)

- 我们使⽤ $\alpha^t$ 来获得编码器隐藏状态的加权和，得到注意⼒输出$a_t$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122429142.png#pic_center)

- 最后，我们将注意输出 $a_t$ 与解码器隐藏状态连接起来，并按照⾮注意seq2seq模型继续进⾏
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122446355.png#pic_center)


## Attention is great
- 注意⼒显著提⾼了NMT性能
  - 这是⾮常有⽤的，让解码器专注于某些部分的源语句
- 注意⼒解决瓶颈问题
  - 注意⼒允许解码器直接查看源语句；绕过瓶颈
- 注意⼒帮助消失梯度问题
  - 提供了通往遥远状态的捷径
- 注意⼒提供了⼀些可解释性
  - 通过检查注意⼒的分布，我们可以看到解码器在关注什么
  - 我们可以免费得到(软)对⻬
  - 这很酷，因为我们从来没有明确训练过对⻬系统
  - ⽹络只是⾃主学习了对⻬
## Attention is a general Deep Learning technique
- 我们已经看到，注意⼒是改进机器翻译的序列到序列模型的⼀个很好的⽅法
- 然⽽ ：您可以在许多体系结构(不仅仅是seq2seq)和许多任务(不仅仅是MT)中使⽤注意⼒
- 注意⼒的更⼀般定义
  - 给定⼀组向量 值 和⼀个向量 查询 ，注意⼒是⼀种根据查询，计算值的加权和的技术
- 我们有时说 query attends to the values
- 例如，在seq2seq + attention模型中，每个解码器的隐藏状态(查询)关注所有编码器的隐藏状态(值)
- 直觉
  - 加权和是值中包含的信息的选择性汇总，查询在其中确定要关注哪些值
  - 注意是⼀种获取任意⼀组表示(值)的固定⼤⼩表示的⽅法，依赖于其他⼀些表示(查询)。
## There are several attention variants
- 我们有⼀些值 $h_1,…,h_N \in R^{d_1}$ 和⼀个查询 $s \in R^{d_2}$
- 注意⼒总是包括
  - 计算注意⼒得分 $e \in R^N$（很多种计算⽅式）
  - 采取softmax来获得注意⼒分布$\alpha$
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122853943.png#pic_center)

  - 使⽤注意⼒分布对值进⾏加权求和：从⽽得到注意输出$a$ (有时称为上下⽂向量)
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122920336.png#pic_center)

## Attention variants
有⼏种⽅法可以从$h_1,…,h_N \in R^{d_1}$  计算 $e \in R^N$ 和$s \in R^{d_2}$
- 基本的点乘注意⼒$e_i=s^Th_i \in R$
  - 注意：这⾥假设$d_1=d_2$
  - 这是我们之前看到的版本
- 乘法注意⼒$e_i=s^TWh_i \in R$
  - $W \in R^{d_2 \times d_1}$是权重矩阵
- 加法注意⼒$e_i=v^Ttanh(W_1h_i + W_2s)\in R$
  - 其中$W_1 \in R^{d_3 \times d_1}$ , $W_2 \in R^{d_3 \times d_2}$是权重矩阵， $v \in R^{d_3}$是权重向量
  - $d_3$(注意⼒维度)是⼀个超参数
- 你们将在作业4中考虑这些的相对优势/劣势！
## Summary of today’s lecture
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212123017112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)