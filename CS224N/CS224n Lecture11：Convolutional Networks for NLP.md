# Lecture Plan
1. Announcements
2. Intro to CNNs
3. Simple CNN for Sentence Classification: Yoon (2014)
4. CNN potpourri
5. Deep CNN for Sentence Classification: Conneauet al. (2017)
6. Quasi-recurrent Neural Networks
# 1.Welcome to the second half of the course!
- 现在，我们正在为您准备成为DL+NLP研究⼈员/实践者
- 讲座不会总是有所有的细节
  - 这取决于你在⽹上搜索/阅读来了解更多
  - 这是⼀个活跃的研究领域，有时候没有明确的答案
  - Staff 很乐意与你讨论，但你需要⾃⼰思考
- 作业的设计是为了应付项⽬的真正困难
  - 每个任务都故意⽐上⼀个任务有更少的帮助材料。
  - 在项⽬中，没有提供 autograder 或 合理性检查
  - DL调试很困难，但是您需要学习如何进⾏调试！
## Wanna read a book?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134411825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 2. From RNNs to Convolutional Neural Nets
- 递归神经⽹络不能捕获没有前缀上下⽂的短语
- 经常在最终向量中捕获太多的最后单词
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134432819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 例如，softmax通常只在最后⼀步计算
## From RNNs to Convolutional Neural Nets
- 卷积⽹络的主要想法：
- 如果我们为每个可能的⼦序列计算⼀定⻓度的向量呢？
- 例如：“tentative deal reached to keep government open” 计算的向量为
  - tentative deal reached, deal reached to, reached to keep, to keep government, keep government open
- 不管短语是否合乎语法
- 在语⾔学上或认知上不太可信
- 然后将它们分组(很快)
## What is a convolution anyway?
- ⼀维离散卷积⼀般为：$(f*g)[n]=\sum_{m=-M}^Mf[n-m]g[m]$
- 卷积经典地⽤于从图像中提取特征
  - 模型位置不变的识别
  - 转到cs231n！
- 2d example
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134643186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⻩⾊和红⾊数字显示过滤器(=内核)权重
- 绿⾊显示输⼊
- 粉⾊显示输出
## A 1D convolution for text
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134714255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 1D convolution for text with padding
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134732921.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 输⼊⻓度为 的词序列，假设单词维度为 4，即有 4 channels
- 卷积后将会得到 1 channel
## 3 channel 1D convolution with padding = 1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134754514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 多个channel则最终得到多个channel的输出，关注的⽂本潜在特征也不同
## conv1d, padded with max pooling over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134815134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 通过 max pooling over time，获得最⼤的激活值
## conv1d, padded with avepooling over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134839279.png#pic_center)

## In PyTorch

```python
batch_size= 16
word_embed_size= 4
seq_len= 7
input = torch.randn(batch_size, word_embed_size, seq_len)
conv1 = Conv1d(in_channels=word_embed_size, out_channels=3, kernel_size=3)
# can add: padding=1
hidden1 = conv1(input)
hidden2 = torch.max(hidden1, dim=2) # max pool
```


## Other less useful notions: stride = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134942153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- stride 步⻓，减少计算量
## Less useful: local max pool, stride = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135007794.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
- 每两⾏做 max pooling，被称为步⻓为2的局部最⼤池化
## conv1d, k-max pooling over time, k= 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135040720.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 记录每⼀个channel的所有时间的 top k的激活值，并且按原有顺序保留（上例中的-0.2 0.3）
## Other somewhat useful notions: dilation = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135059135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 扩张卷积
- 上例中，对1 3 5⾏进⾏卷积，通过两个filter得到两个channel的激活值
- 可以在第⼀步的卷积中将卷积核从3改为5，即可实现这样的效果，既保证了矩阵很⼩，⼜保证了⼀次卷积中看到更⼤范围的句⼦

**Summary**
- 在CNN中，⼀次能看⼀个句⼦的多少内容是很重要的概念
- 可以使⽤更⼤的filter、扩张卷积或者增⼤卷积深度（层数）
# 3. Single Layer CNN for Sentence Classification

- ⽬标：句⼦分类
  - 主要是句⼦的积极或消极情绪
  - 其他任务
    - 判断句⼦主观或客观
    - 问题分类：问题是关于什么实体的？关于⼈、地点、数字、……
- ⼀个卷积层和池化层的简单使⽤
- 词向量：$x_i \in R^k$
- 句⼦：$x_{1:n}=x_1 \bigoplus x_2 \bigoplus …\bigoplus x_n$ (向量连接)
- 连接 $X_{i:i+j}$范围内的句⼦ (对称更常⻅)
- 卷积核 $w \in R^{HK}$(作⽤范围为 h 个单词的窗⼝)
- 注意，filter是向量，size 可以是2,3或4
## Single layer CNN
- 过滤器 $w$应⽤于所有可能的窗⼝(连接向量) 
- 为CNN层计算特征(⼀个通道)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135623448.png#pic_center)

- 句⼦
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135710704.png#pic_center)

- 所有可能的⻓度为 h 的窗⼝
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135722754.png#pic_center)

- 结果是⼀个 feature map 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135739607.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135755918.png#pic_center)

## Pooling and channels
- 池化：max-over-time pooling layer
- 想法：捕获最重要的激活(maximum over time)
- 从feature map中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135918142.png#pic_center)

- 池化得到单个数字
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135928837.png#pic_center)

- 使⽤多个过滤器权重w
- 不同窗⼝⼤⼩ h 是有⽤的
- 由于最⼤池化 ，和c 的⻓度⽆关
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140008879.png#pic_center)

- 所以我们可以有⼀些 filters 来观察 unigrams, bigrams, tri-grams, 4-grams,等等
## Multi-channel input idea
- 使⽤预先训练的单词向量初始化(word2vec或Glove)
- 从两个副本开始
- 只有⼀个副本进⾏了反向传播，保持其他“静态”
- 两个通道集都在最⼤池化前添加到$c_i$
## Classification after one CNN layer
- ⾸先是⼀个卷积，然后是⼀个最⼤池化
- 为了获得最终的特征向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140106544.png#pic_center)

  - 假设我们有 m 个 filter 
  - 使⽤100个⼤⼩分别为3、4、5的特征图
- 最终是简单的 softmax layer 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140138206.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140524434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 输⼊⻓度为 7 的⼀句话，每个词的维度是 5 ，即输⼊矩阵是
- 使⽤不同的filter_size : (2,3,4)，并且每个size都是⽤两个filter，获得两个channel的feature，即共计6个filter
- 对每个filter的feature进⾏1-max pooling后，拼接得到 6 维的向量，并使⽤softmax后再获得⼆分类结果
## Regularization
- 使⽤ Dropout : 使⽤概率 p (超参数)的伯努利随机变量（只有0 1并且p是为1的概率）创建mask向量 r
- 训练过程中删除特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140627105.png#pic_center)

- 解释：防⽌互相适应(对特定特征的过度拟合)(Srivastava, Hinton, et al. 2014)
- 在测试时不适⽤dropout，使⽤概率p缩放最终向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140647605.png#pic_center)

- 此外：限制每个类的权重向量的L2 Norm(softmax 权重 的每⼀⾏)不超过固定数s (也是超参
数)
- 如果$||W_c{(S)}||>s$ ，则重新缩放为$||W_c{(S)}||=s$
  - 不是很常⻅
- All hyperparametersin Kim (2014)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140827760.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Problem with comparison?
- Dropout提供了2 - 4%的精度改进
- 但⼏个⽐较系统没有使⽤Dropout，并可能从它获得相同的收益
- 仍然被视为⼀个简单架构的显著结果
- 与我们在前⼏节课中描述的窗⼝和RNN架构的不同之处：池化、许多过滤器和dropout
- 这些想法中有的可以被⽤在RNNs中
# 4. Model comparison: Our growing toolkit
- Bag of Vectors ：对于简单的分类问题，这是⼀个⾮常好的基线。特别是如果后⾯有⼏个ReLU层(See paper: Deep Averaging Networks)
- Window Model ：对于不需要⼴泛上下⽂的问题（即适⽤于local问题），适合单字分类。例如POS, NER
- CNNs ：适合分类，较短的短语需要零填充，难以解释，易于在gpu上并⾏化
- RNNs ：从左到右的认知更加具有可信度，不适合分类(如果只使⽤最后⼀种状态)，⽐CNNs慢得多，适合序列标记和分类以及语⾔模型，结合注意⼒机制时⾮常棒
RNN对序列标记和分类之类的事情有很好的效果，以及语⾔模型预测下⼀个单词，并且结合注意⼒机制会取得很好的效果，但是对于某个句⼦的整体解释，CNN做的是更好的
## Gated units used vertically
- 我们在LSTMs和GRUs中看到的⻔/跳过是⼀个普遍的概念，现在在很多地⽅都使⽤这个概念
- 您还可以使⽤垂直的⻔
- 实际上，关键的概念——⽤快捷连接对候选更新求和——是⾮常深的⽹络⼯作所需要的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140945669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Note: pad x for conv so same size when add them
## Batch Normalization (BatchNorm)
- 常⽤于 CNNs
- 通过将激活量缩放为零均值和单位⽅差，对⼀个mini-batch的卷积输出进⾏变换
  - 这是统计学中熟悉的 Z-transform
  - 但在每组mini-batch都会更新，所以波动的影响不⼤
- 使⽤BatchNorm使模型对参数初始化的敏感程度下降，因为输出是⾃动重新标度的
  - 也会让学习率的调优更简单
  - 模型的训练会更加稳定
- PyTorch: nn.BatchNorm1d
## 1 x 1 Convolutions
- 这个概念有意义吗?是的
- 1x1卷积，即⽹络中的 Network-in-network (NiN) connections，是内核⼤⼩为1的卷积内核
- 1x1卷积为您提供了⼀个跨通道的全连接的线性层
- 它可以⽤于从多个通道映射到更少的通道
- 1x 1卷积添加了额外的神经⽹络层，附加的参数很少，
  - 与全连接(FC)层不同，全连接(FC)层添加了⼤量的参数
## CNN application: Translation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141128771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 最早成功的神经机器翻译之⼀
- 使⽤CNN进⾏编码，使⽤RNN进⾏解码

## Learning Character-level Representations for Part-of-Speech Tagging
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021114115771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 对字符进⾏卷积以⽣成单词嵌⼊
- 固定窗⼝的词嵌⼊被⽤于POS标签
## Character-Aware Neural Language Models![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141226799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 基于字符的单词嵌⼊
- 利⽤卷积、highway network和LSTM
# 5. Very Deep Convolutional Networks for Text Classification
- 起始点：序列模型(LSTMs)在NLP中占主导地位；还有CNNs、注意⼒等等，但是所有的模型基本上都不是很深⼊——不像计算机视觉中的深度模型
- 当我们为NLP构建⼀个类似视觉的系统时会发⽣什么
- 从字符级开始⼯作
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141329412.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Convolutional block in VD-CNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141343409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 每个卷积块是两个卷积层，每个卷积层后⾯是BatchNorm和⼀个ReLU
- 卷积⼤⼩为3
- pad 以保持(或在局部池化时减半)维数
## Experiments
- 使⽤⼤⽂本分类数据集
  - ⽐NLP中经常使⽤的⼩数据集⼤得多，如Yoon Kim(2014)的论⽂
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141414596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 以上数据均为错误率，所以越低越好
- 深度⽹络会取得更好的结果，残差层取得很好的结果，但是深度再深时并未取得效果提升
- 实验表明使⽤ MaxPooling ⽐ KMaxPooling 和 使⽤stride的卷积 的两种其他池化⽅法要更好
- ConvNets可以帮助我们建⽴很好的⽂本分类系统
# 6. RNNs are Slow …
- RNNs是深度NLP的⼀个⾮常标准的构建块
- 但它们的并⾏性很差，因此速度很慢
- 想法：取RNNs和CNNs中最好且可并⾏的部分

## Quasi-Recurrent Neural Network
- 努⼒把两个模型家族的优点结合起来
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141502416.png#pic_center)

- 时间上并⾏的卷积，卷积计算候选，遗忘⻔和输出⻔
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021114151730.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141524647.png#pic_center)

- 跨通道并⾏性的逐元素的⻔控伪递归是在池化层中完成的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141540437.png#pic_center)

## Q-RNN Experiments: Language Modeling
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021114160411.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Q-RNNs for Sentiment Analysis
- 通常⽐LSTMs更好更快
- 更加的可解释
- 例如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211141624415.png#pic_center)

## QRNN limitations
- 对于字符级的LMs并不像LSTMs那样有效
  - 建模时遇到的更⻓的依赖关系问题
- 通常需要更深⼊的⽹络来获得与LSTM⼀样好的性能
  - 当它们更深⼊时，速度仍然更快
  - 有效地使⽤深度作为真正递归的替代