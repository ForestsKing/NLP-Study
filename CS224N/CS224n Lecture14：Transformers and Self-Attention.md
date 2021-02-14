> Ashish Vaswani and Anna Huang

学习变⻓数据的表示，这是序列学习的基本组件（序列学习包括 NMT, text summarization, QA）

通常使⽤ RNN 学习变⻓的表示：RNN 本身适合句⼦和像素序列

- LSTMs, GRUs 和其变体在循环模型中占主导地位。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213091857870.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 但是序列计算抑制了并⾏化。
- 没有对⻓期和短期依赖关系进⾏显式建模。
- 我们想要对层次结构建模。
- RNNs(顺序对⻬的状态)看起来很浪费！

## 卷积神经⽹络
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213091920390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 并⾏化(每层)很简单
- 利⽤局部依赖
- 不同位置的交互距离是线性或是对数的
- 远程依赖需要多层

## 注意⼒
NMT 中，编码器和解码器之间的 Attention 是⾄关重要的

为什么不把注意⼒⽤于表示呢？

## Self-Attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213091947589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 任何两个位置之间的路径⻓度都是常数级别的
- ⻔控 / 乘法 的交互
- 可以并⾏化（每层）
- 可以完全替代序列计算吗？
# Text generation
## Previous work
**Classification & regression with self-attention:**
Parikh et al. (2016), Lin et al. (2016)

**Self-attention with RNNs:**
Long et al. (2016), Shao, Gows et al. (2017)

**Recurrent attention:**
Sukhbaatar et al. (2015)
## The Transformer
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092045419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Encoder Self-Attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092055876.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Decoder Self-Attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092106660.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 复杂度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092117731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

由于计算只涉及到两个矩阵乘法，所以是序列⻓度的平⽅
当维度⽐⻓度⼤得多的时候，⾮常有效
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092129457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Problem
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092141996.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

上例中，我们想要知道谁对谁做了什么，通过卷积中的多个卷积核的不同的线性操作，我们可以分别获取到 who, did what, to whom 的信息。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092156941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

但是对于 Attention ⽽⾔，如果只有⼀个Attention layer，那么对于⼀句话⾥的每个词都是同样的线性变换，不能够做到在不同的位置提取不同的信息

{>>这就是多头注意⼒的来源，灵感来源于 CNN 中的多个卷积核的设计<<}

## Solution
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092214933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Who, Did What, To Whom 分别拥有注意⼒头
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092227849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092232583.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092239796.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092246246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 将注意⼒层视为特征探测器
- 可以并⾏完成
- 为了效率，减少注意⼒头的维度，并⾏操作这些注意⼒层，弥补了计算 差距
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092303157.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021309230887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092319287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 但我们并不⼀定⽐ LSTM 取得了更好的表示，只是我们更适合 SGD，可以更好的训练
- 我们可以对任意两个词之间构建连接
# Importance of residuals
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092333623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

位置信息最初添加在了模型的输⼊处，通过残差连接将位置信息传递到每⼀层，可以不需要再每⼀层都添加位置信息

## Training Details
- ADAM optimizer with a learning rate warmup (warmup + exponential decay)
- Dropout during training at every layer just before adding residual
- Layer-norm
- Attention dropout (for some experiments)
- Checkpoint-averaging
- Label smoothing
- Auto-regressive decoding with beam search and length biasing
- ……
# Self-Similarity, Image and Music Generation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092420689.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092429275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Probabilistic Image Generation
- 模拟像素的联合分布
- 把它变成⼀个序列建模问题
- 分配概率允许度量泛化
- RNNs和CNNs是最先进的(PixelRNN, PixelCNN)
- incorporating gating CNNs 现在在效果上与 RNNs 相近
- 由于并⾏化，CNN 要快得多
- 图像的⻓期依赖关系很重要(例如对称性)
- 可能随着图像⼤⼩的增加⽽变得越来越重要
- 使⽤CNNs建模⻓期依赖关系需要两者之⼀
  - 多层可能使训练更加困难
  - ⼤卷积核 参数/计算成本相应变⼤
## Texture Synthesis with Self-Similarity
⾃相似性的研究案例

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092509195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092515396.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

A Non-local Algorithm for Image Denoising (Buades, Coll, and Morel. CVPR 2005)

Non-local Neural Networks (Wang et al., 2018)

## Previous work
**Self-attention:**
Parikh et al. (2016), Lin et al. (2016), Vaswani et al. (2017)

**Autoregressive Image Generation:**
A Oord et al. (2016), Salimans et al. (2017)

## The Image Transformer
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092542736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Combining Locality with Self-Attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092555875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092601523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 将注意⼒窗⼝限制为本地范围
- 由于空间局部性，这在图像中是很好的假设
# Music generation using relative self-attention
## Raw representations in music and language
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092630225.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092636395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

传统的 RNN 模型需要将⻓序列嵌⼊到固定⻓度的向量中
## Continuations to given initial motif
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092649806.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

给定⼀段⾳乐并⽣成后续⾳乐
- 不能直接去重复过去的⽚段
- 难以处理⻓距离
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092702634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092709210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092714556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 移动的固定过滤器捕获相对距离
- Music Transformer 使⽤平移不变性来携带超过其训练⻓度的关系信息，进⾏传递

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092727644.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092733184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 位置之间的相关性
- 但是⾳乐中的序列⻓度通常⾮常⻓

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092747179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092752938.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 将相对距离转化为绝对距离

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092802417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213092807700.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# Self-Attention
- 任意两个位置之间的路径⻓度是常数级的
- 没有边界的内存
- 并⾏化
- 对⾃相似性进⾏建模
- 相对注意⼒提供了表达时间、equivariance，可以⾃然延伸⾄图表