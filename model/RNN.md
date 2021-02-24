# 0 代码：
- [RNN](https://github.com/ForestsKing/NLP-Study/blob/master/demo/RNN.ipynb)
- [BiLSTM](https://github.com/ForestsKing/NLP-Study/blob/master/demo/BiLSTM.ipynb)
- [BiLSTM_CRF](https://github.com/ForestsKing/NLP-Study/blob/master/demo/biLSTM_CRF.ipynb)
- [Seq2Seq](https://github.com/ForestsKing/NLP-Study/blob/master/demo/Seq2Seq.ipynb)
- [Seq2Seq(Attention)](https://github.com/ForestsKing/NLP-Study/blob/master/demo/Seq2Seq(Attention).ipynb)

# 1 RNN
核⼼想法：重复使⽤ **相同** 的权重矩阵 W
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222908171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222920929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## 1.1 Training a RNN Language Model
- 获取⼀个较⼤的⽂本语料库，该语料库是⼀个单词序列
- 输⼊RNN-LM；计算每个步骤 t 的输出分布
  - 即预测到⽬前为⽌给定的每个单词的概率分布
- 步骤 t 上的损失函数为预测概率分布$y^{(t)*}$与真实下⼀个单词$y^{(t)}$ ( $x^{(t+1)}$的one-hot向量)之间的交叉熵
- 将其平均，得到整个培训集的总体损失
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223226546.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 1.2 Generating text with a RNN Language Model
就像n-gram语⾔模型⼀样，您可以使⽤RNN语⾔模型通过 重复采样 来 ⽣成⽂本 。采样输出是下⼀步的输⼊。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122374939.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
## 1.3 Bidirectional RNNs
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231315182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
## 1.4 Multi-layer RNNs

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231455481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
# 2 LSTM(Long Short-Term Memory)
## 2.1 LSTM 网络
所有循环神经网络结构都是由结构完全相同的模块进行复制而成的。在普通的 RNN 中，这个模块非常简单，比如一个单一的$tanh$层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223100117656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
LSTM 也有类似的结构，唯一的区别就是中间的部分，LSTM 不再只是一个单一的$tanh$层，而使用了四个相互作用的层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223100157986.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
首先，我先解释一下里面用到的符号
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223100233266.png#pic_center)
在网络结构图中，每条线都传递着一个向量，从一个节点中输入到另一个节点。黄色的矩阵表示的是一个神经网络层；粉红色的圆圈表示逐点操作，如向量乘法、加法等；合并的线表示把两条线上所携带的向量进行合并（比如一个是$h_{t-1}$，另一个是$x_t$，那么合并后的输出就是$[h_{t-1},x_t]$；分开的线表示将线上传递的向量复制一份，传给两个地方
## 2.2 LSTM 核心思想
LSTM 的关键是 cell 状态，即贯穿图顶部的水平线。cell 状态的传输就像一条传送带，向量从整个 cell 中穿过，只是做了少量的线性操作，这种结构能很轻松地实现信息从整个 cell 中穿过而不做改变（这样就可以实现长时期地记忆保留）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223100639445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
LSTM 也有能力向 cell 状态中添加或删除信息，这是由称为门（gates）的结构仔细控制的。门可以选择性的让信息通过，它们由 sigmoid 神经网络层和逐点相乘实现
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223100809333.png#pic_center)
每个 LSTM 有三个这样的门结构来实现控制信息（分别是 forget gate 遗忘门；input gate 输入门；output gate 输出门）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230600378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
你可以把LSTM⽅程想象成这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021123065989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
## 2.3 逐步理解 LSTM
### 2.3.1 遗忘门
LSTM 的第一步是决定要从 cell 状态中丢弃什么信息，这个决定是由一个叫做 forget gate layer 的 sigmoid 神经层来实现的。它的输入是  $h_{t-1}$和$x_t$ ，输出是一个数值都在 0~1 之间的向量（向量长度和 $C_{t-1}$ 一样），表示让$C_{t-1}$的各部分信息通过的比重，0 表示不让任何信息通过，1 表示让所有信息通过

思考一个具体的例子，假设一个语言模型试图基于前面所有的词预测下一个单词，在这种情况下，每个 cell 状态都应该包含了当前主语的性别（保留信息），这样接下来我们才能正确使用代词。但是当我们又开始描述一个新的主语时，就应该把旧主语的性别给忘了才对（忘记信息）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223101043485.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
### 2.3.2 输入门
下一步是决定要让多少新的信息加入到 cell 状态中。实现这个需要包括两个步骤：首先，一个叫做 input gate layer 的 sigmoid 层决定哪些信息需要更新。另一个$tanh$层创建一个新的 candidate 向量 $\widetilde C_t$。最后，我们把这两个部分联合起来对 cell 状态进行更新
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022310145581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
在我们的语言模型的例子中，我们想把新的主语性别信息添加到 cell 状态中，替换掉老的状态信息。有了上述的结构，我们就能够更新 cell 状态了，即把$C_{t-1}$更新为$C_t$ 。从结构图中应该能一目了然，首先我们把旧的状态$C_{t-1}$和$f_t$相乘，把一些不想保留的信息忘掉，然后加上$i_t*\widetilde C_t$。这部分信息就是我们要添加的新内容
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223101717350.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
### 2.3.3 输出门
最后，我们需要决定输出什么值了。这个输出主要是依赖于 cell 状态 $C_t$，但是是经过筛选的版本。首先，经过一个 sigmoid 层，它决定$C_t$中的哪些部分将会被输出。接着，我们把$C_t$通过一个$tanh$层（把数值归一化到 - 1 和 1 之间），然后把$tanh$层的输出和 sigmoid 层计算出来的权重相乘，这样就得到了最后的输出结果

在语言模型例子中，假设我们的模型刚刚接触了一个代词，接下来可能要输出一个动词，这个输出可能就和代词的信息有关了。比如说，这个动词应该采用单数形式还是复数形式，那么我们就得把刚学到的和代词相关的信息都加入到 cell 状态中来，才能够进行正确的预测
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223102107168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
# 3 GRU
介绍完LSTM的工作原理后，下面来看下门控循环单元GRU。GRU是RNN的另一类演化变种，与LSTM非常相似。GRU结构中去除了单元状态，而使用隐藏状态来传输信息。它只有两个门结构，分别是更新门和重置门。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230950846.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223113444881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 更新⻔：更新门的作用类似于LSTM中的遗忘门和输入门，它能决定要丢弃哪些信息和要添加哪些新信息。
- 重置⻔：重置门用于决定丢弃先前信息的程度。
# 4 Seq2Seq
## 4.1 架构
在Seq2Seq结构中，编码器Encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由解码器Decoder解码。在解码器Decoder解码的过程中，不断地将前一个时刻 $t-1$ 的输出作为后一个时刻 $t$ 的输入，循环解码，直到输出停止符为止。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223120202969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
与经典RNN结构不同的是，Seq2Seq结构不再要求输入和输出序列有相同的时间长度！


![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021211440027.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
## 4.2 Attention
在Seq2Seq结构中，encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由Decoder解码。由于context包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个Context可能存不下那么多信息，就会造成精度的下降。除此之外，如果按照上述方式实现，只用到了编码器的最后一个隐藏层状态，信息利用率低下。

所以如果要改进Seq2Seq结构，最好的切入角度就是：利用Encoder所有隐藏层状态解决Context长度限制问题。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212122122116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- **Step1 计算Encoder的隐藏状态和Decoder的隐藏状态**
  首先计算第一个解码器隐藏状态（红色）和所有可用的编码器隐藏状态（绿色）。下图中有4个编码器隐藏状态和当前解码器的隐藏状态。要想输出Decoder的第一个隐藏的状态，需要给Decoder一个初始状态和一个输入，例如采用Encoder的最后一个状态作为Decoder的初始状态，输入为0。
- **Step2 获取每个编码器隐藏状态对应的分数**
  计算Decoder的第一个隐藏状态和Encoder所有的隐藏状态的相关性，这里采用点积的方式（默认两个向量长度一样）。
- **Step3 通过softmax归一化分数**
  我们把得到的分数输入到softmax层进行归一化，归一化之后的分数(标量)加起来等于1，归一化后的分数代表注意力分配的权重 。
- **Step4 用每个编码器的隐藏状态乘以其softmax得分**
  通过将每个编码器的隐藏状态与其softmax之后的分数(标量)相乘，我们得到对齐向量 或标注向量。这正是对齐产生的机制。
- **Step5 把所有对齐的向量加起来**
  对齐向量进行求和，生成上下文向量（语义编码）。上下文向量是前一步对齐向量的聚合信息。
- **Step6 将上下文向量输入到Decoder中**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223121648339.gif#pic_center#pic_center)
# 5 ELMo

ELMo 是 Embedding from Language Model 的缩写，它通过无监督的方式对语言模型进行预训练来学习单词表示

它的思路是用深度的双向 Language Model 在大量未标注数据上训练语言模型，如下图所示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224115831822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/202102241159324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
答案是全都要！
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022411595527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)