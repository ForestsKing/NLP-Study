## Lecture Plan
上节课我们学了递归神经⽹络(RNNs) 以及为什么它们对于语⾔建模(LM)很有⽤。今天我们将学习
- RNNs的 问题 以及如何修复它们
- 更复杂的 RNN变体
下⼀节课我们将学习
- 如何使⽤基于 RNN-based 的体系结构，即 sequence-to-sequence with attention 来实现 神经机器翻译 (NMT)
## Today’s lecture
- 梯度消失问题 $rightarrow$ 两种新类型RNN：LSTM和GRU
- 其他梯度消失（爆炸）的解决⽅案
  - Gradient clipping
  - Skip connections
- 更多花哨的RNN变体
  - 双向RNN
  - 多层RNN
## Vanishing gradient intuition
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211225416169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 当这些梯度很⼩的时候，反向传播的越深⼊，梯度信号就会变得越来越⼩
## Vanishing gradient proof sketch
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211225447176.png#pic_center)

- 因此通过链式法则得到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211225459311.png#pic_center)

- 考虑第 $i$ 步上的损失梯度$J^{(i)}(\theta)$ ，相对于第 $j$ 步上的隐藏状态$h^{(j)}$
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122565174.png#pic_center)

如果权重矩阵$W_h$ 很⼩，那么这⼀项也会随着 $i$ 和 $j$ 的距离越来越远⽽变得越来越⼩
- 考虑矩阵的 L2 范数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211225730805.png#pic_center)

- Pascanu et al 表明，如果 $W_h$ 的 最⼤特征值 < 1 ，梯度$|| \frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}}||$ 将呈指数衰减
  - 这⾥的界限是1因为我们使⽤的⾮线性函数是 sigmoid
- 有⼀个类似的证明将⼀个 最⼤的特征值 >1 与 梯度爆炸 联系起来
## Why is vanishing gradient a problem?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230013276.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 来⾃远处的梯度信号会丢失，因为它⽐来⾃近处的梯度信号⼩得多。
- 因此，模型权重只会根据近期效应⽽不是⻓期效应进⾏更新。
- 另⼀种解释 ：梯度 可以被看作是 过去对未来的影响 的衡量标准
- 如果梯度在较⻓⼀段距离内(从时间步 t 到 t+n )变得越来越⼩，那么我们就不能判断:
  - 在数据中，步骤 t 和 t+n 之间没有依赖关系
  - 我们⽤ 错误的参数 来捕获 t 和 t+n 之间的真正依赖关系
## Effect of vanishing gradient on RNN-LM
- 语⾔模型任务
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230055754.png#pic_center)

- 为了从这个训练示例中学习，RNN-LM需要对第7步的“tickets”和最后的⽬标单词“tickets”之间的依赖关系建模。
- 但是如果梯度很⼩，模型就 不能学习这种依赖关系
  - 因此模型⽆法在测试时 预测类似的⻓距离依赖关系
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230124104.png#pic_center)

- Correct answer : The writer of the books is planning a sequel
- 语法近因
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230142529.png#pic_center)

- 顺序近因
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230156828.png#pic_center)

- 由于梯度的消失，RNN-LMs更善于从 顺序近因 学习⽽不是 语法近因 ，所以他们犯这种错误的频率⽐我们希望的要⾼[Linzen et al . 2016]
## Why is exploding gradient a problem?
- 如果梯度过⼤，则SGD更新步骤过⼤
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021123023272.png#pic_center)

- 这可能导致 错误的更新 ：我们更新的太多，导致错误的参数配置(损失很⼤)
- 在最坏的情况下，这将导致⽹络中的 Inf 或 NaN (然后您必须从较早的检查点重新启动训练)
## Gradient clipping: solution for exploding gradient
- 梯度裁剪 ：如果梯度的范数⼤于某个阈值，在应⽤SGD更新之前将其缩⼩
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021123025943.png#pic_center)

- 直觉 ：朝着同样的⽅向迈出⼀步，但要⼩⼀点
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230315852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 这显示了⼀个简单RNN的损失⾯(隐藏层状态是⼀个标量不是⼀个向量)
-  “悬崖”是危险的，因为它有陡坡
- 在左边，由于陡坡，梯度下降有两个⾮常⼤的步骤，导致攀登悬崖然后向右射击(都是胡浩的更新)
- 在右边，梯度剪裁减少了这些步骤的⼤⼩,所以效果不太激烈
## How to fix vanishing gradient problem?
- 主要问题是RNN很难学习在多个时间步⻓的情况下保存信息
- 在普通的RNN中，隐藏状态不断被重写
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230408419.png#pic_center)

- ⼀个具有独⽴记忆的RNN怎么样？
## Long Short-Term Memory (LSTM)
- Hochreiter和Schmidhuber在1997年提出了⼀种RNN，⽤于解决梯度消失问题。
- 在第 t 步，有⼀个隐藏状态 $h^{(t)}$ 和⼀个单元状态 $c^{(t)}$
  - 都是⻓度为 n 的向量
  - 单元存储⻓期信息
  - LSTM可以从单元格中删除、写⼊和读取信息
- 信息被 擦除 / 写⼊ / 读取 的选择由三个对应的⻔控制
  - ⻔也是⻓度为 n 的向量
  - 在每个时间步⻓上，⻔的每个元素可以打开(1)、关闭(0)或介于两者之间
  - ⻔是动态的：它们的值是基于当前上下⽂计算的
我们有⼀个输⼊序列 $x^{(t)}$，我们将计算⼀个隐藏状态 $h^{(t)}$ 和单元状态 $c^{(t)}$ 的序列。在时间步 t 时
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230600378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 遗忘⻔：控制上⼀个单元状态的保存与遗忘
- 输⼊⻔：控制写⼊单元格的新单元内容的哪些部分
- 输出⻔：控制单元的哪些内容输出到隐藏状态
- 新单元内容：这是要写⼊单元的新内容
- 单元状态：删除(“忘记”)上次单元状态中的⼀些内容，并写⼊(“输⼊”)⼀些新的单元内容
- 隐藏状态：从单元中读取(“output”)⼀些内容
- Sigmoid函数：所有的⻔的值都在0到1之间
- 通过逐元素的乘积来应⽤⻔
- 这些是⻓度相同的向量

你可以把LSTM⽅程想象成这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021123065989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## How does LSTM solve vanishing gradients?
- RNN的LSTM架构更容易保存许多时间步上的信息
  - 如果忘记⻔设置为记得每⼀时间步上的所有信息，那么单元中的信息被⽆限地保存
  - 相⽐之下，普通RNN更难学习重复使⽤并且在隐藏状态中保存信息的矩阵$W_h$
- LSTM并不保证没有消失/爆炸梯度，但它确实为模型提供了⼀种更容易的⽅法来学习远程依赖关系
## LSTMs: real-world success
- 2013-2015年，LSTM开始实现最先进的结果
  - 成功的任务包括：⼿写识别、语⾳识别、机器翻译、解析、图像字幕
  - LSTM成为主导⽅法
- 现在(2019年)，其他⽅法(如Transformers)在某些任务上变得更加主导。
  - 例如在WMT (a MT conference + competition)中 
  - 在2016年WMT中，总结报告包含“RNN”44次 
  - 在2018年WMT中，总结报告包含“RNN”9次，“Transformers” 63次
## Gated Recurrent Units (GRU)
- Cho等⼈在2014年提出了LSTM的⼀个更简单的替代⽅案
- 在每个时间步 $t$ 上，我们都有输⼊ $x^{(t)}$ 和隐藏状态 $h^{(t)}$ (没有单元状态)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211230950846.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 更新⻔：控制隐藏状态的哪些部分被更新，哪些部分被保留
- 重置⻔：控制之前隐藏状态的哪些部分被⽤于计算新内容
- 新的隐藏状态内容：重置⻔选择之前隐藏状态的有⽤部分。使⽤这⼀部分和当前输⼊来计算新的隐藏状态内容
- 隐藏状态：更新⻔同时控制从以前的隐藏状态保留的内容，以及更新到新的隐藏状态内容的内容
- 这如何解决消失梯度？
  - 与LSTM类似，GRU使⻓期保存信息变得更容易(例如，将update gate设置为0)
## LSTM vs GRU
- 研究⼈员提出了许多⻔控RNN变体，其中LSTM和GRU的应⽤最为⼴泛
- 最⼤的区别是GRU计算速度更快，参数更少
- 没有确凿的证据表明其中⼀个总是⽐另⼀个表现得更好
- LSTM是⼀个很好的默认选择(特别是当您的数据具有⾮常⻓的依赖关系，或者您有很多训练数据时)
- 经验法则：从LSTM开始，但是如果你想要更有效率，就切换到GRU
## Is vanishing/exploding gradient just a RNN problem?
- 不！这对于所有的神经结构(包括前馈和卷积)都是⼀个问题，尤其是对于深度结构
  - 由于链式法则/选择⾮线性函数，反向传播时梯度可以变得很⼩很⼩
  - 因此，较低层次的学习⾮常缓慢(难以训练)
  - 解决⽅案：⼤量新的深层前馈 / 卷积架构，添加更多的直接连接(从⽽使梯度可以流动)
例如：
- Residual connections 残差连接⼜名“ResNet”
- 也称为跳转连接
- 默认情况下，标识连接保存信息
- 这使得深层⽹络更容易训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231127112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例如：
- Dense connections 密集连接⼜名“DenseNet”
- 直接将所有内容连接到所有内容
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231146580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例如：
- Highway connections ⾼速公路连接⼜称“⾼速公路⽹”
- 类似于剩余连接，但标识连接与转换层由动态⻔控制
- 灵感来⾃LSTMs，但适⽤于深度前馈/卷积⽹络
结论 ：虽然消失/爆炸梯度是⼀个普遍的问题，但由于重复乘以相同的权矩阵，RNN尤其不稳定[Bengio et al, 1994]
## Recap
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231221254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Bidirectional RNNs: motivation
Task: Sentiment Classification

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231240324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 我们可以把这种隐藏状态看作是这个句⼦中单词“terribly”的⼀种表示。我们称之为上下⽂表示。
- 这些上下⽂表示只包含关于左上下⽂的信息(例如“the movie was”)。
- 那么正确的上下⽂呢?
- 在这个例⼦中，“exciting”在右上下⽂中，它修饰了“terribly”的意思(从否定变为肯定)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231315182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- “terribly”的上下⽂表示同时具有左上下⽂和右上下⽂

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231339545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 这是⼀个表示“计算RNN的⼀个向前步骤”的通⽤符号——它可以是普通的、LSTM或GRU计算。
- 我们认为这是⼀个双向RNN的“隐藏状态”。这就是我们传递给⽹络下⼀部分的东⻄。
- ⼀般来说，这两个RNNs有各⾃的权重
## Bidirectional RNNs: simplified diagram
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231402415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 双向箭头表示双向性，所描述的隐藏状态是正向+反向状态的连接
- 注意：双向RNNs只适⽤于访问整个输⼊序列的情况
  - 它们不适⽤于语⾔建模，因为在LM中，您只剩下可⽤的上下⽂
- 如果你有完整的输⼊序列(例如任何⼀种编码)，双向性是强⼤的(默认情况下你应该使⽤它)
- 例如，BERT(来⾃transformer的双向编码器表示)是⼀个基于双向性的强⼤的预训练的上下⽂表示系统
  - 你会在课程的后⾯学到更多关于BERT的知识!
## Multi-layer RNNs
- RNNs在⼀个维度上已经是“深的”(它们展开到许多时间步⻓)
- 我们还可以通过应⽤多个RNNs使它们“深⼊”到另⼀个维度——这是⼀个多层RNN
  - 较低的RNNs应该计算较低级别的特性，⽽较⾼的RNNs应该计算较⾼级别的特性
- 多层RNNs也称为堆叠RNNs。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231455481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

RNN层 $i$ 的隐藏状态是RNN层 $i+1$ 的输⼊
## Multi-layer RNNs in practice
- ⾼性能的RNNs通常是多层的(但没有卷积或前馈⽹络那么深)
- 例如：在2017年的⼀篇论⽂，Britz et al 发现在神经机器翻译中，2到4层RNN编码器是最好的,和4 层RNN解码器
  - Keyphrases: RNN⽆法并⾏化，计算代价过⼤，所以不会过深
- Transformer-based 的⽹络(如BERT)可以多达24层。他们有很多skipping-like的连接
## In summary
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211231609378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)