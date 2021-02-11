# Lecture Plan
- Matrix gradients for our simple neural net and some tips
- Computation graphs and backpropagation
- Stuff you should know
  - Regularization to prevent overfitting
  - Vectorization
  - Nonlinearities
  - Initialization
  - Optimizers
  - Learning rates
# 1. Derivative wrt a weight matrix
- 让我们仔细看看计算$\frac{\partial s}{\partial W}$
  - 再次使⽤链式法则
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160203658.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Deriving gradients for backprop
- 这个函数(从上次开始)
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160240101.png#pic_center)

- 考虑单个权重$w_{ij}$的导数
- $W_{ij}$只对$z_i$有贡献
  - $W_{23}$例如$z_2$只对 有贡献，对$z_1$没有贡献
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160412294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160423437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 对于单个$W_{IJ}$的导数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160448895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 我们想要整个 W 的梯度，但是每种情况都是⼀样的
- 总体答案：外积
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160515859.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Deriving gradients: Tips
- 技巧1：仔细定义变量并跟踪它们的维度！
- 技巧2：链式法则！如果$y=f(u),u=g(x)$，即$y=f(g(x))$则
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160635964.png#pic_center)

  - 要清楚哪些变量⽤于哪些计算
- 提示3：模型的最上⾯的softmax部分：⾸先考虑当 $c=y$(正确的类)的导数$f_c$ ，然后再考虑当$c\neq y$(所有不正确的类)的导数$f_c$
- 技巧4：如果你被矩阵微积分搞糊涂了，请计算逐个元素的偏导数！
- 技巧5：使⽤形状约定。注意：到达隐藏层的错误消息 具有与该隐藏层相同的维度
## Deriving gradients wrt words for window model
- 到达并更新单词向量的梯度可以简单地分解为每个单词向量的梯度
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160835427.png#pic_center)

- 则得到
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211160858753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> 我们将根据梯度逐个更新对应的词向量矩阵中的词向量，所以实际上是对词向量矩阵的更新是⾮常稀疏的

## Updating word gradients in window model
- 当我们将梯度更新到词向量中时，这将推动单词向量，使它们(在原则上)在确定命名实体时更有帮助。
- 例如，模型可以了解到，当看到$x_{in}$是中⼼词之前的单词时，指示中⼼词是⼀个 Location
## A pitfall when retraining word vectors
- 背景：我们正在训练⼀个单词电影评论情绪的逻辑回归分类模型。
- 在训练数据中，我们有“TV”和“telly”
- 在测试数据中我们有“television””
- 预先训练的单词向量有三个相似之处：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161013397.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 问题：当我们更新向量时会发⽣什么
- 回答：
  - 那些在训练数据中出现的单词会四处移动
    - “TV”和“telly”
  - 没有包含在训练数据中的词汇保持原样
    - “television”
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161103470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## So what should I do?
- 问题：应该使⽤可⽤的“预训练”字向量吗？
- 回答：
  - ⼏乎总应该是⽤
  - 他们接受了⼤量的数据训练，所以他们会知道训练数据中没有的单词，也会知道更多关于训练数据中的单词
  - 拥有上亿字的数据吗？好的，随机开始
- 问题：我应该更新(“fine tune”)我⾃⼰的单词向量吗？
- 回答：
  - 如果你只有⼀个⼩的训练数据集，不要训练词向量
  - 如果您有⼀个⼤型数据集，那么 train = update = fine-tune 词向量到任务可能会更好
## Backpropagation
我们⼏乎已经向你们展示了反向传播
- 求导并使⽤(⼴义)链式法则
另⼀个技巧：在计算较低层的导数时，我们重⽤对较⾼层计算的导数，以使计算最⼩化
# 2. Computation Graphs and Backpropagation
我们把神经⽹络⽅程表示成⼀个图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161226984.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161237601.png#pic_center)

## Forward Propagation
- 源节点：输⼊
- 内部节点：操作
- 边传递操作的结果
## Back Propagation
- 沿着边回传梯度
## Backpropagation: Single Node
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161326183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161335428.png#pic_center)

- 节点接收“上游梯度”
- ⽬标是传递正确的“下游梯度”
- 每个节点都有局部梯度 local gradient
  - 它输出的梯度是与它的输⼊有关
- [downstream gradient] = [upstream gradient] x [local gradient]
## 有多个输⼊的节点呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161421822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161431120.png#pic_center)

- 多个输⼊$\rightarrow$多个局部梯度
## An Example
### Forward
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161516808.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

### Backward
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161531740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161549336.png#pic_center)

## Gradients sum at outward branches
上图中的$\frac{\partial f}{\partial y}$的梯度的计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161639646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161649246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Node Intuitions
 - $+$“分发” 上游梯度给每个$summand$
 - $max$“路由” 上游梯度，将梯度发送到最⼤的⽅向
 - $*$“切换”上游梯度
## Efficiency: compute all gradients at once
- 不重复计算梯度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161838372.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161849811.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Back-Prop in General Computation Graph
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211161912382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

1. Fprop：按拓扑排序顺序访问节点
  - 计算给定⽗节点的节点的值
2. Bprop：
  - 初始化输出梯度为 1
  - 以相反的顺序⽅位节点，使⽤节点的后继的梯度来计算每个节点的梯度
  - ![是 的后继](https://img-blog.csdnimg.cn/2021021116201513.png#pic_center)

  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211162024140.png#pic_center)

  - 正确地说，Fprop 和 Bprop 的计算复杂度是⼀样的
  - ⼀般来说，我们的⽹络有固定的层结构，所以我们可以使⽤矩阵和雅可⽐矩阵
## Automatic Differentiation
- 梯度计算可以从 Fprop 的符号表达式中⾃动推断
- 每个节点类型需要知道如何计算其输出，以及如何在给定其输出的梯度后计算其输⼊的梯度
- 现代DL框架(Tensorflow, Pytoch)为您做反向传播，但主要是令作者⼿⼯计算层/节点的局部导数
## Backprop Implementations
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211162117136.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211162140813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

为了计算反向传播，我们需要在前向传播时存储⼀些变量的值
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211162144868.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Gradient checking: Numeric Gradient
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211162212690.png#pic_center)

- 易于正确实现
- 但近似且⾮常缓慢
  - 必须对模型的每个参数重新计算$f$
- ⽤于检查您的实现
  - 在过去我们⼿写所有东⻄的时候，在任何地⽅都这样做是关键。
  - 现在，当把图层放在⼀起时，就不需要那么多了
## Summary
- 我们已经掌握了神经⽹络的核⼼技术
- 反向传播：沿计算图递归应⽤链式法则
  - [downstream gradient] = [upstream gradient] x [local gradient]
- 前向传递：计算操作结果并保存中间值
- 反向传递：应⽤链式法则计算梯度
## Why learn all these details about gradients?
- 现代深度学习框架为您计算梯度
- 但是，当编译器或系统为您实现时，为什么要学习它们呢？
  - 了解引擎盖下发⽣了什么是有⽤的
- 反向传播并不总是完美地⼯作
  - 理解为什么对调试和改进模型⾄关重要
- 未来课程的例⼦:爆炸和消失的梯度
# 3. We have models with many params! Regularization!
- 实际上⼀个完整的损失函数包含了所有参数$\theta$的正则化（下式中最后⼀项），例如L2正则化：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211164729333.png#pic_center)

- 正则化(在很⼤程度上)可以防⽌在我们有很多特征时过拟合(或者是⼀个⾮常强⼤/深层的模型等等)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211164812268.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## “Vectorization”
- 例如，对单词向量进⾏循环，⽽不是将它们全部连接到⼀个⼤矩阵中，然后将softmax权值与该矩阵相乘
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211164834622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 1000 loops, best of 3: 639 μs per loop
- 10000 loops, best of 3: 53.8 μs per loop
- (10x)更快的⽅法是使⽤$C \times N$矩阵
- 总是尝试使⽤向量和矩阵，⽽不是循环
- 你也应该快速测试你的代码
- 简单来说：矩阵太棒了
## Non-linearities: The starting points
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211164937304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

tanh 只是⼀个重新放缩和移动的 sigmoid (两倍陡峭，[-1,1])
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211164955396.png#pic_center)

logistic 和 tanh 仍然被⽤于特定的⽤途，但不再是构建深度⽹络的默认值。
> logistic和tanh: 设计复杂的数学运算，指数计算会减慢速度。所以⼈们提出了 hard tanh，并且效果很不错。于是才有了 ReLU

## Non-linearities: The new world order
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211165033799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 为了建⽴⼀个前馈深度⽹络，你应该做的第⼀件事是ReLU——由于良好的梯度回流，训练速度快，性能好
- 每个单元要么已经死了，要么在传递信息。
- ⾮零范围内只有⼀个斜率，这⼀位置梯度⼗分有效的传递给了输⼊，所以模型⾮常有效的训练
## Parameter Initialization
- 通常 必须将权重初始化为⼩的随机值 （这样才能在激活函数的有效范围内， 即存在梯度可以使其更新）
  - 避免对称性妨碍学习/特殊化的
- 初始化隐含层偏差为0，如果权重为0，则输出(或重构)偏差为最优值(例如，均值⽬标或均值⽬标的反s形)
- 初始化 所有其他权重 为Uniform(–r, r)，选择使数字既不会太⼤也不会太⼩的 r
- Xavier初始化中，⽅差与 fan-in$n_{in}$ (前⼀层尺⼨)和 fan-out $n_{out}$(下⼀层尺⼨)成反⽐
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021116515169.png#pic_center)

## Optimizers
- 通常，简单的SGD就可以了
  - 然⽽，要得到好的结果通常需要⼿动调整学习速度(下⼀张幻灯⽚)
- 对于更复杂的⽹络和情况，或者只是为了避免担⼼，更有经验的复杂的 “⾃适应”优化器通常会令你做得更好，通过累积梯度缩放参数调整。
  - 这些模型给出了每个参数的学习速度
    - Adagrad
    - RMSprop
    - Adam 相当好,在许多情况下是安全的选择
    - SparseAdam
    - …
## Learning Rates
- 你可以⽤⼀个固定的学习速度。从lr = 0.001开始？
  - 它必须是数量级的——尝试10的幂
    - 太⼤：模型可能会发散或不收敛
    - 太⼩：你的模型可能训练不出很好的效果
- 如果你在训练时降低学习速度，通常可以获得更好的效果
  - ⼿⼯：每隔k个阶段(epoch)将学习速度减半
    - epoch = 遍历⼀次数据 (打乱或采样的)
    
- 通过⼀个公式：
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211165405538.png#pic_center)
  
- 还有更新奇的⽅法，⽐如循环学习率(q.v.)
- 更⾼级的优化器仍然使⽤学习率，但它可能是优化器缩⼩的初始速度——因此可能可以从较⾼的速度开始。