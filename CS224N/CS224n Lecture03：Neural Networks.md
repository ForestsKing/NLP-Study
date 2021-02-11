# Lecture Plan
- Classification review/introduction
- Neural networks introduction
- Named Entity Recognition
- Binary true vs. corrupted word window classification
- Matrix calculus introduction

提示：这对⼀些⼈⽽⾔将是困难的⼀周，课后需要阅读提供的资料。
# Classification setup and notation
通常我们有由样本组成的训练数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150306246.png#pic_center)

$x_i$是输⼊，例如单词（索引或是向量），句⼦，⽂档等等，维度为d
$y_i$是我们尝试预测的标签（ 个类别中的⼀个），例如：
- 类别：感情，命名实体，购买/售出的决定
- 其他单词
- 之后：多词序列的
## Classification intuition
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150403536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

训练数据：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150417476.png#pic_center)

简单的说明情况
- 固定的⼆维单词向量分类
- 使⽤softmax/logistic回归
- 线性决策边界

**传统的机器学习/统计学⽅法**：假设$x_i$ 是固定的，训练 softmax/logistic 回归的权重$W \in R^{C\times d}$来决定决定边界(超平⾯)
**⽅法**：对每个$x$ ，预测
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150609550.png#pic_center)

我们可以将预测函数分为两个步骤：
1. 将$W$ 的$y^{th}$ ⾏和 $x$中的对应⾏相乘得到分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150708173.png#pic_center)

计算所有的$f_c, for\ c=1,…,C$
2. 使⽤softmax函数获得归⼀化的概率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150757656.png#pic_center)

## Training with softmax and cross-entropy loss
对于每个训练样本$(x,y)$ ，我们的⽬标是最⼤化正确类$y$的概率，或者我们可以最⼩化该类的负对数概率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150834455.png#pic_center)

## Background: What is “cross entropy” loss/error?
- 交叉熵”的概念来源于信息论，衡量两个分布之间的差异
- 令真实概率分布为$p$
- 令我们计算的模型概率为$q$
- 交叉熵为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211150925523.png#pic_center)

- 假设 groud truth (or true or gold or target)的概率分布在正确的类上为1，在其他任何地⽅为0：p = [0,…，0,1，0，…，0]
- 因为$p$是独热向量，所以唯⼀剩下的项是真实类的负对数概率
## Classification over a full dataset
在整个数据集$\{x_i,y_i\}_{i=1}^N$ 上的交叉熵损失函数，是所有样本的交叉熵的均值
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151105623.png#pic_center)

我们不使⽤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151118766.png#pic_center)

我们使⽤矩阵来表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151127600.png#pic_center)

## Traditional ML optimization
- ⼀般机器学习的参数$\theta$通常只由W的列组成
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151232304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 因此，我们只通过以下⽅式更新决策边界
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021115124214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# Neural Network Classifiers
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151256466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 单独使⽤Softmax(≈logistic回归)并不⼗分强⼤
- Softmax只给出线性决策边界
  - 这可能是相当有限的，当问题很复杂时是⽆⽤的
  - 纠正这些错误不是很酷吗?
## Neural Nets for the Win!
神经⽹络可以学习更复杂的函数和⾮线性决策边界
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151330898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

更⾼级的分类需要
- 词向量
- 更深层次的深层神经⽹络
## Classification difference with word vectors
⼀般在NLP深度学习中
- 我们学习了矩阵$W$和词向量$x$
- 我们学习传统参数和表示
- 词向量是对独热向量的重新表示——在中间层向量空间中移动它们——以便使⽤(线性)softmax分类器通过 x = Le 层进⾏分类
  - 即将词向量理解为⼀层神经⽹络，输⼊单词的独热向量并获得单词的词向量表示，并且我们需要对其进⾏更新。其中，$Vd$是数量很⼤的参数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151451178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Neural computation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151509701.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## An artificial neuron
- 神经⽹络有⾃⼰的术语包
- 但如果你了解 softmax 模型是如何⼯作的，那么你就可以很容易地理解神经元的操作
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151553519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## A neuron can be a binary logistic regression unit
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151644151.png#pic_center)


b : 我们可以有⼀个“总是打开”的特性，它给出⼀个先验类，或者将它作为⼀个偏向项分离出来
w, b是神经元的参数
## A neural network = running several logistic regressions at the same time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151717714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

如果我们输⼊⼀个向量通过⼀系列逻辑回归函数，那么我们得到⼀个输出向量，但是我们不需要提前决定这些逻辑回归试图预测的变量是什么。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151741239.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

我们可以输⼊另⼀个logistic回归函数。损失函数将指导中间隐藏变量应该是什么，以便更好地预测下⼀层的⽬标。我们当然可以使⽤更多层的神经⽹络。
## Matrix notation for a layer
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151813303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151824826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- $f(x)$ 在运算时是 element-wise 逐元素的
## Non-linearities (aka “f ”): Why they’re needed
例如：函数近似，如回归或分类
- 没有⾮线性，深度神经⽹络只能做线性变换
- 多个线性变换可以组成⼀个的线性变换$W_1W_2x=Wx$
  - 因为线性变换是以某种⽅式旋转和拉伸空间，多次的旋转和拉伸可以融合为⼀次线性变换
- 对于⾮线性函数⽽⾔，使⽤更多的层，他们可以近似更复杂的函数
# Named Entity Recognition (NER)
- 任务：例如，查找和分类⽂本中的名称
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211151950331.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 可能的⽤途
  - 跟踪⽂档中提到的特定实体（组织、个⼈、地点、歌曲名、电影名等）
  - 对于问题回答，答案通常是命名实体
  - 许多需要的信息实际上是命名实体之间的关联
  - 同样的技术可以扩展到其他 slot-filling 槽填充 分类
- 通常后⾯是命名实体链接/规范化到知识库
## Named Entity Recognition on word sequences
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152021348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

我们通过在上下⽂中对单词进⾏分类，然后将实体提取为单词⼦序列来预测实体
## Why might NER be hard?
- 很难计算出实体的边界
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152058842.png#pic_center)

  - 第⼀个实体是 “First National Bank” 还是 “National Bank”
- 很难知道某物是否是⼀个实体
  - 是⼀所名为“Future School” 的学校，还是这是⼀所未来的学校？
- 很难知道未知/新奇实体的类别
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021115212692.png#pic_center)

  - “Zig Ziglar” ? ⼀个⼈
- 实体类是模糊的，依赖于上下⽂
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152152633.png#pic_center)

  - 这⾥的“Charles Schwab” 是 PER 不是 ORG
# Binary word window classification
为在上下⽂中的语⾔构建分类器
- ⼀般来说，很少对单个单词进⾏分类
- 有趣的问题，如上下⽂歧义出现
- 例⼦：auto-antonyms
  - "To sanction" can mean "to permit" or "to punish”
  - "To seed" can mean "to place seeds" or "to remove seeds"
- 例⼦：解决模糊命名实体的链接
  - Paris → Paris, France vs. Paris Hilton vs. Paris, Texas
  - Hathaway → Berkshire Hathaway vs. Anne Hathaway
## Window classification
- 思想：在相邻词的上下⽂窗⼝中对⼀个词进⾏分类
- 例如，上下⽂中⼀个单词的命名实体分类
  - ⼈、地点、组织、没有
- 在上下⽂中对单词进⾏分类的⼀个简单⽅法可能是对窗⼝中的单词向量进⾏平均，并对平均向量进⾏分类
  - 问题：这会丢失位置信息
## Window classification: Softmax
- 训练softmax分类器对中⼼词进⾏分类，⽅法是在⼀个窗⼝内将中⼼词周围的词向量串联起来
- 例⼦：在这句话的上下⽂中对“Paris”进⾏分类，窗⼝⻓度为2
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152340760.png#pic_center)

- 结果向量$x_{window}=x \in R^{5d}$ 是⼀个列向量
## Simplest window classifier: Softmax
对于 $x=x_{window}$，我们可以使⽤与之前相同的softmax分类器
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152447409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 如何更新向量？
- 简⽽⾔之：就像上周那样，求导和优化
## Binary classification with unnormalized scores
- 之前的例⼦：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152516127.png#pic_center)

- 假设我们要对中⼼词是否为⼀个地点，进⾏分类
- 与word2vec类似，我们将遍历语料库中的所有位置。但这⼀次，它将受到监督，只有⼀些位置能够得到⾼分。
- 例如，在他们的中⼼有⼀个实际的NER Location的位置是“真实的”位置会获得⾼分
## Binary classification for NER Location
- 例⼦：Not all museums in Paris are amazing
- 这⾥：⼀个真正的窗⼝，以Paris为中⼼的窗⼝和所有其他窗⼝都“损坏”了，因为它们的中⼼没有指定的实体位置。
  - museums in Paris are amazing
- “损坏”窗⼝很容易找到，⽽且有很多：任何中⼼词没有在我们的语料库中明确标记为NER位置的窗⼝
  - Not all museums in Paris
## Neural Network Feed-forward Computation
使⽤神经激活 $a$ 简单地给出⼀个⾮标准化的分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152635773.png#pic_center)

我们⽤⼀个三层神经⽹络计算⼀个窗⼝的得分

- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152749795.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152801236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Main intuition for extra layer
中间层学习输⼊词向量之间的⾮线性交互
例如：只有当“museum”是第⼀个向量时，“in”放在第⼆个位置才重要
## The max-margin loss
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211152826370.png#pic_center)

- 关于训练⽬标的想法：让真实窗⼝的得分更⾼，⽽破坏窗⼝的得分更低(直到⾜够好为⽌)
- $s = score("museum in Paris are amazing")$
- $s_c = score("Not all museum in Paris")$
- 最⼩化$J=max(0,1-s+s_c)$
- 这是不可微的，但它是连续的→我们可以⽤SGD。
- 每个选项都是连续的
- 单窗⼝的⽬标函数为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153038397.png#pic_center)

- 每个中⼼有NER位置的窗⼝的得分应该⽐中⼼没有位置的窗⼝⾼1分
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153055749.png#pic_center)

- 要获得完整的⽬标函数：为每个真窗⼝采样⼏个损坏的窗⼝。对所有培训窗⼝求和
- 类似于word2vec中的负抽样
- 使⽤SGD更新参数
  - $\theta^{new}=\theta^{old}-\alpha\nabla_{\theta}J(\theta)$
  - $\alpha$是 步⻓或是学习率
- 如何计算$\nabla_{\theta}J(\theta)$ ？
  - ⼿⼯计算（本课）
  - 算法：反向传播（下⼀课）
## Computing Gradients by Hand
- 回顾多元导数
- 矩阵微积分：完全⽮量化的梯度
  - ⽐⾮⽮量梯度快得多，也更有⽤
  - 但做⼀个⾮⽮量梯度可以是⼀个很好的实践；以上周的讲座为例
  - notes 更详细地涵盖了这些材料
# Gradients
给定⼀个函数，有1个输出和1个输⼊
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153402508.png#pic_center)

斜率是它的导数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153411653.png#pic_center)

给定⼀个函数，有1个输出和 n 个输⼊
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153420583.png#pic_center)

梯度是关于每个输⼊的偏导数的向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153429741.png#pic_center)

## Jacobian Matrix: Generalization of the Gradient
- 给定⼀个函数，有 m 个输出和 n 个输⼊
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153454699.png#pic_center)

- 其雅可⽐矩阵是⼀个$m \times n$ 的偏导矩阵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153523242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Chain Rule
对于单变量函数：乘以导数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153545512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

对于⼀次处理多个变量：乘以雅可⽐矩阵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153601833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Example Jacobian: Elementwise activation Function
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021115363182.png#pic_center)

由于使⽤的是 element-wise，所以$h_i=f(z_i)$
函数有n个输出和n个输⼊ → n×n 的雅可⽐矩阵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153706405.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Other Jacobians
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153727532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

这是正确的雅可⽐矩阵。稍后我们将讨论“形状约定”；⽤它则答案是 $h$。
## Back to our Neural Net!
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153758159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

如何计算$\frac{\partial s}{\partial b}$ ？
实际上，我们关⼼的是损失的梯度，但是为了简单起⻅，我们将计算分数的梯度
## Break up equations into simple pieces
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153854992.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Apply the chain rule
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153916210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211153933303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

如何计算 $\frac{\partial s}{\partial W}$ ？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154012349.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

前两项是重复的，⽆须重复计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154025563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

其中， $\delta$是局部误差符号
## Derivative with respect to Matrix: Output shape
- $W \in R^{n \times m}$， $\frac{\partial s}{\partial W}$的形状是
- 1个输出，$n \times m$ 个输⼊：$1 \times nm $的雅可⽐矩阵？
  - 不⽅便更新参数$\theta^{new}=\theta^{old}-\alpha\nabla_{\theta}J(\theta)$
- ⽽是遵循惯例：导数的形状是参数的形状 （形状约定）
  - $\frac{\partial s}{\partial W}$的形状是$n \times m$
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154317480.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Derivative with respect to Matrix
- $\frac{\partial s}{\partial W}=\delta\frac{\partial z}{\partial W}$
  - $\delta$将出现在我们的答案中
  - 另⼀项应该是$x$ ，因为$z=Wx+b$
- 这表明$\frac{\partial s}{\partial W}=\delta^Tx^T$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154554897.png#pic_center)

## Why the Transposes?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154608188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 粗糙的回答是：这样就可以解决尺⼨问题了
  - 检查⼯作的有⽤技巧
- 课堂讲稿中有完整的解释
  - 每个输⼊到每个输出——你得到的是外部积
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211154636119.png#pic_center)

## What shape should derivatives be?
- $\frac{\partial s}{\partial b}=h^T \circ f^'(z)$是⾏向量
  - 但是习惯上说梯度应该是⼀个列向量因为 b是⼀个列向量
- 雅可⽐矩阵形式(这使得链式法则很容易)和形状约定(这使得SGD很容易实现)之间的分歧
  - 我们希望答案遵循形状约定
  - 但是雅可⽐矩阵形式对于计算答案很有⽤
- 两个选择
  - 尽量使⽤雅可⽐矩阵形式，最后按照约定进⾏整形
    - 我们刚刚做的。但最后转置 $\frac{\partial s}{\partial b}$使导数成为列向量，得到$\delta^T$
  - 始终遵循惯例
    - 查看维度，找出何时转置 和/或 重新排序项。
反向传播
- 算法⾼效地计算梯度
- 将我们刚刚⼿⼯完成的转换成算法
- ⽤于深度学习软件框架(TensorFlow, PyTorch, Chainer, etc.)