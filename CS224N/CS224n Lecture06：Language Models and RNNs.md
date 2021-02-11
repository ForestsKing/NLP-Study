# Overview
- 介绍⼀个新的NLP任务
  - Language Modeling (motivate RNNs)
- 介绍⼀个新的神经⽹络家族
  - Recurrent Neural Networks (RNNs)
# Language Modeling
- 语⾔建模的任务是预测下⼀个单词是什么。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211220631170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 更正式的说法是：给定⼀个单词序列 $x^{(1)},x^{(2)},…,x^{(t)}$，计算下⼀个单词 $x^{(t+1)}$的概率分布
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122083455.png#pic_center)

- 其中，$x^{(t+1)}$ 可以是词表中的任意单词$V=\{w_1,…,w_{|V|}\}$
- 这样做的系统称为 Language Model 语⾔模型
- 还可以将语⾔模型看作是⼀个将概率分配给⼀段⽂本的系统
- 例如，如果我们有⼀段⽂本$x^{(1)},x^{(2)},…,x^{(T)}$ 则这段⽂本的概率(根据语⾔模型)为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221034516.png#pic_center)

- 语⾔模型提供的是
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221113493.png#pic_center)

## n-gram Language Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221144901.png#pic_center)

- 问题 ：如何学习⼀个语⾔模型？
- 回答 (pre-DeepLearning)：学习⼀个 n-gram 语⾔模型
- 定义 ：n-gram 是 ⼀个由n个连续单词组成的块
  - unigrams: “the”, “students”, “opened”, ”their”
  - bigrams: “the students”, “students opened”, “opened their”
  - trigrams: “the students opened”, “students opened their”
  - 4-grams: “the students opened their”
- 想法 ：收集关于不同n-gram出现频率的统计数据，并使⽤这些数据预测下⼀个单词。
- ⾸先，我们做⼀个 简化假设 ： $x^{(t+1)}$只依赖于前⾯的n-1个单词
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122125280.png#pic_center)

具体含义如下图所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221304920.png#pic_center)

- 问题 ：如何得到n-gram和(n-1)-gram的概率？
- 回答 ：通过在⼀些⼤型⽂本语料库中计算它们(统计近似)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221331400.png#pic_center)

假设我们正在学习⼀个 4-gram 的语⾔模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221355783.png#pic_center)

例如，假设在语料库中：
- “students opened their” 出现了1000次
- “students opened their books” 出现了400次
  - P(books | students opened their) = 0.4
- “students opened their exams” 出现了100次
  - - P(exams | students opened their) = 0.1
- 我们应该忽视上下⽂中的“proctor”吗？
  - 在本例中，上下⽂⾥出现了“proctor”，所以exams在这⾥的上下⽂中应该是⽐books概率更⼤的。
## Sparsity Problems with n-gram Language Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221540521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 问题 ：如果“students open their $w$” 从未出现在数据中，那么概率值为 0
- (Partial)解决⽅案 ：为每个$w \in V$ 添加极⼩数 $\delta$ 。这叫做平滑。这使得词表中的每个单词都⾄少有很⼩的概率。
- 问题 ：如果“students open their” 从未出现在数据中，那么我们将⽆法计算任何单词 $w$的概率值
- (Partial)解决⽅案 ：将条件改为“open their”。这叫做后退。
> Note: n 的增加使稀疏性问题变得更糟。⼀般情况下 n 不能⼤于5。

## Storage Problems with n-gram Language Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221749978.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

增加 n 或增加语料库都会增加模型⼤⼩
## n-gram Language Models in practice
- 你可以在你的笔记本电脑上，在⼏秒钟内建⽴⼀个超过170万个单词库(Reuters)的简单的三元组语⾔模型
- Reuters 是 商业和⾦融新闻的数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221818940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

**稀疏性问题** ：概率分布的粒度不⼤。“today the company” 和 “today the bank”都是 $\frac{4}{26}$，都只出现过四次
## Generating text with a n-gram Language Model
- 还可以使⽤语⾔模型来⽣成⽂本
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122191494.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221925505.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211221935827.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

使⽤trigram运⾏以上⽣成过程时，会得到如下⽂本
>today the price of gold per ton , while production of shoe lasts and shoe
industry , the bank intervened just after it considered and rejected an imf
demand to rebuild depleted european stocks , sept 30 end primary 76 cts a
share .

令⼈惊讶的是其具有语法但是是不连贯的。如果我们想要很好地模拟语⾔，我们需要同时考虑三个以上的单词。但增加 n 使模型的稀疏性问题恶化，模型尺⼨增⼤。
## How to build a neural Language Model?
- 回忆⼀下语⾔模型任务
  - 输⼊：单词序列$x^{(1)},x^{(2)},…,x^{(t)}$
  - 输出：下⼀个单词的概率分布$P(x^{(t+1)} | x^{(t)}, …, x^{(1)})$
window-based neural model 在第三讲中被⽤于NER问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222705269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## A fixed-window neural Language Model
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222722959.png#pic_center)

使⽤和NER问题中同样⽹络结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122273717.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

超越 n-gram 语⾔模型的 改进
- 没有稀疏性问题
- 不需要观察到所有的n-grams
存在的问题
- 固定窗⼝太⼩
- 扩⼤窗⼝就需要扩⼤权重矩阵 W
- 窗⼝再⼤也不够⽤
- $x^{(1)}$ 和 $x^{(2)}$ 乘以完全不同的权重。输⼊的处理 不对称。
我们需要⼀个神经结构，可以处理任何⻓度的输⼊
# Recurrent Neural Networks (RNN)
核⼼想法：重复使⽤ **相同** 的权重矩阵 W
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222908171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211222920929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

RNN的 优点
- 可以处理 任意⻓度 的输⼊
- 步骤 t 的计算(理论上)可以使⽤ 许多步骤前 的信息
- 模型⼤⼩不会 随着输⼊的增加⽽增加
- 在每个时间步上应⽤相同的权重，因此在处理输⼊时具有 对称性

RNN的 缺点
- 递归计算速度 慢
- 在实践中，很难从 许多步骤前返回信息
- 后⾯的课程中会详细介绍
## Training a RNN Language Model
- 获取⼀个较⼤的⽂本语料库，该语料库是⼀个单词序列
- 输⼊RNN-LM；计算每个步骤 t 的输出分布
  - 即预测到⽬前为⽌给定的每个单词的概率分布
- 步骤 t 上的损失函数为预测概率分布$y^{(t)*}$与真实下⼀个单词$y^{(t)}$ ( $x^{(t+1)}$的独热向量)之间的交叉熵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223155360.png#pic_center)

- 将其平均，得到整个培训集的 总体损失
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223210629.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223226546.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223236701.png#pic_center)

- 然⽽：计算 整个语料库 $x^{(1)},…,x^{(T)}$的损失和梯度太昂贵了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223328949.png#pic_center)

- 在实践中，我们通常将$x^{(1)},…,x^{(T)}$ 看做⼀个 句⼦ 或是 ⽂档
- 回忆 ：随机梯度下降允许我们计算⼩块数据的损失和梯度，并进⾏更新。
- 计算⼀个句⼦的损失 $J(\theta)$(实际上是⼀批句⼦)，计算梯度和更新权重。重复上述操作。
## Backpropagation for RNNs
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122342651.png#pic_center)

问题 ：关于 重复的 $W_h$权重矩阵 的偏导数$J^{(T)}(\theta)$
回答 ：重复权重的梯度是每次其出现时的梯度的总和
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223518584.png#pic_center)

## Multivariable Chain Rule
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223536521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

对于⼀个多变量函数 $f(x,y)$和两个单变量函数$x(t)$ 和 $y(t)$，其链式法则如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223615372.png#pic_center)

## Backpropagation for RNNs: Proof sketch
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122363783.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Backpropagation for RNNs
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223655206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 问题 ： 如何计算？
- 回答 ：反向传播的时间步⻓ $i=t,…,0$。累加梯度。这个算法叫做“backpropagation through time”
## Generating text with a RNN Language Model
就像n-gram语⾔模型⼀样，您可以使⽤RNN语⾔模型通过 重复采样 来 ⽣成⽂本 。采样输出是下⼀步的输⼊。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021122374939.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 相⽐n-gram更流畅，语法正确，但总体上仍然很不连贯
- ⻝谱的例⼦中，⽣成的⽂本并没有记住⽂本的主题是什么
- 哈利波特的例⼦中，甚⾄有体现出了⼈物的特点，并且引号的开闭也没有出现问题
  - 也许某些神经元或者隐藏状态在跟踪模型的输出是否在引号中
- RNN是否可以和⼿⼯规则结合？
  - 例如Beam Serach，但是可能很难做到
## Evaluating Language Models
- 标准语⾔模型评估指标是 perplexity 困惑度
- 这等于交叉熵损失 $J(\theta)$ 的指数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211223852182.png#pic_center)

- 低困惑度是更好的
## RNNs have greatly improved perplexity
## Why should we care about Language Modeling?
- 语⾔模型是⼀项 基准测试 任务，它帮助我们 衡量 我们在理解语⾔⽅⾯的 进展
  - ⽣成下⼀个单词，需要语法，句法，逻辑，推理，现实世界的知识等
- 语⾔建模是许多NLP任务的 ⼦组件，尤其是那些涉及 ⽣成⽂本 或 估计⽂本概率 的任务
  - 预测性打字
  - 语⾳识别
  - ⼿写识别
  - 拼写/语法纠正
  - 作者识别
  - 机器翻译
  - 摘要
  - 对话
  - 等等
## Recap
- 语⾔模型： 预测下⼀个单词 的系统
- 递归神经⽹络：⼀系列神经⽹络
  - 采⽤任意⻓度的顺序输⼊
  - 在每⼀步上应⽤相同的权重
  - 可以选择在每⼀步上⽣成输出
- 递归神经⽹络 语⾔模型
- 我们已经证明，RNNs是构建LM的⼀个很好的⽅法。
- 但RNNs的⽤处要⼤得多!
## RNNs can be used for tagging
e.g. part-of-speech tagging, named entity recognition
## RNNs can be used for sentence classification
e.g. sentiment classification
如何计算句⼦编码
- 使⽤最终隐层状态
- 使⽤所有隐层状态的逐元素最值或均值
## RNNs can be used as an encoder module
e.g. question answering, machine translation, many other tasks!
Encoder的结构在NLP中⾮常常⻅
## RNN-LMs can be used to generate text
e.g. speech recognition, machine translation, summarization
这是⼀个条件语⾔模型的示例。我们使⽤语⾔模型组件，并且最关键的是，我们根据条件来调整它
稍后我们会更详细地看到机器翻译。
## A note on terminology
本课提到的RNN是 ““vanilla RNN”
下节课将会学习GRU和LSTM以及多层RNN
本课程结束时，你会理解类似“stacked bidirectional LSTM with residual connections and self-attention”的短语