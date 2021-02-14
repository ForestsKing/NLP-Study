# Lecture Plan
- Motivation: Compositionality and Recursion
- Structure prediction with simple Tree RNN: Parsing
- Backpropagation through Structure
- More complex TreeRNN units (35 mins)
- Other uses of tree-recursive neural nets (5 mins)
- Institute for Human-Centered Artificial Intelligence
# 1. The spectrum of language in CS
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212205541168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 词袋模型 和 复杂形式的语⾔表达结构
## Semantic interpretation of language – Not just word vectors
我们怎样才能弄清楚更⼤的短语的含义？
- The snowboarder is leaping over a mogul
- A person on a snowboard jumps into the air

The snowboarder 在语义上相当于 A person on a snowboard，但它们的字⻓不⼀样
- ⼈们之所以可以理解 A person on a snowboard ，是因为 the principle of compositionality组合原则
- ⼈们知道每个单词的意思，从⽽知道了 on a snowboard 的意思
- 知道组件的含义并将他们组合成为更⼤的组件

⼈们通过较⼩元素的语义成分来解释较⼤⽂本单元的意义 - 实体，描述性术语，事实，论点，故事
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212205948169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212205959476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210006403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 语⾔理解 - 和⼈⼯智能 - 需要能够通过了解较⼩的部分来理解更⼤的事物
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221002243.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 我们拥有将较⼩的部分组合起来制作出更⼤东⻄的能⼒
## Are languages recursive?
- 认知上有点争议（需要前往⽆限）
- 但是：递归对于描述语⾔是很⾃然的
  - [The person standing next to [the man from [the company that purchased [the firm that you used to work at]]]]
  - 包含名词短语的名词短语，包含名词短语
- 它是语⾔结构的⼀个⾮常强⼤的先验
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210056149.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210103767.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 2. Building on Word Vector Space Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210116407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- the country of my birth
- the place where I was born
- 我们怎样表示更⻓短语的意思呢？
- 通过将他们映射到相同的向量空间！
## How should we map phrases into a vector space?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210135312.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 基于组合原则，使⽤单词的含义和组合他们的规则，得到⼀个句⼦的含义向量
- 同时学习解析树以及组合向量表示
## Constituency Sentence Parsing: What we want
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210151881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Learn Structure and Representation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210205277.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 我们需要能够学习如何解析出正确的语法结构，并学习如何基于语法结构，来构建句⼦的向量表示
## Recursive vs. recurrent neural networks
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210219169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 递归神经⽹络需要⼀个树结构
- 循环神经⽹络不能在没有前缀上下⽂的情况下捕捉短语，并且经常在最终的向量中过度捕捉最后⼀个单词
## Recursive Neural Networks for Structure Prediction
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210236926.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

如果我们⾃上⽽下的⼯作，那么我们在底层有单词向量，所以我们想要递归地计算更⼤成分的含义

输⼊：两个候选的⼦节点的表示
输出：
  - 两个节点被合并后的语义表示
  - 新节点的合理程度
## Recursive Neural Network Definition
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221030297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Parsing a sentence with an RNN (greedily)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210312423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Parsing a sentence
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210322567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210329506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⾃左向右重复遍历，每次将得分最⾼的两者组合在⼀起
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210340668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Max-Margin Framework - Details
- 树的得分是通过每个节点的解析决策得分的总和来计算的：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210402826.png#pic_center)

- x 是句⼦，y 是解析树
- 类似于最⼤边距解析（Taskar et al.2004），⼀个受监督的最⼤边际⽬标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210443515.png#pic_center)

- 损失 $\Delta(y,y_i)$惩罚所有不正确的决策
- 结构搜索 $A(x)$是贪婪的（每次加⼊最佳节点）
  - 相反：使⽤ Beam search 搜索图
## Scene Parsing
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210615342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

和组合类似的准则
- 场景图像的含义也是较⼩区域的函数
- 它们如何组合成部分以形成更⼤的对象
- 以及对象如何相互作⽤
## Algorithm for Parsing Images
Same Recursive Neural Network as for natural language parsing! (Socher et al. ICML 2011)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210637233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Multi-class segmentation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210648483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 3. Backpropagation Through Structure
Introduced by Goller & Küchler (1996)

和通⽤的反向传播的规则相同

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210727769.png#pic_center)

- 递归和树结构导致的计算：
  - 从所有节点（如RNN）求和W的导数
  - 在每个节点处拆分导数（对于树）
  - 从⽗节点和节点本身添加错误消息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210752975.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210759103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221080635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210810994.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210817692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Discussion: Simple TreeRNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210828539.png#pic_center)

- 使⽤单矩阵TreeRNN的结果
- 单个权重矩阵TreeRNN可以捕获⼀些现象但不适合更复杂的现象以及更⾼阶的构成和解析⻓句
- 输⼊词之间没有真正的交互
- 组合函数对于所有句法类别，标点符号等都是相同的
# 4. Version 2: Syntactically-Untied RNN
[Socher, Bauer, Manning, Ng 2013]
- 符号的上下⽂⽆关的语法（Context Free Grammar CFG）主⼲是⾜以满⾜基本的句法结构
- 我们使⽤⼦元素的离散句法类别来选择组合矩阵
- 对于不同的语法环境，TreeRNN可以针对不同的组合矩阵做得更好
- 结果为我们提供了更好的语义
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212210857681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 左边 RNN 是使⽤概率的上下⽂⽆关的语法解析，为句⼦⽣成可能的树结构
- 右边 SU-RNN 是语法解开的递归神经⽹络，此时每个节点和序列都有⼀个类别，我们可以使⽤对应不同类别的矩阵组合起来，例如将类别 B 和类别 C 的矩阵组合起来作为本次计算的权重矩阵，所以这个权重矩阵是更符合句⼦结构的
## Compositional Vector Grammars
- 问题：速度。Beam search 中的每个候选分数都需要⼀次矩阵向量乘法
- 解决⽅案：仅针对来⾃更简单，更快速模型(Probabilistic Context Free Grammar (PCFG))的树的⼦集计算得分
  - 修剪⾮常不可能的速度候选⼈
  - 为每个 beam 候选者提供⼦句的粗略语法类别
- 组合⽮量语法= PCFG + TreeRNN
## Related Work for parsing
- 产⽣的 CVG Parser 与以前扩展PCFG解析器的⼯作有关
- Klein and Manning (2003a)：⼿⼯特征⼯程
- Petrov et al. (2006)：分解和合并句法类别的学习算法
- 词汇化解析器(Collins, 2003; Charniak, 2000)：⽤词汇项描述每个类别
- Hall and Klein (2012) 在⼀个因式解析器中结合了⼏个这样的注释⽅案
- CVGs 将这些想法从离散表示扩展到更丰富的连续表达
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211006266.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## SU-RNN / CVG [Socher, Bauer, Manning, Ng 2013]
Learns soft notion of head words
初始化：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211035272.png#pic_center)

- 初始化为⼀对对⻆矩阵
- 学习的是⼀个短语中哪个⼦节点是重要的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211045369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211054347.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Analysis of resulting vector representations
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211104130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 所有数据均根据季节变化进⾏调整
  - 所有数字都根据季节性波动进⾏调整
  - 调整所有数字以消除通常的季节性模式
- Knight-Ridder不会评论这个提议
  - Harsco declined to say what country placed the order
  - Coastal wouldn’t disclose the terms
- Sales grew almost 7% to $UNK m. from $UNK m.
  - Sales rose more than 7% to 94.0 m. from 88.3 m.
  - Sales surged 40% to UNK b. yen from UNK b.
## Version 3: Compositionality Through Recursive Matrix-Vector Spaces
[Socher, Huval, Bhat, Manning, & Ng, 2012]
- 之前：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211353764.png#pic_center)

  - $c_1$ 和 $c_2$ 之间并没有相互影响
- 使组合函数更强⼤的⼀种⽅法是解开权重 W
- 但是，如果单词主要作为运算符，例如 “very” in “very good”，是没有意义的，是⽤于增加 good 的规模的运算符
- 提案：新的组合函数
- 问题是如何定义呢，因为不知道 $c_1$ 和 $c_2$ 哪个是 operator，⽐如 very good ，就应该讲 very 视为作⽤在 good 的矩阵上的向量
## Compositionality Through Recursive Matrix-Vector Recursive Neural Networks
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211447129.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 每个单词都拥有⼀个向量意义和⼀个矩阵意义
## Matrix-vector RNNs
[Socher, Huval, Bhat, Manning, & Ng, 2012]
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221150253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 左侧计算得到合并后的向量意义
- 右侧计算得到合并后的矩阵意义
- 可以捕获运算符语义，即中⼀个单词修饰了另⼀个单词的含义
## Predicting Sentiment Distributions
语⾔中⾮线性的好例⼦
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211517786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Classification of Semantic Relationships
- MV-RNN 可以学习到⼤的句法上下⽂传达语义关系吗？
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211535283.png#pic_center)

- 为包括两项的最⼩成分构建单个组合语义

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211543291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 问题：参数量过⼤，并且获得短语的矩阵意义的⽅式不够好
## Version 4: Recursive Neural Tensor Network
Socher, Perelygin, Wu, Chuang, Manning, Ng, and Potts 2013
- ⽐ MV-RNN 更少的参数量
- 允许两个单词或短语向量乘法交互
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211602322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Beyond the bag of words: Sentiment detection
⼀段⽂字的语调是积极的，消极的还是中性的？
- 某种程度上情绪分析是容易的
- 较⻓⽂档的检测精度~90％，但是
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211617201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Stanford Sentiment Treebank
- 215,154 phrases labeled in 11,855 sentences
- 可以真的训练和测试组合
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211630752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

http://nlp.stanford.edu:8080/sentiment/
## Better Dataset Helped All Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211648340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 严重的否定的案例仍然⼤多判断错误的
- 我们还需要⼀个更强⼤的模型！

想法：允许载体的加性和介导的乘法相互作⽤
- 在树中使⽤结果向量作为逻辑回归的分类器的输⼊
- 使⽤梯度下降联合训练所有权重
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211705954.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211711962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 回到最初的使⽤向量表示单词的意义，但不是仅仅将两个表示单词含义的向量相互作⽤，左上图是在中间插⼊⼀个矩阵，以双线性的⽅式做注意⼒并得到了注意⼒得分。即令两个单词的向量相互作⽤并且只产⽣⼀个数字作为输出
- 如上中图所示，我们可以拥有三维矩阵，即多层的矩阵（⼆维），从⽽得到了两个得分
- 使⽤ softmax 做分类
## Positive/Negative Results on Treebank
Classifying Sentences: Accuracy improves to 85.4
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211735238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Experimental Results on Treebank
- RNTN 可以捕捉类似 X but Y 的结构
- RNTN accuracy of 72%, compared to MV-RNN (65%), biword NB (58%) and RNN (54%)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211752790.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Negation Results
双重否定时，积极反应应该上升
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211803375.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Demo: http://nlp.stanford.edu:8080/sentiment/
## Version 5: Improving Deep Learning Semantic Representations using a TreeLSTM
[Tai et al., ACL 2015; also Zhu et al. ICML 2015]
⽬标：
- 仍试图将句⼦的含义表示为（⾼维，连续）向量空间中的位置
- ⼀种准确处理语义构成和句⼦含义的⽅式
- 将⼴泛使⽤的链式结构LSTM推⼴到树结构
## Long Short-Term Memory (LSTM) Units for Sequential Composition
⻔是 $[0, 1]^d$的向量，⽤于逐元素乘积的软掩蔽元素
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211846725.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Tree-Structured Long Short-Term Memory Networks
[Tai et al., ACL 2015]
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221190577.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Tree-structured LSTM

将连续LSTM推⼴到具有任何分⽀因⼦的树
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211917997.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211925139.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Results: Sentiment Analysis: Stanford Sentiment Treebank
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211934942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221194197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Results: Semantic Relatedness SICK 2014 (Sentences Involving Compositional Knowledge)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212211950618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Forget Gates: Selective State Preservation
Stripes = forget gate activations; more white 㱺 more preserved
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212003873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 5. QCD-Aware Recursive Neural Networks for Jet Physics
Gilles Louppe, Kyunghun Cho, Cyril Becot, Kyle Cranmer (2017)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212014407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Tree-to-tree Neural Networks for Program Translation
[Chen, Liu, and Song NeurIPS 2018]
- 探索在编程语⾔之间使⽤树形结构编码和⽣成进⾏翻译
- 在⽣成中，将注意⼒集中在源树上
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212029891.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212034801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212038720.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212045510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Human-Centered Artificial Intelligence
- ⼈⼯智能有望改变经济和社会，改变我们沟通和⼯作的⽅式，重塑治理和政治，并挑战国际秩序
- HAI的使命是推进⼈⼯智能研究，教育，政策和实践，以改善⼈类状况

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212212058594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
