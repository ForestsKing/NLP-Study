# Lecture Plan
- Syntactic Structure: Consistency and Dependency
- Dependency Grammar and Treebanks
- Transition-based dependency parsing
- Neural dependency parsing
> Dependency Parsing: 依存关系语法分析，简称 依存分析。

# 1. Two views of linguistic structure
## Constituency Parsing
**Constituency = phrase structure grammar = context-free grammars (CFGs)**
> context-free grammars (CFGs) ⽆上下⽂语法

句⼦是使⽤逐步嵌套的单元构建的
- 短语结构将单词组织成嵌套的成分
- 起步单元：单词被赋予⼀个类别 (part of speech=pos 词性)
- 单词组合成不同类别的短语
- 短语可以递归地组合成更⼤的短语
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211180435232.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

上图中
- Det 指的是 Determiner，在语⾔学中的含义为 限定词
- NP 指的是 Noun Phrase ，在语⾔学中的含义为 名词短语
- VP 指的是 Verb Phrase ，在语⾔学中的含义为 动词短语
- P 指的是 Preposition ，在语⾔学中的含义为 介词
  - PP 指的是 Prepositional Phrase ，在语⾔学中的含义为 介词短语

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211180635254.png#pic_center)


- Example : The cat by the large crate on the large table by the door

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211180605801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

但是另⼀种观点在计算语⾔学中占主导地位。
## Dependency Parsing
不是使⽤各种类型的短语，⽽是直接通过单词与其他的单词关系表示句⼦的结构，显示哪些单词依赖于(修饰或是其参数)哪些其他单词

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211180705770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- look 是整个句⼦的根源，look  依赖于 crate （或者说 crate 是 look 的依赖）
  - in, the, large 都是 crate 的依赖
  - in the kitchen 是 crate 的修饰
  - in, the 都是 kitchen 的依赖
  - by the door 是 crate 的依赖
## Why do we need sentence structure?
- 为了能够正确地解释语⾔，我们需要理解句⼦结构
- ⼈类通过将单词组合成更⼤的单元来传达复杂的意思，从⽽交流复杂的思想
- 我们需要知道什么与什么相关联
  - 除⾮我们知道哪些词是其他词的参数或修饰词，否则我们⽆法弄清楚句⼦是什么意思
## Prepositional phrase attachment ambiguity
介词短语依附歧义
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021118100622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 警察⽤⼑杀了那个男⼦
  - cops 是 kill 的 subject (subject 指 主语)
  - man 是 kill 的 object (object 指 宾语)
  - knife 是 kill 的 modifier (modifier 指 修饰符)
- 警察杀了那个有⼑的男⼦
  - knife 是 man 的 modifier (名词修饰符，简称为 nmod )

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181106317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

from space 这⼀介词短语修饰的是前⾯的动词 count 还是名词 whales ？
这就是⼈类语⾔和编程语⾔中不同的地⽅
- 关键的解析决策是我们如何 “依存” 各种成分
  - 介词短语、状语或分词短语、不定式、协调等。

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181132659.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 上述句⼦中有四个介词短语
- board 是 approved 的 主语， acquisition 是 approved 的谓语
- by Royal Trustco Ltd. 是修饰 acquisition 的，即董事会批准了这家公司的收购
- of Toronto 可以修饰 approved, acquisition, Royal Trustco Ltd. 之⼀，经过分析可以得知是修饰 Royal Trustco Ltd. 即表示这家公司的位置
- for $27 a share 修饰 acquisition
- at its monthly meeting 修饰 approved ，即表示批准的时间地点
⾯对这样复杂的句⼦结构，我们需要考虑 指数级 的可能结构，这个序列被称为 Catalan numbers
![Catalan numbers :](https://img-blog.csdnimg.cn/20210211181218814.png#pic_center)

- ⼀个指数增⻓的序列，出现在许多类似树的环境中
  - 例如，⼀个 n+2 边的多边形可能的三⻆剖分的数量
    - 出现在概率图形模型的三⻆剖分中(CS228)
## Coordination scope ambiguity
协调范围模糊
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181310300.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例句：Shuttle veteran and longtime NASA executive Fred Gregory appointed to board
- ⼀个⼈：[[Shuttle veteran and longtime NASA executive] Fred Gregory] appointed to board
- 两个⼈：[Shuttle veteran] and [longtime NASA executive Fred Gregory] appointed to board
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181323642.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例句：Doctor: No heart, cognitive issues
## Adjectival Modifier Ambiguity
形容词修饰语歧义
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181357430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例句：Students get first hand job experience
- first hand 表示 第⼀⼿的，直接的，即学⽣获得了直接的⼯作经验
  - first 是 hand 的形容词修饰语(amod)
- first 修饰 experience, hand 修饰 job 

## Verb Phrase (VP) attachment ambiguity
动词短语(VP)依存歧义
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181429348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

例句：Mutilated body washes up on Rio beach to be used for Olympic beach volleyball.
- to be used for Olympic beach volleyball 是 动词短语 (VP)
- 修饰的是 body 还是 beach
## Dependency paths identify semantic relations –e.g., for protein interaction
依赖路径识别语义关系
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181502906.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 2. Dependency Grammar and Dependency Structure
关联语法假设句法结构包括词汇项之间的关系，通常是⼆元不对称关系(“箭头”)，称为依赖关系
Dependency Structure有两种表现形式
- ⼀种是直接在句⼦上标出依存关系箭头及语法关系
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021118152784.png#pic_center)

- 另⼀种是将其做成树状机构（Dependency Tree Graph）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181542343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 箭头通常标记(type)为语法关系的名称(主题、介词对象、apposition等)。
- 依赖关系标签的系统，例如 universal dependency 通⽤依赖
- 箭头连接头部(head)(调速器,上级,regent)和⼀个依赖(修饰词,下级,下属)
  - A 依赖于 A 的事情
- 通常,依赖关系形成⼀棵树(单头 ⽆环 连接图)
## Dependency Grammar/Parsing History
- 依赖结构的概念可以追溯到很久以前
  - Pāṇini的语法(公元前5世纪)
  - ⼀千年 阿拉伯语的语法的基本⽅法
- 选区/上下⽂⽆关⽂法是⼀个新奇的发明
  - 20世纪发明(R.S.Wells,1947; then Chomsky)
- 现代依赖⼯作经常源于 L. Tesnière(1959)
  - 是20世纪“东⽅”的主导⽅法(俄罗斯，中国，…)
    - 有利于更⾃由的语序语⾔
- NLP中最早类型的解析器在美国
  - David Hays 是美国计算语⾔学的创始⼈之⼀，他很早就(第⼀个?)构建了依赖解析器(Hays 1962)。
## Dependency Grammar and Dependency Structure
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181649700.png#pic_center)

- ⼈们对箭头指向的⽅式不⼀致：有些⼈把箭头朝⼀个⽅向画；有⼈是反过来的
  - Tesnière 从头开始指向依赖，本课使⽤此种⽅式
- 通常添加⼀个伪根指向整个句⼦的头部，这样每个单词都精确地依赖于另⼀个节点
## The rise of annotated data: Universal Dependencies treebanks
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181720777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

**Universal Dependencies：**我们想要拥有⼀个统⼀的、并⾏的依赖描述，可⽤于任何⼈类语⾔
- 从前⼿⼯编写语法然后训练得到可以解析句⼦的解析器
- ⽤⼀条规则捕捉很多东⻄真的很有效率，但是事实证明这在实践中不是⼀个好主意
  - 语法规则符号越来越复杂，并且没有共享和重⽤⼈类所做的⼯作
- 句⼦结构上的treebanks ⽀持结构更有效
## The rise of annotated data
从⼀开始，构建 treebank 似乎⽐构建语法慢得多，也没有那么有⽤
但是 treebank 给我们提供了许多东⻄
- 劳动⼒的可重⽤性
  - 许多解析器、词性标记器等可以构建在它之上
  - 语⾔学的宝贵资源
- ⼴泛的覆盖⾯，⽽不仅仅是⼀些直觉
- 频率和分布信息
- ⼀种评估系统的⽅法
## Dependency Conditioning Preferences
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181838276.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

依赖项解析的信息来源是什么？
1. Bilexical affinities (两个单词间的密切关系)
  - [discussion issues] 是看上去有道理的
2. Dependency distance 依赖距离
  - 主要是与相邻词
3. Intervening material 介于中间的物质
  - 依赖很少跨越介于中间的动词或标点符号
4. Valency of heads
  - How many dependents on which side are usual for a head?
## Dependency Parsing
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211181913703.png#pic_center)

- 通过为每个单词选择它所依赖的其他单词(包括根)来解析⼀个句⼦
- 通常有⼀些限制
  - 只有⼀个单词是依赖于根的
  - 不存在循环
- 这使得依赖项成为树
- 最后⼀个问题是箭头是否可以交叉(⾮投影的 non-projective)
  - 没有交叉的就是non-projectice
## Projectivity
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021118194869.png#pic_center)

- 定义：当单词按线性顺序排列时，没有交叉的依赖弧，所有的弧都在单词的上⽅
- 与CFG树并⾏的依赖关系必须是投影的
  - 通过将每个类别的⼀个⼦类别作为头来形成依赖关系
- 但是依赖理论通常允许⾮投射结构来解释移位的成分
  - 如果没有这些⾮投射依赖关系，就不可能很容易获得某些结构的语义
## Methods of Dependency Parsing
1. Dynamic programming
  - Eisner(1996)提出了⼀种复杂度为 的聪明算法，它⽣成头部位于末尾⽽不是中间的解析项
2. Graph algorithms
  - 为⼀个句⼦创建⼀个最⼩⽣成树
  - McDonald et al.’s (2005) MSTParser 使⽤ML分类器独⽴地对依赖项进⾏评分(他使⽤MIRA进⾏在线学习，但它也可以是其他东⻄)
3. Constraint Satisfaction
  - 去掉不满⾜硬约束的边 Karlsson(1990), etc.
4. “Transition-based parsing” or “deterministic dependency parsing“
  - 良好的机器学习分类器 MaltParser(Nivreet al. 2008) 指导下的依存贪婪选择。已证明⾮常有效。
# 3. Greedy transition-based parsing [Nivre 2003]
- 贪婪判别依赖解析器 greedy discriminative dependency parser 的⼀种简单形式
- 解析器执⾏⼀系列⾃底向上的操作
  - ⼤致类似于shift-reduce解析器中的“shift”或“reduce”，但“reduce”操作专⻔⽤于创建头在左或右的依赖项
- 解析器如下：
  - 栈 $\sigma$以 ROOT 符号开始，由若⼲$w_i$ 组成
  - 缓存 $\beta$以输⼊序列开始，由若⼲$w_i$ 组成
  - ⼀个依存弧的集合$A$ ，⼀开始为空。每条边的形式是$(w_i,r,w_j)$ ，其中$r$ 描述了节点的依存关系
  - ⼀组操作
- 最终⽬标是 $\sigma=[ROOT], \ \beta=\emptyset$， $A$包含了所有的依存弧
## Basic transition-based dependency parser
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211182500553.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

state之间的transition有三类：
- SHIFT：将buffer中的第⼀个词移出并放到stack上。
- LEFT-ARC：将$(w_j,r,w_i)$ 加⼊边的集合 A ，其中$w_i$是stack上的次顶层的词，$w_j$是stack上的最顶层的词。
- RIGHT-ARC:将 $(w_i,r,w_j)$加⼊边的集合 A ，其中 $w_i$是stack上的次顶层的词， $w_j$是stack上的最顶层的词。
我们不断的进⾏上述三类操作，直到从初始态达到最终态。

在每个状态下如何选择哪种操作呢？当我们考虑到LEFT-ARC与RIGHT-ARC各有 $|R|$（ $|R|$为$r$ 的类的个数）种类，我们可以将其看做是class数为 $2|R|+1$的分类问题，可以⽤SVM等传统机器学习⽅法解决。
## Arc-standard transition-based parser
> 还有其他的 transition ⽅案

Analysis of “I ate fish”
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211182813648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102111828311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## MaltParser [Nivreand Hall 2005]
- 我们需要解释如何选择下⼀步⾏动
  - Answer：机器学习
- 每个动作都由⼀个有区别分类器(例如softmax classifier)对每个合法的移动进⾏预测
  - 最多三种⽆类型的选择，当带有类型时，最多 $2|R|+1$ 种
  - Features：栈顶单词，POS；buffer中的第⼀个单词，POS；等等
- 在最简单的形式中是没有搜索的
  - 但是，如果你愿意，你可以有效地执⾏⼀个 Beam search 束搜索(虽然速度较慢，但效果更好)：你可以在每个时间步骤中保留 k个好的解析前缀
- 该模型的精度略低于依赖解析的最⾼⽔平，但它提供了⾮常快的线性时间解析，性能⾮常好
## Conventional Feature Representation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211182951618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

传统的特征表示使⽤⼆元的稀疏向量$10^6~10^7$ 
- 特征模板：通常由配置中的1 ~ 3个元素组成
- Indicator features
## Evaluation of Dependency Parsing: (labeled) dependency accuracy
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211183041688.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

其中，**UAS** (unlabeled attachment score) 指 ⽆标记依存正确率 ，**LAS** (labeled attachment score) 指有标记依存正确率
## Handling non-projectivity
- 我们提出的弧标准算法只构建投影依赖树
- 头部可能的⽅向
  - 在⾮投影弧上宣布失败
  - 只具有投影表示时使⽤依赖形式
    - CFG只允许投影结构
  - 使⽤投影依赖项解析算法的后处理器来识别和解析⾮投影链接
  - 添加额外的转换，⾄少可以对⼤多数⾮投影结构建模（添加⼀个额外的交换转换，冒泡排序）
  - 转移到不使⽤或不需要对投射性进⾏任何约束的解析机制(例如，基于图的MSTParser)
# 4. Why train a neural dependency parser? Indicator Features Revisited
Indicator Features的问题
- 稀疏
- 不完整
- 计算复杂
  - 超过95%的解析时间都⽤于特征计算
## A neural dependency parser [Chen and Manning 2014]
斯坦福依赖关系的英语解析
- Unlabeled attachment score (UAS) = head
- Labeled attachment score (LAS) = head and label
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211183224855.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 效果好，速度快
## Distributed Representations
- 我们将每个单词表示为⼀个d维稠密向量（如词向量）
  - 相似的单词应该有相近的向量
- 同时，part-of-speech tags 词性标签(POS)和 dependency labels 依赖标签也表示为d维向量
  - 较⼩的离散集也表现出许多语义上的相似性。
- NNS(复数名词)应该接近NN(单数名词)
- num(数值修饰语)应该接近amod(形容词修饰语)。
对于Neural Dependency Parser，其输⼊特征通常包含三种
- stack和buffer中的单词及其dependent word
- 单词的part-of-speech tag
- 描述语法关系的arc label
我们根据堆栈/缓冲区位置提取⼀组令牌:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211183314986.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

我们将其转换为词向量并将它们联结起来作为输⼊层，再经过若⼲⾮线性的隐藏层，最后加⼊softmax layer得到shift-reduce解析器的动作
## Model Architecture
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102111833373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Dependency parsing for sentence structure
神经⽹络可以准确地确定句⼦的结构，⽀持解释
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021118335290.png#pic_center)

Chen and Manning(2014)是第⼀个简单，成功的神经依赖解析器
密集的表示使得它在精度和速度上都优于其他贪婪的解析器
## Further developments in transition-based neural dependency parsing
这项⼯作由其他⼈进⼀步开发和改进，特别是在⾕歌
- 更⼤、更深的⽹络中，具有更好调优的超参数
- Beam Search 更多的探索动作序列的可能性，⽽不是只考虑当前的最优
- 全局、条件随机场(CRF)的推理出决策序列
这就引出了SyntaxNet和Parsey McParseFace模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211183423642.png#pic_center)

## Graph-based dependency parsers
- 为每条边的每⼀个可能的依赖关系计算⼀个分数
  - 然后将每个单词的边缘添加到其得分最⾼的候选头部
  - 并对每个单词重复相同的操作
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211183453375.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021118350595.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## A Neural graph-based dependency parser [Dozat and Manning 2017; Dozat, Qi, and Manning 2017]
- 在神经模型中为基于图的依赖分析注⼊活⼒
  - 为神经依赖分析设计⼀个双仿射评分模型
    - 也使⽤神经序列模型，我们将在下周讨论
- ⾮常棒的结果
  - 但是⽐简单的基于神经传递的解析器要慢
    - 在⼀个⻓度为 n 的句⼦中可能有 个依赖项
