# Lecture Plan
- What is Coreference Resolution? (15 mins)
- Applications of coreference resolution (5 mins)
- Mention Detection (5 mins)
- Some Linguistics: Types of Reference (5 mins) Four Kinds of Coreference Resolution Models
- Rule-based (Hobbs Algorithm) (10 mins)
- Mention-pair models (10 mins)
- Mention ranking models (15 mins)
  - Including the current state-of-the-art coreference system!
- Mention clustering model (5 mins – only partial coverage)
- Evaluation and current results (10 mins)
# 1. What is Coreference Resolution?
- 识别所有涉及到相同现实世界实体的 提及
- He, her 都是实体的提及 mentions of entities
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231902929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231906954.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231912578.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231916468.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231921462.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Applications
- 全⽂理解
  - 信息提取, 回答问题, 总结, …
  - “他⽣于1961年”(谁?)
- 机器翻译
  - 语⾔对性别，数量等有不同的特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231948312.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213231955637.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 对话系统
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232006504.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Coreference Resolution in Two Steps
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232016883.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 3. Mention Detection
- Mention : 指向某个实体的⼀段⽂本
- 三种 mention
  - Pronouns 代词
    - I, your, it, she, him, etc.
    - 因为代词是 POS 检测结果的⼀种，所以只要使⽤ POS 检测器即可
  - Named entities 命名实体
    - People, places, etc.
    - Use a NER system (like hw3)
  - Noun phrases 名词短语
    - “a dog,” “the big fluffy cat stuck in the tree”
    - Use a parser (especially a 依存解析器 constituency parser – next week!)
- Marking all pronouns, named entities, and NPs as mentions over-generates mentions
- Are these mentions?
  - It is sunny
  - Every student
  - No student
  - The best donut in the world
  - 100 miles
## How to deal with these bad mentions?
- 可以训练⼀个分类器过滤掉 spurious mentions
- 更为常⻅的：保持所有 mentions 作为 “candidate mentions”
  - 在你的共指系统运⾏完成后，丢弃所有的单个引⽤(即没有被标记为与其他任何东⻄共同引⽤的)
## Can we avoid a pipelined system?
- 我们可以训练⼀个专⻔⽤于 mention 检测的分类器，⽽不是使⽤POS标记器、NER系统和解析器。
- 甚⾄端到端共同完成 mention 检测和共指解析，⽽不是两步
# 4. On to Coreference! First, some linguistics
- Coreference is when two mentions refer to the same entity in the world 当两个 mention 指向世界上的同⼀个实体时，被称为共指
  - Barack Obama 和 Obama
- 相关的语⾔概念是 anaphora 回指：when a term (anaphor) refers to another term(antecedent) 下⽂的词返指或代替上⽂的词
  - anaphor 的解释在某种程度上取决于 antecedent 先⾏词的解释
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232231694.png#pic_center)

## Anaphora vs Coreference

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232240138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Not all anaphoric relations are coreferential
- Not all noun phrases have reference 不是所有的名词短语都有指代
  - Every dancer twisted her knee
  - No dancer twisted her knee
  - 每⼀个句⼦有三个NPs；因为第⼀个是⾮指示性的，另外两个也不是
- Not all anaphoric relations are coreferential
  - We went to see a concert last night. The tickets were really expensive.
  - 这被称为桥接回指 bridging anaphora
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021323230934.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 通常先⾏词在回指（例如代词）之前，但并不总是
## Cataphora
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232327843.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Four Kinds of Coreference Models
- Rule-based (pronominal anaphora resolution)
- Mention Pair
- Mention Ranking
- Clustering
# 5. Traditional pronominal anaphora resolution:Hobbs’ naive algorithm
该算法仅⽤于寻找代词的参考，也可以延伸到其他案例
1. Begin at the NP immediately dominating the pronoun
2. Go up tree to first NP or S. Call this X, and the path p.
3. Traverse all branches below X to the left of p, left-to-right,breadth-first. Propose as antecedent any NP that has a NP or Sbetween it and X
4. If X is the highest S in the sentence, traverse the parse trees ofthe previous sentences in the order of recency. Traverse eachtree left-to-right, breadth first. When an NP is encountered,propose as antecedent. If X not the highest node, go to step 5.
5. From node X, go up the tree to the first NP or S. Call it X, andthe path p.
6. If X is an NP and the path p to X came from a non-head phraseof X (a specifier or adjunct, such as a possessive, PP, apposition, orrelative clause), propose X as antecedent(The original said “did not pass through the N’ that X immediatelydominates”, but the Penn Treebank grammar lacks N’ nodes….)
7. Traverse all branches below X to the left of the path, in a leftto-right, breadth first manner. Propose any NP encountered asthe antecedent
8. If X is an S node, traverse all branches of X to the right of thepath but do not go below any NP or S encountered. Proposeany NP as the antecedent.9. Go to step 4
## Hobbs Algorithm Example
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232437473.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

这是⼀个很简单、但效果很好的共指消解的 baseline
## Knowledge-based Pronominal Coreference
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232450458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 第⼀个例⼦中，两个句⼦具有相同的语法结构，但是出于外部世界知识，我们能够知道倒⽔之后，满的是杯⼦（第⼀个 it 指向的是 the cup），空的是壶（第⼆个 it 指向的是 the pitcher）
- 可以将世界知识编码成共指问题
## Hobbs’ algorithm: commentary
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232512508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


# 6. Coreference Models: Mention Pair
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232614145.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 训练⼀个⼆元分类器，为每⼀对 mention 的分配共参的概率$p(m_i, m_j)$
  - 例如，为了寻找 “she” 的共指，查看所有候选先⾏词(以前出现的 mention )，并确定哪些与之相关

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232700168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232704415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232709464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⽂章的 N 个 mention
- 如果 $m_i$和$m_j$ 是共指的，则$y_{ij}=1$ ，否则$y_{ij}=-1$
- 只是训练正常的交叉熵损失(看起来有点不同，因为它是⼆元分类)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232809277.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



- 遍历 mentions
- 遍历候选先⾏词(前⾯出现的 mention)
- 共指 mention 对应该得到⾼概率，其他应该得到低概率
## Mention Pair Test Time
- 共指解析是⼀项聚类任务，但是我们只是对 mentions 对进⾏了评分……该怎么办？
- 选择⼀些阈值(例如0.5)，并将 $p(m_i,m_j)$在阈值以上的 mentions 对之间添加共指链接
- 利⽤传递闭包得到聚类
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232854670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232858965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 共指连接具有传递性，即使没有不存在 link 的两者也会由于传递性，处于同⼀个聚类中

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232908562.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 这是⼗分危险的
- 如果有⼀个共指 link 判断错误，就会导致两个 cluster 被错误地合并了
## Mention Pair Models: Disadvantage
- 假设我们的⻓⽂档⾥有如下的 mentions
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213232933197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



- 许多 mentions 只有⼀个清晰的先⾏词
  - 但我们要求模型来预测它们
- 解决⽅案：相反，训练模型为每个 mention 只预测⼀个先⾏词
  - 在语⾔上更合理
# 7. Coreference Models: Mention Ranking
- 根据模型把其得分最⾼的先⾏词分配给每个 mention
- 虚拟的 NA mention 允许模型拒绝将当前 mention 与任何内容联系起来(“singleton” or “first” mention)
  - first mention： I 只能选择 NA 作为⾃⼰的先⾏词

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233008481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233012627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233017748.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233022236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



## Coreference Models: Training
- 我们希望当前 mention $m_j$ 与它所关联的任何⼀个候选先⾏词相关联。
- 在数学上，我们可能想要最⼤化这个概率

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233048212.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 公式解析
  - 遍历候选先⾏词集合
  - 对于$y_{ij}=1$ 的情况，即 $m_i$与$m_j$ 是共指关系的情况
  - 我们希望模型能够给予其⾼可能性
- 该模型可以为⼀个正确的先⾏词产⽣概率 0.9 ，⽽对其他所有产⽣较低的概率，并且总和仍然很⼤

- Turning this into a loss function
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233147336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Mention Ranking Models: Test Time


和 mention-pair 模型⼏乎⼀样，除了每个 mention 只分配⼀个先⾏词
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233204812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## How do we compute the probabilities?
A. Non-neural statistical classifier
B. Simple neural network
C. More advanced model using LSTMs, attention
## A. Non-Neural Coref Model: Features
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233227234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 使⽤如下特征进⾏分类
  - ⼈、数字、性别
  - 语义相容性
  - 句法约束
  - 更近的提到的实体是个可能的参考对象
  - 语法⻆⾊：偏好主语位置的实体
  - 排⽐
## B. Neural Coref Model
- 标准的前馈神经⽹络
  - 输⼊层：词嵌⼊和⼀些类别特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233300844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

### Neural Coref Model: Inputs
- 嵌⼊
  - 每个 mention 的前两个单词，第⼀个单词，最后⼀个单词，head word，…
    - head word是 mention 中“最重要”的单词—可以使⽤解析器找到它
    - 例如：The fluffy cat stuck in the tree
- 仍然需要⼀些其他特征
  - 距离
  - ⽂档体裁
  - 说话者的信息
## C. End-to-end Model
- 当前最先进的模型算法(Kenton Lee et al. from UW, EMNLP 2017)
- Mention 排名模型
- 改进了简单的前馈神经⽹络
  - 使⽤LSTM
  - 使⽤注意⼒
  - 端到端的完成 mention 检测和共指
    - 没有 mention 检测步骤！
    - ⽽是考虑每段⽂本(⼀定⻓度)作为候选 mention
      - a span 是⼀个连续的序列
### End-to-end Model
- ⾸先将⽂档⾥的单词使⽤词嵌⼊矩阵和 charCNN embed 为词嵌⼊
- 接着在⽂档上运⾏双向 LSTM
- 接着将每段⽂本 $i$从 $START(i)$到 $END(i)$表示为⼀个向量
  - span 是句⼦中任何单词的连续⼦句
  - General, General Electric, General Electric said, … Electric, Electric said, …都会得到它⾃⼰的向量表示
- span representation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233522316.png#pic_center)


  - 例如 “the postal service”
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233535645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



- $x^*_i$是 span 的注意⼒加权平均的词向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233607459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 为什么要在 span 中引⼊所有的这些不同的项
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233619880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 最后，为每个 span 对打分来决定他们是不是共指 mentions
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233630270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 打分函数以 span representations 作为输⼊
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021323364167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 为每对 span 打分是棘⼿的
  - ⼀个⽂档中有$O(T^2)$ spans，T 是词的个数
  - $O(T^4)$的运⾏时间
  - 所以必须做⼤量的修剪⼯作(只考虑⼀些可能是 mention 的span)
- 关注学习哪些单词是重要的在提到(有点像head word)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233729540.png#pic_center)


# 8. Last Coreference Approach: Clustering-Bas
- 共指是个聚类任务，让我们使⽤⼀个聚类算法吧
  - 特别是我们将使⽤ agglomerative 凝聚聚类 ⾃下⽽上的
- 开始时，每个 mention 在它⾃⼰的单独集群中
- 每⼀步合并两个集群
  - 使⽤模型来打分那些聚类合并是好的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233754178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233759305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Clustering Model Architecture
From Clark & Manning, 2016
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233813267.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- ⾸先为每个 mention 对⽣成⼀个向量
  - 例如，前馈神经⽹络模型中的隐藏层的输出
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233825886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 接着将池化操作应⽤于 mentino-pair 表示的矩阵上，得到⼀个 cluster-pair 聚类对的表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233838556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 通过⽤权重向量与表示向量的点积，对 candidate cluster merge 进⾏评分
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233905438.png#pic_center)

- 当前候选簇的合并，取决于之前的合并
  - 所以不能⽤常规的监督学习
  - 使⽤类似强化学习训练模型
    - 为每个合并分配奖励：共指评价指标的变化
# 9. Coreference Evaluation
- 许多不同的评价指标：MUC, CEAF, LEA, B-CUBED, BLANC
  - 经常使⽤⼀些不同评价指标的均值
- 例如 B-cubed
  - 对于每个 mention ，计算其准确率和召回率
  - 然后平均每个个体的准确率和召回率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213233959869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021323400496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## System Performance
- OntoNotes数据集:~ 3000⼈类标注的⽂档
  - 英语和中⽂
- Report an F1 score averaged over 3 coreference metrics
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234022372.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Where do neural scoring models help?
- 特别是对于没有字符串匹配的NPs和命名实体。神经与⾮神经评分:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234037171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Conclusion
- 共指是⼀个有⽤的、具有挑战性和有趣的语⾔任务
  - 许多不同种类的算法系统
- 系统迅速好转，很⼤程度上是由于更好的神经模型
  - 但总的来说,还没有惊⼈的结果
- Try out a coreference system yourself
  - http://corenlp.run/ (ask for coref in Annotations)
  - https://huggingface.co/coref/
