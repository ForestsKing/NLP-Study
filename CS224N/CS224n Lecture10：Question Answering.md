# 1.Lecture Plan
- Final final project notes, etc. 
- Motivation/History
- The SQuADdataset
- The Stanford Attentive Reader model
- BiDAF
- Recent, more advanced architectures
- ELMo and BERT preview
## Project writeup
- Abstract Introduction
- Prior related work
- Model
- Data
- Experiments
- Results
- Analysis & Conclusion
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212124913163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

在⾕歌中检索谁是澳⼤利亚第三任总理，可以获得答案。

技术说明：这是从web⻚⾯中提取的“特性⽚段”回答，⽽不是使⽤(结构化的)⾕歌知识图(以前称为Freebase)回答的问题。

我们今天要谈论的就是这样的问题，⽽不是基于结构化数据存储的问答。
# 2. Motivation: Question answering
- 拥有⼤量的全⽂⽂档集合，例如⽹络，简单地返回相关⽂档的作⽤是有限的
- 相反，我们经常想要得到问题的答案
- 尤其是在移动设备上
- 或是使⽤像Alexa、Google assistant这样的数字助理设备。
- 我们可以把它分解成两部分
  - 查找(可能)包含答案的⽂档
    - 可以通过传统的信息检索/web搜索处理
    - (下个季度我将讲授cs276，它将处理这个问题
  - 在⼀段或⼀份⽂件中找到答案
    - 这个问题通常被称为阅读理解
    - 这就是我们今天要关注的
## A Brief History of Reading Comprehension
- 许多早期的NLP⼯作尝试阅读理解
  - Schank, Abelson, Lehnert et al. c. 1977 –“Yale A.I. Project”
- 由Lynette Hirschman在1999年复活
  - NLP系统能回答三⾄六年级学⽣的⼈类阅读理解问题吗?简单的⽅法尝试
- Chris Burges于2013年通过 MCTest ⼜重新复活 RC
  - 再次通过简单的故事⽂本回答问题
- 2015/16年，随着⼤型数据集的产⽣，闸⻔开启，可以建⽴监督神经系统
  - Hermann et al. (NIPS 2015) DeepMind CNN/DM dataset
  - Rajpurkaret al. (EMNLP 2016) SQuAD
  - MS MARCO, TriviaQA, RACE, NewsQA, NarrativeQA, …
## Machine Comprehension (Burges 2013)
“⼀台机器能够理解⽂本的段落，对于⼤多数⺟语使⽤者能够正确回答的关于⽂本的任何问题，该机器都能提供⼀个字符串，这些说话者既能回答该问题，⼜不会包含与该问题⽆关的信息。”
## MCTestReading Comprehension
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131104147.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021213111486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## A Brief History of Open-domain Question Answering
- Simmons et al. (1964) ⾸先探索了如何基于匹配问题和答案的依赖关系解析，从说明性⽂本中回答问题
- Murax(Kupiec1993) 旨在使⽤IR和浅层语⾔处理在在线百科全书上回答问题
- NIST TREC QA track 始于1999年，⾸次严格调查了对⼤量⽂档的事实问题的回答
- IBM的冒险！System (DeepQA, 2011)提出了⼀个版本的问题;它使⽤了许多⽅法的集合
- DrQA(Chen et al. 2016)采⽤IR结合神经阅读理解，将深度学习引⼊开放领域的QA
## Turn-of-the Millennium Full NLP QA
[architecture of LCC (Harabagiu/Moldovan) QA system, circa 2003] 复杂的系统，但他们在“事实”问题上做得相当好
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131153510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- ⾮常复杂的多模块多组件的系统
  - ⾸先对问题进⾏解析，使⽤⼿写的语义规范化规则，将其转化为更好的语义形式
  - 在通过问题类型分类器，找出问题在寻找的语义类型
  - 信息检索系统找到可能包含答案的段落，排序后进⾏选择
  - NER识别候选实体再进⾏判断
- 这样的QA系统在特定领域很有效：Factoid Question Answering 针对实体的问答
# 3. Stanford Question Answering Dataset (SQuAD)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131227101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Passage 是来⾃维基百科的⼀段⽂本，系统需要回答问题，在⽂章中找出答案
- 答案必须是⽂章中的⼀系列单词序列，也就是提取式问答
- 100k examples

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131247393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## SQuAD evaluation, v1.1
- 作者收集了3个⻩⾦答案
- 系统在两个指标上计算得分
  - 精确匹配：1/0的准确度，您是否匹配三个答案中的⼀个
  - F1：将系统和每个答案都视为词袋，并评估
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021213132583.png#pic_center)

  - Precision 和 Recall 的调和平均值
  - 分数是(宏观)平均每题F1分数
- F1测量被视为更可靠的指标，作为主要指标使⽤
  - 它不是基于选择是否和⼈类选择的跨度完全相同，⼈类选择的跨度容易受到各种影响，包括换⾏
  - 在单次级别匹配不同的答案
- 这两个指标忽视标点符号和冠词(a, an, the only)
## SQuAD2.0
- SQuAD1.0的⼀个缺陷是，所有问题都有答案的段落
- 系统(隐式地)排名候选答案并选择最好的⼀个，这就变成了⼀种排名任务
- 你不必判断⼀个span是否回答了这个问题
- SQuAD2.0中 1/3 的训练问题没有回答，⼤约 1/2 的开发/测试问题没有回答
  - 对于No Answer examples, no answer 获得的得分为1，对于精确匹配和F1，任何其他响应的得分都为0
- SQuAD2.0最简单的系统⽅法
  - 对于⼀个 span 是否回答了⼀个问题有⼀个阈值评分
- 或者您可以有第⼆个确认回答的组件
  - 类似 ⾃然语⾔推理 或者 答案验证
**Example**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131431364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

得分⾼的系统并不能真正理解⼈类语⾔！
## Good systems are great, but still basic NLU errors
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131449735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

系统没有真正了解⼀切，仍然在做⼀种匹配问题
## SQuAD limitations
- SQuAD 也有其他⼀些关键限制
  - 只有 span-based 答案(没有 yes/no，计数，隐式的为什么)
  - 问题是看着段落构造的
    - 通常不是真正的信息需求
    - ⼀般来说，问题和答案之间的词汇和句法匹配⽐IRL更⼤
    - 问题与⽂章⾼度重叠，⽆论是单词还是句法结构
  - 除了共同参照，⼏乎没有任何多事实/句⼦推理
- 不过这是⼀个⽬标明确，结构良好的⼲净的数据集
  - 它⼀直是QA dataset上最常⽤和最具竞争⼒的数据集
  - 它也是构建⾏业系统的⼀个有⽤的起点（尽管域内数据总是很有帮助！）
  - 并且我们正在使⽤它
# 4. Stanford Attentive Reader

- 展示了⼀个最⼩的，⾮常成功的阅读理解和问题回答架构
- 后来被称为 the Stanford Attentive Reader
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131611875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131627867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131640165.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

⾸先将问题⽤向量表示
- 对问题中的每个单词，查找其词嵌⼊
- 输⼊到双向LSTM中并将最终的 hidden state 拼接
再处理⽂章
- 查找每个单词的词嵌⼊并输⼊到双向LSTM中
- 使⽤双线性注意⼒，将每个LSTM的表示(LSTM的两个隐藏状态的连接)与问题表示做运算，获得了不同位置的注意⼒，从⽽获得答案的开始位置，再以同样⽅式获得答案的结束位置
  - 为了在⽂章中找到答案，使⽤问题的向量表示，来解决答案在什么位置使⽤注意⼒
## Stanford Attentive Reader++
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131721686.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

整个模型的所有参数都是端到端训练的，训练的⽬标是开始位置与结束为⽌的准确度，优化有两种⽅式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131741384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

问题部分
- 不⽌是利⽤最终的隐藏层状态，⽽是使⽤所有隐层状态的加权和
  - 使⽤⼀个可学习的向量 与 每个时间步的隐层状态相乘
- 深层LSTM
⽂章部分
⽂章中每个token的向量表示由⼀下部分连接⽽成
- 词嵌⼊(GloVe300d)
- 词的语⾔特点:POS &NER 标签，one-hot 向量
- 词频率(unigram概率)
- 精确匹配:这个词是否出现在问题
  - 三个⼆进制的特征： exact, uncased, lemma
- 对⻬问题嵌⼊(“⻋”与“汽⻋”)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131829786.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131847891.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131859214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 单词相似度的语义匹做得更好
# 5. BiDAF: Bi-Directional Attention Flow for Machine Comprehension
(Seo, Kembhavi, Farhadi, Hajishirzi, ICLR 2017)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212131929368.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 多年来，BiDAF architecture有许多变体和改进，但其核⼼思想是 the Attention Flow layer
- Idea ：attention 应该双向流动——从上下⽂到问题，从问题到上下⽂
- 令相似矩阵( $w$的维数为6d)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132030834.png#pic_center)

- Context-to-Question (C2Q) 注意⼒ (哪些查询词与每个上下⽂词最相关)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021213204360.png#pic_center)

- Question-to-Context (Q2C) 注意⼒ (上下⽂中最重要的单词相对于查询的加权和——通过max略有不对称)
  - 通过max取得上下⽂中的每个单词对于问题的相关度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132108862.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 对于⽂章中的每个位置，BiDAF layer的输出为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132126168.png#pic_center)


- 然后有“modelling”层
  - ⽂章通过另⼀个深(双层)BiLSTM
- 然后回答跨度选择更为复杂
  - Start：通过BiDAF 和 modelling 的输出层连接到⼀个密集的全连接层然后softmax
  - End：把 modelling 的输出 $M$ 通过另⼀个BiLSTM得到 $M_2$，然后再与BiDAF layer连接，并通过密集的全连接层和softmax
# 6. Recent, more advanced architectures
2016年、2017年和2018年的⼤部分⼯作都采⽤了越来越复杂的架构，其中包含了多种注意⼒变体——通常可以获得很好的任务收益

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132236869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


⼈们⼀直在尝试不同的 Attention
## Dynamic CoattentionNetworks for Question Answering
(CaimingXiong, Victor Zhong, Richard Socher ICLR 2017)
- 缺陷：问题具有独⽴于输⼊的表示形式
- ⼀个全⾯的QA模型需要相互依赖
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132300539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Coattention Encoder
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132316314.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- Coattention layer 再次提供了⼀个上下⽂之间的双向关注问题
- 然⽽，coattention包括两级注意⼒计算
  - 关注那些本身就是注意⼒输出的表象
- 我们使⽤C2Q注意⼒分布 $\alpha_i$，求得Q2C注意⼒输出$b_j$ 的加权和。这给了我们第⼆级注意⼒输出$s_i$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132428319.png#pic_center)


## FusionNet(Huang, Zhu, Shen, Chen 2017)
Attention functions
#pic_center
MLP(Additive)形式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132453299.png#pic_center)

- Space: O(mnk), W is kxd
Bilinear (Product) form
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132521775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Smaller space, Non-linearity
- Space: O((m+n)k)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021213254098.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Multi-level inter-attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212132554943.png#pic_center)

经过多层次的inter-attention，使⽤RNN、self-attention 和另⼀个RNN得到上下⽂的最终表示$\{u_i^C\}$
# 7. ELMo and BERT preview
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102121326343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)