# Lecture Plan
- Reflections on word representations
- Pre-ELMo and ELMO
- ULMfit and onward
- Transformer architectures (20 mins)
- BERT

The remaining lectures
- Transformers
- BERT
- Question answering
- Text generation and summarization
- “New research, latest updates in the field”
- “Successful applications of NLP in industry today”
- “More neural architectures that are used to solve NLP problem”
- “More linguistics stuff and NLU!”

# 1. Representations for a word
现在我们可以获得⼀个单词的表示
- 我们开始时学过的单词向量 Word2vec, GloVe, fastText
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213082536482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213082544138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Tips for unknown words with word vectors
- 简单且常⻅的解决⽅案
- 训练时：词汇表 Vocab 为 $\{words occurring,say,\geq 5 times\} \cup \{<UNK>\}$
- 将所有罕⻅的词（数据集中出现次数⼩于 5）都映射为$<UNK>$ ，为其训练⼀个词向量
- 运⾏时：使⽤ $<UNK>$ 代替词汇表之外的词 OOV
- 问题
  - 没有办法区分不同UNK话说,⽆论是身份还是意义
- 解决⽅案
  - 使⽤字符级模型学习期词向量
    - 特别是在 QA 中，match on word identity 是很重要的,即使词向量词汇表以外的单词
  - Try these tips (from Dhingra, Liu, Salakhutdinov, Cohen 2017)
    - 如果测试时的 $<UNK>$ 单词不在你的词汇表中，但是出现在你使⽤的⽆监督词嵌⼊中，测试时直接使⽤这个向量
    - 此外，你可以将其视为新的单词，并为其分配⼀个随机向量，将它们添加到你的词汇
表。
- 帮助很⼤ 或者 也许能帮点忙
- 你可以试试另⼀件事
  - 将它们分解为词类（如未知号码，⼤写等等），每种都对应⼀个$<UNK-class>$ 
## Representations for a word
存在两个⼤问题
- 对于⼀个 word type 总是是⽤相同的表示，不考虑这个 word token 出现的上下⽂
  - ⽐如 star 这个单词，有天⽂学上的含义以及娱乐圈中的含义
  - 我们可以进⾏⾮常细粒度的词义消歧
- 我们对⼀个词只有⼀种表示，但是单词有不同的⽅⾯，包括语义，句法⾏为，以及表达 / 含义
  - 表达：同样的意思可以是⽤多个单词表示，他们的词义是⼀样的
## Did we all along have a solution to this problem?
- 在模型中，我们通过LSTM层(也许只在语料库上训练)
- 那些LSTM层被训练来预测下⼀个单词
- 但这些语⾔模型在每⼀个位置⽣成特定于上下⽂的词表示
# 2. Peters et al. (2017): TagLM – “Pre-ELMo”
https://arxiv.org/pdf/1705.00108.pdf
- 想法：想要获得单词在上下⽂的意思，但标准的 RNN 学习任务只在 task-labeled 的⼩数据上（如NER ）
- 为什么不通过半监督学习的⽅式在⼤型⽆标签数据集上训练 NLM，⽽不只是词向量
## Tag LM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083013929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 与上⽂⽆关的单词嵌⼊ + RNN model 得到的 hidden states 作为特征输⼊
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083028403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083041938.png#pic_center)

- Char CNN / RNN + Token Embedding 作为 bi-LSTM 的输⼊
- 得到的 hidden states 与 Pre-trained bi-LM（冻结的） 的 hidden states 连接起来输⼊到第⼆层的 bi-LSTM 中
## Named Entity Recognition (NER)
- ⼀个⾮常重要的NLP⼦任务：查找和分类⽂本中的实体，例如
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083106114.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## CoNLL 2003 Named Entity Recognition (en news testb)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083118424.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Peters et al. (2017): TagLM – “Pre-ELMo”
语⾔模型在“Billion word benchmark”的8亿个训练单词上训练

语⾔模型观察结果
- 在监督数据集上训练的LM并不会受益
- 双向LM仅有助于 forward 过程，提升约0.2
- 具有巨⼤的 LM 设计（困惑度 30） ⽐较⼩的模型（困惑度 48）提升约0.3
>困惑度

任务特定的BiLSTM观察结果
- 仅使⽤LM嵌⼊来预测并不是很好：88.17 F1
  - 远低于仅在标记数据上使⽤ BiLSTM 标记器
## Also in the air: McCann et al. 2017: CoVe
https://arxiv.org/pdf/1708.00107.pdf
- 也有使⽤训练好的序列模型为其他NLP模型提供上下⽂的想法
- 想法：机器翻译是为了保存意思，所以这也许是个好⽬标？
- 使⽤seq2seq + attention NMT system中的Encoder，即 2层 bi-LSTM ，作为上下⽂提供者
- 所得到的 CoVe 向量在各种任务上都优于 GloVe 向量
- 但是，结果并不像其他幻灯⽚中描述的更简单的NLM培训那么好，所以似乎被放弃了
  - 也许NMT只是⽐语⾔建模更难？
  - 或许有⼀天这个想法会回来？
## Peters et al. (2018): ELMo: Embeddings from Language Models
Deep contextualized word representations. NAACL 2018. https://arxiv.org/abs/1802.05365

- word token vectors or contextual word vectors 的爆发版本
- 使⽤⻓上下⽂⽽不是上下⽂窗⼝学习 word token 向量(这⾥，整个句⼦可能更⻓)
- 学习深度Bi-NLM，并在预测中使⽤它的所有层
- 训练⼀个双向LM
- ⽬标是 performant 但LM不要太⼤
  - 使⽤2个biLSTM层 
  - (仅)使⽤字符CNN构建初始单词表示
  - 2048 个 char n-gram filters 和 2 个 highway layers，512 维的 projection
  - 4096 dim hidden/cell LSTM状态，使⽤ 512 dim的对下⼀个输⼊的投影
  - 使⽤残差连接
  - 绑定 token 的输⼊和输出的参数(softmax)，并将这些参数绑定到正向和反向LMs之间
- ELMo学习biLM表示的特定任务组合
- 这是⼀个创新，TagLM 中仅仅使⽤堆叠LSTM的顶层，ELMo 认为BiLSTM所有层都是有⽤的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083352732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- $\gamma^{task}$衡量ELMo对任务的总体有⽤性，是为特定任务学习的全局⽐例因⼦
- $s^task$是 softmax 归⼀化的混合模型权重，是 BiLSTM 的加权平均值的权重，对不同的任务是不同的，因为不同的任务对不同层的 BiLSTM 的
## Peters et al. (2018): ELMo: Use with a task
- ⾸先运⾏ biLM 获取每个单词的表示
- 然后让(⽆论什么)最终任务模型使⽤它们
- 冻结ELMo的权重，⽤于监督模型
- 将ELMo权重连接到特定于任务的模型中
  - 细节取决于任务
    - 像 TagLM ⼀样连接到中间层是典型的
    - 可以在⽣产输出时提供更多的表示，例如在问答系统中
## ELMo used in a sequence tagger
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083507169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083519467.png#pic_center)

## ELMo results: Great for all tasks
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083529922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## ELMo: Weighting of layers
- 这两个biLSTM NLM层有不同的⽤途/含义
  - 低层更适合低级语法，例如
    - 词性标注(part-of-speech tagging)、句法依赖(syntacticdependency)、NER
  - ⾼层更适合更⾼级别的语义
    - 情绪、Semantic role labeling 语义⻆⾊标记 、question answering、SNLI
- 这似乎很有趣，但它是如何通过两层以上的⽹络来实现的看起来更有趣
## Also around: ULMfit
Howard and Ruder (2018) Universal Language Model Fine-tuning for Text Classification. 
https://arxiv.org/pdf/1801.06146.pdf
- 转移NLM知识的⼀般思路是⼀样的
- 这⾥应⽤于⽂本分类
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083617665.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 3. ULMfit
- 在⼤型通⽤领域的⽆监督语料库上使⽤ biLM 训练
- 在⽬标任务数据上调整 LM
- 对特定任务将分类器进⾏微调

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083636482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## ULMfit emphases
- 使⽤合理⼤⼩的“1 GPU”语⾔模型，并不是真的很⼤
- 在LM调优中要注意很多
  - 不同的每层学习速度
  - 倾斜三⻆形学习率(STLR)计划
  - 学习分类器时逐步分层解冻和STLR
  - 使⽤ $[h_r,maxpool(h),meanpool(h)]$进⾏分类
- 使⽤⼤型的预训练语⾔模型是⼀种提⾼性能的⾮常有效的⽅法
## ULMfit performance
- ⽂本分类器错误率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083746864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## ULMfit transfer learning
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083758672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 如果使⽤监督数据进⾏训练⽂本分类器，需要⼤量的数据才能学习好
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083810342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## GPT-2 language model (cherry-picked) output
- ⽂本⽣成的样例
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083827210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308383929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213083846384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Transformer 不仅很强⼤，⽽且允许扩展到更⼤的尺⼨
# 4. The Motivation for Transformers
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308390051.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 我们想要并⾏化，但是RNNs本质上是顺序的
- 尽管有GRUs和LSTMs, RNNs仍然需要注意机制来处理⻓期依赖关系——否则状态之间的 path length 路径⻓度 会随着序列增⻓
- 但如果注意⼒让我们进⼊任何⼀个状态……也许我们可以只⽤注意⼒⽽不需要RNN?
## Transformer Overview
Attention is all you need. 2017. Aswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin https://arxiv.org/pdf/1706.03762.pdf

- Non-recurrent sequence-to-sequence encoder-decoder model
- 任务：平⾏语料库的机器翻译
- 预测每个翻译单词
- 最终成本/误差函数是 softmax 分类器基础上的标准交叉熵误差

## Transformer Basics
- ⾃学
  - 主要推荐资源
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    - The Annotated Transformer by Sasha Rush
  - An Jupyter Notebook using PyTorch that explains everything!
- 现在：我们定义 Transformer ⽹络的基本构建块：第⼀，新的注意⼒层
## Dot-Product Attention (Extending our previous def.)
- 输⼊：对于⼀个输出⽽⾔的查询 q 和⼀组键-值对 k-v
- Query, keys, values, and output 都是向量
- 输出值的加权和
- 权重的每个值是由查询和相关键的内积计算结果
- Query 和 keys 有相同维数 ，value 的维数为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084052682.png#pic_center)


## Dot-Product Attention – Matrix notation
- 当我们有多个查询 q 时，我们将它们叠加在⼀个矩阵 Q 中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084112894.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084120893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Scaled Dot-Product Attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084132550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 问题： $d_k$变⼤时，$q^Tk$ 的⽅差增⼤$\rightarrow$ ⼀些 softmax 中的值的⽅差将会变⼤ $\rightarrow$softmax 得到的是峰值 $\rightarrow$因此梯度变⼩了
- 解决⽅案：通过query/key向量的⻓度进⾏缩放

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084246749.png#pic_center)

## Self-attention in the encoder
- 输⼊单词向量是queries, keys and values
- 换句话说：这个词向量⾃⼰选择彼此
- 词向量堆栈= Q = K = V
- 我们会通过解码器明⽩为什么我们在定义中将他们分开
## Multi-head attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084309620.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 简单self-attention的问题
- 单词只有⼀种相互交互的⽅式
- 解决⽅案：多头注意⼒
- ⾸先通过矩阵 W 将 Q, K, V 映射到 h = 8 的许多低维空间
- 然后应⽤注意⼒，然后连接输出，通过线性层

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084336790.png#pic_center)

## Complete transformer block
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084344426.png#pic_center)

- 每个 block 都有两个“⼦层”
  - 多头 attention
  - 两层的前馈神经⽹络，使⽤ ReLU

这两个⼦层都：
- 残差连接以及层归⼀化
  - LayerNorm(x+Sublayer(x))
  - 层归⼀化将输⼊转化为均值是 0，⽅差是 1 ，每⼀层和每⼀个训练点（并且添加了两个参数）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084430641.png#pic_center)

Layer Normalization by Ba, Kiros and Hinton, https://arxiv.org/pdf/1607.06450.pdf
## Encoder Input
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084440613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 实际的词表示是 byte-pair 编码
- 还添加了⼀个 positional encoding 位置编码，相同的词语在不同的位置有不同的整体表征

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084504902.png#pic_center)

## Complete Encoder
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084515547.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- encoder 中，每个 block 都是来⾃前⼀层的 Q, K, V
- Blocks 被重复 6 次（垂直⽅向）
- 在每个阶段，你可以通过多头注意⼒看到句⼦中的各个地⽅，累积信息并将其推送到下⼀层。在任⼀⽅向上的序列逐步推送信息来计算感兴趣的值
- ⾮常善于学习语⾔结构
## Attention visualization in layer 5
- 词语开始以合理的⽅式关注其他词语
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084545343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 不同的颜⾊对应不同的注意⼒头
## Attention visualization: Implicit anaphora resolution
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084558122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 对于代词，注意⼒头学会了如何找到其指代物
- 在第五层中，从 head 5 和 6 的单词“its”中分离出来的注意⼒。请注意，这个词的注意⼒是⾮常鲜明的。
## Transformer Decoder
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084617874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- decoder 中有两个稍加改变的⼦层
- 对之前⽣成的输出进⾏ Masked decoder self-attention
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084634704.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Encoder-Decoder Attention，queries 来⾃于前⼀个 decoder 层，keys 和 values 来⾃于 encoder 的输出
- Blocks 同样重复 6 次
## Tips and tricks of the Transformer
细节(论⽂/讲座)
- Byte-pair encodings
- Checkpoint averaging
- Adam 优化器控制学习速率变化
- 训练时，在每⼀层添加残差之前进⾏ Dropout
- 标签平滑
- 带有束搜索和⻓度惩罚的 Auto-regressive decoding
- 因为 transformer 正在蔓延，但他们很难优化并且不像LSTMs那样开箱即⽤，他们还不能很好与其他任务的构件共同⼯作
## Experimental Results for MT
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084723450.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Experimental Results for Parsing
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084742810.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 5. BERT: Devlin, Chang, Lee, Toutanova (2018)
BERT (Bidirectional Encoder Representations from Transformers):

Pre-training of Deep Bidirectional Transformers for Language Understanding

- 问题：语⾔模型只使⽤左上下⽂或右上下⽂，但语⾔理解是双向的
- 为什么LMs是单向的？
- 原因1：⽅向性对于⽣成格式良好的概率分布是有必要的
  - 我们不在乎这个
- 原因2：双向编码器中单词可以“看到⾃⼰”
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084812382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 解决⽅案：mask out k % 的输⼊单词，然后预测 masked words
- 不再是传统的计算⽣成句⼦的概率的语⾔模型，⽬标是填空
  - 总是使⽤k = 15%

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084827376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
- Masking 太少：训练太昂贵
- Masking 太多：没有⾜够的上下⽂
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102130848406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- GPT 是经典的单项的语⾔模型
- ELMo 是双向的，但是两个模型是完全独⽴训练的，只是将输出连接在⼀起，并没有使⽤双向的 context
- BERT 使⽤ mask 的⽅式进⾏整个上下⽂的预测，使⽤了双向的上下⽂信息
## BERT complication: Next sentence prediction
- 学习句⼦之间的关系，判断句⼦ B 是句⼦ A 的后⼀个句⼦还是⼀个随机的句⼦。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308490735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## BERT sentence pair encoding
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213084916895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- token embeddings 是 word pieces （play, ##ing ）
- 使⽤学习好的分段嵌⼊表示每个句⼦
- 位置嵌⼊与其他 Transformer 体系结构类似
- 将以上三种 embedding 相加，作为最终输⼊的表示
## BERT model architecture and training
- Transformer encoder（和之前的⼀样）
- ⾃注意⼒ $\rightarrow$ 没有位置偏差
  - ⻓距离上下⽂“机会均等”
- 每层乘法  $\rightarrow$  GPU / TPU上⾼效
- 在 Wikipedia + BookCorpus 上训练
- 训练两种模型尺⼨
  - BERT-Base: 12-layer, 768-hidden, 12-head
  - BERT-Large: 24-layer, 1024-hidden, 16-head
- Trained on 4x4 or 8x8 TPU slice for 4 days
## BERT model fine tuning
- 只学习⼀个建⽴在顶层的分类器，微调的每个任务
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213085126205.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## BERT results on GLUE tasks
- GLUE benchmark 是由⾃然语⾔推理任务,还有句⼦相似度和情感
- MultiNLI
- Premise: Hills and mountains are especially sanctified in Jainism.
  Hypothesis: Jainism hates nature.
  Label: Contradiction
- CoLa
- Sentence: The wagon rumbled down the road. Label: Acceptable
- Sentence: The car honked down the road. Label: Unacceptable
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213085201627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## BERT results on SQuAD 1.1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213085217852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## SQuAD 2.0 leaderboard, 2019-02-07
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213085228639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Effect of pre-training task
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213085240593.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Size matters
- 从 119M 到 340M 的参数量改善了很多
- 改进尚未渐进