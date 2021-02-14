## Deep Learning for NLP 5 years ago
- No Seq2Seq
- No Attention
- No large-scale QA/reading comprehension datasets
- No TensorFlow or Pytorch
## Future of Deep Learning + NLP
- 利⽤⽆标签数据
  - Back-translation 和 ⽆监督机器翻译
  - 提⾼预训练和GPT-2
- 接下来呢？
  - NLP技术的⻛险和社会影响
  - 未来的研究⽅向
## Why has deep learning been so successful recently?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212214931713.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212214939490.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 扩展能⼒（模型和数据⼤⼩）是深度学习近些年来成功的原因
- 过去受到计算资源和数据资源的规模限制
## Big deep learning successes
三个使⽤⼤量数据获得成功的范例
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215007799.png#pic_center)

- 图像识别：被 Google, Facebook 等⼴泛使⽤
  - ImageNet: 14 million examples
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215127595.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 机器翻译：⾕歌翻译等
  - WMT: Millions of sentence pairs
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215140962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 打游戏：Atari Games, AlphaGo, and more
  - 10s of millions of frames for Atari AI
  - 10s of millions of self-play games for AlphaZero
## NLP Datasets
- 即使是英语，⼤部分任务也只有 100k 或更少的有标签样本
- 其他语⾔的可⽤数据就更少了
  - 有成千上万的语⾔，其中有成百上千的语⾔的⺟语使⽤者是⼤于⼀百万的
  - 只有 10% 的⼈将英语作为他们的第⼀语⾔
- 越来越多的解决⽅案是使⽤ ⽆标签 数据
# Using Unlabeled Data for Translation
## Machine Translation Data
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215240529.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 获得翻译需要⼈类的专业知识
  - 限制数据的⼤⼩和领域
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221525474.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 语⾔⽂本更容易获得
## Pre-Training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215306901.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 分别将两个预训练好的语⾔模型作为 Encoder 和 Decoder
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215324235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 然后使⽤双语数据共同训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215341120.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- English -> German Results: 2+ BLEU point improvement
Ramachandran et al., 2017
## Self-Training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215355875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 预训练的问题：预训练中两个语⾔之间没有交互
- ⾃训练：标记未标记的数据以获得有噪声的训练样本
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215413245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⾃训练技术没有被⼴泛使⽤，因为其训练的来源是其之前的产出
## Back-Translation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215433527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 有两种⽅向相反的机器翻译模型 $en \rightarrow fr$ 和 $fr \rightarrow en$ 
- 不再循环
- 模型再也看不到“坏”翻译,只有坏输⼊
- 模型训练时会加⼊⼀些标记数据，确保 $en \rightarrow fr$ 模型的输出，即 $fr \rightarrow en$ 模型的输⼊，从⽽保证模型的正常
- 如何协调对标记数据与未标记数据的训练呢？
  - 先在标记数据上训练两个模型
  - 然后在未标记数据上标记⼀些数据
  - 再在未标记数据上进⾏反向翻译的训练
  - 重复如上的过程
## Large-Scale Back-Translation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215620921.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 4.5M English-German sentence pairs and 226M monolingual sentences
## What if there is no Bilingual Data?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215635700.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

当我们只有未标记的句⼦时，我们使⽤⼀种⽐完全的翻译更简单的任务
- 不是做句⼦翻译
- ⽽是做单词翻译
我们想要找到某种语⾔的翻译但不使⽤任何标记数据
## Unsupervised Word Translation
- 跨语⾔⽂字嵌⼊ cross-lingual word embeddings
  - 两种语⾔共享嵌⼊空间
  - 保持词嵌⼊的正常的好属性
  - 但也要接近他们的翻译
- 想从单语语料库中学习
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215707493.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 如上图所示，在共享的嵌⼊空间中，每个英⽂单词都有其对应的德语单词，并且距离很近
- 我们在使⽤时，只需选取英⽂单词在嵌⼊空间中距离最近的德语单词，就可以获得对应的翻译
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215723378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215730571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 词嵌⼊有很多结构
- 假设:不同语⾔之间的结构应该相似

即使是运⾏两次 word2vec 会获得不同的词嵌⼊，嵌⼊空间的结构有很多规律性

- 如上图所示，是英语与意⼤利语的词嵌⼊，⽮量空间看上去彼此⼗分不同，但是结构是⼗分相似的
  - 可以理解为，在英语词嵌⼊空间中的 cat 与 feline 的距离与意⼤利语词典如空间中的 gatto 和 felino 之间的距离是相似的
- 我们在跨语⾔的词嵌⼊中想要学习不同种语⾔的词嵌⼊之间的对⻬⽅式
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021221581288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⾸先在单语语料库上运⾏ word2vec 以得到单词嵌⼊ X 和 Y
- 学习⼀个（正交）矩阵 W 使得 WX~Y
  - 使⽤对抗训练来学习 W
  - 鉴别器：预测⼀个嵌⼊是来⾃于 Y 的还是来⾃于 X 并使⽤ W 转换后的嵌⼊
  - 训练 W 使得鉴别器难以区分这两者
  - 其他可以被⽤来进⼀步提升效果的⽅法参⻅ Word Translation without Parallel Data
  - 正交性来约束词嵌⼊的原因是为了防⽌过拟合
    - 我们假设我们的嵌⼊空间是类似的，只是需要对英语的词向量和意⼤利语的词向量进⾏旋转
## Unsupervised Machine Translation
- 模型：不考虑不同输⼊和输出语⾔，使⽤相同的(共享的) encoder-decoder (没有使⽤注意⼒)
  - 使⽤ cross-lingual 的词嵌⼊来初始化，即其中的英语和法语单词应该看起来完全相同
- 我们可以喂给 encoder ⼀个英⽂句⼦，也可以喂⼀个法语句⼦，从⽽获得 cross-lingual embeddings ，即英⽂句⼦和法语句⼦中各个单词的词嵌⼊，这意味着 encoder 可以处理任何输 ⼊
- 对于 decoder，我们需要喂⼀个特殊的标记 $Fr$ 来告诉模型应该⽣成什么语⾔的输出
  - 可以⽤做⼀个 auto-encoder，完成 $en \rightarrow en$，即再现输⼊序列
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212215958362.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


接下⾥是模型的训练过程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212220006623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Training objective 1: de-noising autoencoder 去噪⾃编码器
  - 输⼊⼀个打乱后的英语 / 法语句⼦
  - 输出其原来的句⼦
  - 由于这是⼀个没有注意⼒机制的模型，编码器将整个源句⼦转换为单个向量，⾃编码器的作⽤是保证来⾃于 encoder 的向量包含和这个句⼦有关的，能使得我们恢复原来的句⼦的所有信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080300799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Training objective 2: back translation (只有⽆标签的数据)
  - ⾸先翻译 $fr \rightarrow en$
  - 然后使⽤⼀个监督样本来训练$en \rightarrow fr$
- 注意，这⾥的 $fr \rightarrow en$ 输出的句⼦，是 $en \rightarrow fr$ 输⼊的句⼦，这个句⼦是有些混乱的，不完美的，例如这⾥的 “I am student”，丢失了 “a”
- 我们需要训练模型，即使是有这样糟糕的输⼊，也能够还原出原始的法语句⼦
## Why Does This Work?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080444458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 跨语⾔嵌⼊和共享编码器为模型提供了⼀个起点
  - 使⽤ cross-lingual 的词嵌⼊来初始化，即其中的英语和法语单词应该看起来完全相同
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080459281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 共享编码器
  - 例如我们以⼀个法语句⼦作为模型的输⼊
  - 由于嵌⼊看起来⾮常相似，并且我们使⽤的是相同的 encoder
  - 因此 encoder 得到的法语句⼦的 representation 应该和英语句⼦的 representation ⾮常相似
  - 所以希望能够获得和原始的英语句⼦相同的输出

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080528287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⽬标⿎励 language-agnostic 语⾔⽆关的表示
  - 获得⼀个与语⾔类型⽆关的 encoder vector
## Unsupervised Machine Translation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080544820.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Lample et al., 2018
- ⽔平线是⽆监督模型，其余的都是有监督的
- 在⼀定的监督数据规模下，⽆监督模型能够取得和监督模型类似的效果
- 当然，随着数据规模的增⼤，监督模型的效果会提升，超过⽆监督模型
## Attribute Transfer
还可以使⽤⽆监督的机器翻译模型完成属性转移
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080603200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Lample et al., 2019
- Collector corpora of “relaxed” and “annoyed” tweets using hashtags
- Learn un unsupervised MT model
## Not so Fast
- 英语，法语和德语是相当类似的语⾔
- 在⾮常不同的语⾔上（例如英语和⼟⽿其语）
  - 完全的⽆监督的词翻译并不⼗分有效。需要种⼦字典可能的翻译
    - 简单的技巧：使⽤相同的字符串从词汇
  - UNMT⼏乎不⼯作
## Cross-Lingual BERT
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080636818.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Lample and Conneau., 2019
- 上图 1 是常规的 BERT ，有⼀系列的英语句⼦，并且会 mask ⼀部分单词
  - ⾕歌实际上已经完成的是训练好的多语⾔的 BERT
  - 基本上是连接⼀⼤堆不同语⾔的语料库，然后训练⼀个模型
    - masked LM training objective
- 上图 2 是 Facebook 提出的
  - 联合了 masked LM training objective 和 翻译
  - 给定⼀个英语句⼦和法语句⼦，并分别 mask ⼀部分单词，并期望模型填补

**Unsupervised MT Results**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080719880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# Huge Models and GPT-2
## Training Huge Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080733736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## This is a General Trend in ML
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308074618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- peta : ⽤于计量单位，表示10的15次⽅，表示千万亿次
- FLOPS = FLoating-point Operations Per Second，每秒浮点运算次数
## Huge Models in Computer Vision
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080802507.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308081025.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Training Huge Models
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080820482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 更好的硬件
- 数据和模型的并⾏化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080831526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## GPT-2
- 只是⼀个⾮常⼤的 Transformer LM
- 40 GB的训练⽂本
  - 投⼊相当多的努⼒去确保数据质量
  - 使⽤ reddit 中获得⾼投票的⽹⻚ link
## So What Can GPT-2 Do?
- 显然，语⾔建模(但是⾮常好)
- 数据集上得到最先进的困惑，甚⾄没有训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213080859659.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Radford et al., 2019
- Zero-Shot Learning: no supervised training data! 在没有接受过训练的情况下常识完成任务
  - Ask LM to generate from a prompt
- Reading Comprehension: <context> <question> A:
- Summarization: <article> TL;DR:
- Translation:
  

```python
<English sentence1> = <French sentence1>
<English sentence 2> = <French sentence 2>
...
<Source sentenc> =
```

- Question Answering: `<question> A`:
## GPT-2 Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081129288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## How can GPT-2 be doing translation?
- 它有⼀个很⼤的语料库，⾥⾯⼏乎全是英语
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081146130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 由于数据集中存在⼀些翻译的例⼦
  - 法语习语及其翻译
  - 法语引⽤及其翻译
## GPT-2 Question Answering
- Simple baseline: 1% accuracy
- GPT-2: ~4% accuracy
- Cherry-picked most confident results 精选出最⾃信的结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081210848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## What happens as models get even bigger?
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308122141.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 对于⼀些任务，性能似乎随着 log(模型⼤⼩) 的增加⽽增加
- 但如下图所示趋势并不明朗
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081233503.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## GPT-2 Reaction
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081246526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- NLP专家应该做这些决定吗？
  - 计算机安全专家？
  - 技术和社会专家？
  - 道德专家？
- 需要更多的跨学科科学
- 许多NLP具有较⼤社会影响的例⼦，尤其是对于偏⻅/公平
## High-Impact Decisions
- 越来越感兴趣⽤NLP帮助⾼影响⼒的决策
  - 司法判决
  - 招聘
  - 等级测试
- ⼀⽅⾯，可以快速评估机器学习系统某些偏⻅
- 然⽽，机器学习反映了训练数据
  - 甚⾄放⼤偏⻅…这可能导致更偏向数据的创建
## Chatbots
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081330942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# What did BERT “solve” and what do we work on next?
## GLUE Benchmark Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081343886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## The Death of Architecture Engineering?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081354193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081359532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081405383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 花费六个⽉来研究 体系结构的设计，得到了 1 F1 的提升
- 或知识让 BERT 变得 3 倍⼤⼩，得到了 5 F1 的提升
- SQuAD 的 TOP20 参赛者都是⽤了 BERT
## Harder Natural Language Understanding
- 阅读理解
  - 在⻓⽂档或多个⽂档
  - 需要多跳推理
  - 在对话中定位问答
- 许多现有阅读理解数据集的关键问题：⼈们写问题时看着上下⽂
  - 不现实的
  - ⿎励简单的问题
## QuAC: Question Answering in Context
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081443252.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Choi et al., 2018
- 学⽣问问题，⽼师回答的对话
  - 教师看到维基百科⽂章主题，学⽣不喜欢
- 仍然和⼈类⽔平有很⼤差距
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081501545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## HotPotQA
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081514372.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Zang et al., 2018
- 设计要求多跳推理
- 问题在多个⽂档
- Human performance is above 90 F1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081530230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Multi-Task Learning
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081541621.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- NLP的另⼀个前沿是让⼀个模型执⾏许多任务。GLUE 和 DecaNLP是最近的例⼦
- 在BERT的基础上，多任务学习产⽣了改进
## Low-Resource Settings
- 不需要很多计算能⼒的模型(不能使⽤BERT)
  - 为移动设备尤其重要
- 低资源语⾔
- 低数据环境(few shot learning ⼩样本学习)
  - ML 中的元学习越来越受欢迎
## Interpreting/Understanding Models
- 我们能得到模型预测的解释吗？
- 我们能理解模型，例如BERT知道什么和他们为什么⼯作这么好？
- NLP中快速增⻓的地区
- 对于某些应⽤程序⾮常重要(如医疗保健)
## Diagnostic/Probing Classifiers
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081629946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 看看模型知道什么语⾔的信息
- 诊断分类器需要表示⼀个模型(例如BERT)作为输⼊，并做⼀些任务
- 只有诊断分类器被训练
- 诊断分类器通常⾮常简单(例如，单个softmax)。否则他们不通过模型表示来⾃省会学会完成任务
- ⼀些诊断任务
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081651693.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Diagnostic/ Probing Classifiers: Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213081700544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 低层的 BERT 在低层的任务中表现更好
## NLP in Industry
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021308171339.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- NLP是快速增⻓的⾏业。尤其是两⼤领域：
- 对话
  - 聊天机器⼈
  - 客户服务
- 健康
  - 理解健康记录
  - 理解⽣物医学⽂献
# Conclusion
- 在过去的5年⾥，由于深度学习，进步很快
- 由于较⼤的模型和更好地使⽤⽆标记数据，在去年有了更⼤的进展
  - 是在NLP领域的激动⼈⼼的时刻
- NLP是正逐渐对社会产⽣巨⼤影响⼒，使偏差和安全等问题越来越重要