# Lecture Plan
- A tiny bit of linguistics
- Purely character-level models
- Subword-models: Byte Pair Encoding and friends
- Hybrid character and word level models
- fastText
# 1. Human language sounds: Phonetics and phonology
- Phonetics 语⾳学是⼀种⾳流——物理学或⽣物学
- Phonology 语⾳体系假定了⼀组或多组独特的、分类的单元：phoneme ⾳素 或者是独特的特征
  - 这也许是⼀种普遍的类型学，但却是⼀种特殊的语⾔实现
  - 分类感知的最佳例⼦就是语⾳体系
    - ⾳位差异缩⼩；⾳素之间的放⼤

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021211043055.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Morphology: Parts of words
- 声⾳本身在语⾔中没有意义
- parts of words 是⾳素的下⼀级的形态学，是具有意义的最低级别

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110457641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 传统上，morphemes 词素是最⼩的语义单位 semantic unit
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110523916.png#pic_center)

- 深度学习:形态学研究较少；递归神经⽹络的⼀种尝试是 (Luong, Socher, & Manning 2013)
  - 处理更⼤词汇量的⼀种可能⽅法——⼤多数看不⻅的单词是新的形态(或数字)
## Morphology
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110547704.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⼀个简单的替代⽅法是使⽤字符 n-grams
  - Wickelphones (Rumelhart& McClelland 1986)
  - Microsoft’s DSSM (Huang, He, Gao, Deng, Acero, & Hect2013)
- 使⽤卷积层的相关想法
- 能更容易地发挥词素的许多优点吗？
## Words in writing systems
书写系统在表达单词的⽅式上各不相同，也不相同
- 没有分词（没有在单词间放置空格） 美国关岛国际机场及其办公室均接获
- ⼤部分的单词都是分开的：由单词组成了句⼦
  - 附着词 clitics
    - 分开的
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110655923.png#pic_center)

    - 连续的
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110704345.png#pic_center)
  - 复合名词
    - 分开的 life insurance company employee
    - 连续的 Lebensversicherungsgesellschaftsangestellter
## Models below the word level
- 需要处理数量很⼤的开放词汇：巨⼤的、⽆限的单词空间
  - 丰富的形态
  - ⾳译（特别是名字，在翻译中基本上是⾳译）
  - ⾮正式的拼写
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110819681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Character-Level Models
- 词嵌⼊可以由字符嵌⼊组成
  - 为未知单词⽣成嵌⼊
  - 相似的拼写共享相似的嵌⼊
  - 解决OOV问题
- 连续语⾔可以作为字符处理：即所有的语⾔处理均建⽴在字符序列上，不考虑 word-level
- 这两种⽅法都被证明是⾮常成功的！
  - 有点令⼈惊讶的是——传统上，⾳素/字⺟不是⼀个语义单元——但DL模型组成了组
  - 深度学习模型可以存储和构建来⾃于多个字⺟组的含义表示，从⽽模拟语素和更⼤单位的意义，从⽽汇总形成语义
## Below the word: Writing systems
⼤多数深度学习NLP的⼯作都是从语⾔的书⾯形式开始的——这是⼀种容易处理的、现成的数据
但是⼈类语⾔书写系统不是⼀回事！各种语⾔的字符是不同的！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212110908898.png#pic_center)

# 2. Purely character-level models
- 上节课我们看到了⼀个很好的纯字符级模型的例⼦⽤于句⼦分类
  - ⾮常深的卷积⽹络⽤于⽂本分类
  - Conneau, Schwenk, Lecun, Barrault.EACL 2017
- 强⼤的结果通过深度卷积堆叠
## Purely character-level NMT models
- 以字符作为输⼊和输出的机器翻译系统
- 最初，效果不令⼈满意
  - (Vilaret al., 2007; Neubiget al., 2013)
- 只有decoder（成功的）
  - (JunyoungChung, KyunghyunCho, YoshuaBengio. arXiv 2016).
- 然后有前景的结果
  - (Wang Ling, Isabel Trancoso, Chris Dyer, Alan Black, arXiv 2015)
  - (Thang Luong, Christopher Manning, ACL 2016)
  - (Marta R. Costa-Jussà, José A. R. Fonollosa, ACL 2016)
## English-Czech WMT 2015 Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111009931.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111017899.png#pic_center)

- Luong和Manning测试了⼀个纯字符级seq2seq (LSTM) NMT系统作为基线
- 它在单词级基线上运⾏得很好
- 对于 UNK，是⽤ single word translation 或者 copy stuff from the source
- 字符级的 model 效果更好了，但是太慢了
  - 但是在运⾏时需要3周的时间来训练，运⾏时没那么快
  - 如果放进了 LSTM 中，序列⻓度变为以前的数倍（⼤约七倍）
## Fully Character-Level Neural Machine Translation without Explicit Segmentation
Jason Lee, KyunghyunCho, Thomas Hoffmann. 2017.
编码器如下；解码器是⼀个字符级的GRU

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111049501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Stronger character results with depth in LSTM seq2seq model
Revisiting Character-Based Neural Machine Translation with Capacity and Compression. 2018. Cherry, Foster, Bapna, Firat, Macherey, Google AI
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111112734.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 在捷克语这样的复杂语⾔中，字符级模型的效果提升较为明显，但是在英语和法语等语⾔中则收效甚微。
- 模型较⼩时，word-level 更佳；模型较⼤时，character-level 更佳
# 3. Sub-word models: two trends
- 与word级模型相同的架构
  - 但是使⽤更⼩的单元:“word pieces”
  - [Sennrich, Haddow, Birch, ACL’16a], [Chung, Cho, Bengio, ACL’16].
- 混合架构
  - 主模型使⽤单词，其他使⽤字符级
  - [Costa-Jussà& Fonollosa, ACL’16], [Luong & Manning, ACL’16].
## Byte Pair Encoding
- BPE 并未深度学习的有关算法，但已成为标准且成功表示 pieces of words 的⽅法，可以获得⼀个有限的词典与⽆限且有效的词汇表。
- 最初的压缩算法
  - 最频繁的字节 ⼀个新的字节。
  - ⽤字符ngram替换字节(实际上，有些⼈已经⽤字节做了⼀些有趣的事情)
  - Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural Machine Translation of Rare Words with SubwordUnits. ACL 2016.
- 分词算法 word segmentation
  - 虽然做得很简单，有点像是⾃下⽽上的短序列聚类
  - 将数据中的所有的Unicode字符组成⼀个unigram的词典
  - 最常⻅的 ngram pairs 视为 ⼀个新的 ngram
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111240325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111251884.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111304487.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111313281.png#pic_center)

- 有⼀个⽬标词汇量，当你达到它的时候就停⽌
- 做确定性的最⻓分词分割
- 分割只在某些先前标记器(通常MT使⽤的 Moses tokenizer )标识的单词中进⾏
- ⾃动为系统添加词汇
  - 不再是基于传统⽅式的 strongly “word”
- 2016年WMT排名第⼀！仍然⼴泛应⽤于2018年WMT

## Wordpiece/Sentencepiece model
- ⾕歌NMT (GNMT) 使⽤了它的⼀个变体
  - V1: wordpiece model
  - V2: sentencepiece model
- 不使⽤字符的 n-gram count，⽽是使⽤贪⼼近似来最⼤化语⾔模型的对数似然函数值，选择对应的pieces
  - 添加最⼤限度地减少困惑的n-gram
- Wordpiece模型标记内部单词
- Sentencepiece模型使⽤原始⽂本
  - 空格被保留为特殊标记(_)，并正常分组
  - 您可以通过将⽚段连接起来并将它们重新编码到空格中，从⽽在末尾将内容反转
- BERT 使⽤了 wordpiece 模型的⼀个变体
  - (相对)在词汇表中的常⽤词
    - at, fairfax, 1910s
  - 其他单词由wordpieces组成
    - hypatia = h ##yp ##ati ##a
- 如果你在⼀个基于单词的模型中使⽤BERT，你必须处理这个
# 4. Character-level to build word-level
Learning Character-level Representations for Part-ofSpeech Tagging (Dos Santos and Zadrozny2014)
- 对字符进⾏卷积以⽣成单词嵌⼊
- 为PoS标签使⽤固定窗⼝的词嵌⼊

![在这里插入图片描述](https://img-blog.csdnimg.cn/202102121114514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Character-based LSTM to build word rep’ns
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111514193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Bi-LSTM构建单词表示
## Character-based LSTM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111533102.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Character-Aware Neural Language Models
Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. 2015
- ⼀个更复杂/精密的⽅法
- 动机
  - 派⽣⼀个强⼤的、健壮的语⾔模型，该模型在多种语⾔中都有效
  - 编码⼦单词关联性：eventful, eventfully, uneventful…
  - 解决现有模型的罕⻅字问题
  - ⽤更少的参数获得可⽐较的表达性

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111616876.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111628903.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Highway Network (Srivastavaet al. 2015)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111648699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 语法交互模型
- 在传递原始信息的同时应⽤转换
- 功能类似于LSTM内存单元
## Long Short-Term Memory Network

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111711612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 分级Softmaxto处理⼤的输出词汇表
- 使⽤ truncated backpropthrough time 进⾏训练
## Quantitative Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111733601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111743455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111755798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Take-aways
- 本⽂对使⽤词嵌⼊作为神经语⾔建模输⼊的必要性提出了质疑
- 字符级的 CNNs + Highway Network 可以提取丰富的语义和结构信息
- 关键思想：您可以构建“building blocks”来获得细致⼊微且功能强⼤的模型！
## Hybrid NMT
- Abest-of-both-worlds architecture
  - 翻译⼤部分是单词级别的
  - 只在需要的时候进⼊字符级别
- 使⽤⼀个复制机制，试图填充罕⻅的单词，产⽣了超过 2 BLEU的改进
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021211183750.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 2-stage Decoding
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111855189.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 单词级别的束搜索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212111915802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 字符级别的束搜索（遇到 $<UNK>$）时
- 混合模型与字符级模型相⽐
  - 纯粹的字符级模型能够⾮常有效地是⽤字符序列作为条件上下⽂
  - 混合模型虽然提供了字符级的隐层表示，但并没有获得⽐单词级别更低的表示
## English-Czech Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112001177.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Sample English-Czech translations
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112018108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112036609.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112104916.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112114250.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112128775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112139536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 5. Chars for word embeddings
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112200943.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

⼀种⽤于单词嵌⼊和单词形态学的联合模型(Cao and Rei 2016)
- 与w2v⽬标相同，但使⽤字符
- 双向LSTM计算单词表示
- 模型试图捕获形态学
- 模型可以推断单词的词根
## FastText embeddings
⽤⼦单词信息丰富单词向量
- ⽬标：下⼀代⾼效的类似于word2vecd的单词表示库，但更适合于具有⼤量形态学的罕⻅单词和语⾔
- 带有字符n-grams的 w2v 的 skip-gram模型的扩展
- 将单词表示为⽤边界符号和整词扩充的字符n-grams
- $where=<wh,whe,her,ere,re>, \ <where>$
  - 注意$<her>$ ， $<her>$是不同于$her$ 的
  - 前缀、后缀和整个单词都是特殊的
- 将word表示为这些表示的和。上下⽂单词得分为
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112426552.png#pic_center)
   - 细节：与其共享所有n-grams的表示，不如使⽤“hashing trick”来拥有固定数量的向量

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112500532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 罕⻅单词的差异收益

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212112509516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)