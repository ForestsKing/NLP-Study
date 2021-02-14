# Lecture Plan
- Recap what we already know about NLG
- More on decoding algorithms
- NLG tasks and neural approaches to them
- NLG evaluation: a tricky situation
- Concluding thoughts on NLG research, current trends, and the future

Today we’ll be learning about what’s happening in the world of neural approaches to Natural Language Generation (NLG)

# Section 1: Recap: LMs and decoding algorithms
## Natural Language Generation (NLG)
- ⾃然语⾔⽣成指的是我们⽣成（即写⼊）新⽂本的任何设置
- NLG 包括以下成员：
  - 机器翻译
  - 摘要
  - 对话（闲聊和基于任务）
  - 创意写作：讲故事，诗歌创作
  - ⾃由形式问答（即⽣成答案，从⽂本或知识库中提取）
  - 图像字幕
## Recap
- 语⾔建模 是给定之前的单词，预测下⼀个单词的任务：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213200600737.png#pic_center)

- ⼀个产⽣这⼀概率分布的系统叫做 语⾔模型
- 如果系统使⽤ RNN，则被称为 RNN-LM
- 条件语⾔建模 是给定之前的单词以及⼀些其他输⼊$x$ ，预测下⼀个单词的任务：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213200623135.png#pic_center)

- 条件语⾔建模任务的例⼦：
  - 机器翻译 x=source sentence, y=target sentence
  - 摘要 x=input text, y=summarized text
  - 对话 x=dialogue history, y=next utterance
## Recap: training a (conditional) RNN-LM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213200656544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 这是神经机器翻译中的例⼦
- 在训练期间,我们将正确的(⼜名引⽤)⽬标句⼦输⼊解码器，⽽不考虑 解码器预测的。这种培训⽅法称为 Teacher Forcing
## Recap: decoding algorithms
- 问题：训练条件语⾔模型后，如何使⽤它⽣成⽂本？
- 答案：解码算法是⼀种算法，⽤于从语⾔模型⽣成⽂本
- 我们了解了两种解码算法
  - 贪婪解码
  - Beam 搜索
## Recap: greedy decoding
- ⼀个简单的算法
- 在每⼀步中，取最可能的单词（即argmax）
- 将其⽤作下⼀个单词，并在下⼀步中将其作为输⼊提供
- 继续前进，直到您产⽣ $<END>$ 或达到某个最⼤⻓度
- 由于缺乏回溯，输出可能很差（例如，不合语法，不⾃然，荒谬）

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320080773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Recap: beam search decoding
- ⼀种旨在通过⼀次跟踪多个可能的序列，找到⾼概率序列（不⼀定是最佳序列）的搜索算法
- 核⼼思想：在解码器的每⼀步，跟踪 k 个最可能的部分序列（我们称之为假设）
  - k是光束⼤⼩
- 达到某个停⽌标准后，选择概率最⾼的序列（考虑⼀些⻓度调整）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213200833643.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## What’s the effect of changing beam size k?
- ⼩的 k 与贪⼼解码有类似的问题（k = 1时就是贪⼼解码）
  - 不符合语法，不⾃然，荒谬，不正确
- 更⼤的 k 意味着您考虑更多假设
  - 增加 k 可以减少上述⼀些问题
  - 更⼤的 k 在计算上更昂贵
  - 但增加 k 可能会引⼊其他问题：
    - 对于NMT，增加 k 太多会降低BLEU评分(Tu et al, Koehnet al)
      - beam size 和 BLEU 之间存在最优性之间的区别，⾼概率序列和⾼的 BLEU 得分是两件独⽴的事情
      - 这主要是因为⼤ k 光束搜索产⽣太短的翻译（即使得分归⼀化）
    - 在闲聊话等开放式任务中，⼤的 k 会输出⾮常通⽤的句⼦（⻅下⼀张幻灯⽚）
## Effect of beam size in chitchat dialogue
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213200921738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 低 beam size
  - 更关于话题但是没有意义的
  - 语法差，重复的
- ⾼ beam size
  - 安全的反应
  - “正确”的反应
  - 但它是通⽤的，不太相关
## Sampling-based decoding
- 纯采样
  - 在每个步骤t，从概率分布 $P_t$中随机抽样以获取你的下⼀个单词。
  - 像贪婪的解码，但是是采样⽽不是argmax。
- Top-n 采样
  - 在每个步骤 t ，从 $P_t$的前 n 个最可能的单词中，进⾏随机采样（即若V = 10, n = 2，就相当于把选择范围限定在了概率排名前两个的单词，再在这两者之间做采样得到⼀个单词）
  - 与纯采样类似，但截断概率分布
  - 此时，n = 1 是贪婪搜索，n = V 是纯采样
  - 增加n以获得更多样化/⻛险的输出
  - 减少n以获得更通⽤/安全的输出
- 这两者都更多⽐光束搜索更有效率，不⽤跟踪多个假设


## Softmax temperature
- 回顾：在时间步 t ，语⾔模型通过对分数向量$s \in R^{|V|}$ 使⽤ softmax 函数计算出概率分布$P_t$
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320121097.png#pic_center)

- 你可以对 softmax 函数时候⽤温度超参数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213201223351.png#pic_center)

- 提⾼温度$\tau$ : $P_t$变得更均匀
  - 因此输出更多样化（概率分布在词汇中）
- 降低温度$\tau$ : $P_t$变得更尖锐
  - 因此输出的多样性较少（概率集中在顶层词汇上）


## Decoding algorithms: in summary
- 贪⼼解码 是⼀种简单的译码⽅法；给低质量输出
- Beam搜索 (特别是⾼beam⼤⼩)搜索⾼概率输出
  - ⽐贪婪提供更好的质量，但是如果 Beam 尺⼨太⼤，可能会返回⾼概率但不合适的输出(如通⽤的或是短的)
- 抽样⽅法 来获得更多的多样性和随机性
  - 适合开放式/创意代(诗歌,故事)
  - Top-n个抽样允许您控制多样性
- Softmax 温度控制 的另⼀种⽅式多样性
  - 它不是⼀个解码算法！这种技术可以应⽤在任何解码算法。
# Section 2: NLG tasks and neural approaches to them
>List of summarization datasets, papers, and codebases: https://github.com/mathsyouth/awesome-text-summarization

## Summarization: task definition
任务：给定输⼊⽂本x，写出更短的摘要 y 并包含 x 的主要信息

摘要可以是单⽂档，也可以是多⽂档
- 单⽂档意味着我们写⼀个⽂档 x 的摘要 y
- 多⽂档意味着我们写⼀个多个⽂档 $x_1,…,x_n$的摘要y

通常  $x_1,…,x_n$有重叠的内容：如对同⼀事件的新闻⽂章

在单⽂档摘要，数据集中的源⽂档具有不同⻓度和⻛格
- Gigaword: 新闻⽂章的前⼀两句  $rightarrow$标题(即句⼦压缩)
- LCSTS (中⽂微博)：段落 $rightarrow$ 句⼦摘要
- NYT, CNN/DailyMail: 新闻⽂章 $rightarrow$ (多个)句⼦摘要
- Wikihow (new!): 完整的 how-to ⽂章 $rightarrow$摘要句⼦

句⼦简化 是⼀个不同但相关的任务：将源⽂本改写为更简单（有时是更短）的版本
- Simple Wikipedia：标准维基百科句⼦ $rightarrow$ 简单版本
- Newsela：新闻⽂章 $rightarrow$ 为⼉童写的版本

## Summarization: two main strategies
### 抽取式摘要 Extractive summarization
- 选择部分(通常是句⼦)的原始⽂本来形成摘要
  - 更简单
  - 限定性的（⽆需解释）
### 抽象式摘要 Abstractive summarization
- 使⽤⾃然语⾔⽣成技术 ⽣成新的⽂本
  - 更困难
  - 更多变（更⼈性化）
## Pre-neural summarization
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320185040.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

>Diagram credit: Speech and Language Processing, Jurafsky and Martin

- Pre-neural摘要系统⼤多是抽取式的
- 类似Pre-neural MT，他们通常有⼀个流⽔线
  - 内容选择 Content selection：选择⼀些句⼦
  - 信息排序 Information ordering：为选择的句⼦排序
  - 句⼦实现 Sentence realization：编辑并输出句⼦序列例如，简化、删除部分、修复连续性问题)

Pre-neural 内容选择 算法
- 句⼦得分函数 可以根据
  - 主题关键词，通过计算如tf-idf等
  - 特性，例如这句话出现在⽂档的哪⾥
- 图算法 将⽂档为⼀组句⼦(节点)，每对句⼦之间存在边
  - 边的权重与句⼦相似度成正⽐
  - 使⽤图算法来识别图中最重要的句⼦
## Summarization evaluation: ROUGE
ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213201937854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

>ROUGE: A Package for Automatic Evaluation of Summaries, Lin, 2004 http://www.aclweb.org/anthology/W04-1013

类似于 BLEU，是基于 n-gram 覆盖的算法，不同之处在于：
- 没有简洁惩罚
- 基于召回率 recall，BLEU 是基于准确率的
  - 可以说，准确率对于MT 来说是更重要的(通过添加简洁惩罚来修正翻译过短)，召回率对于摘要来说是更重要的(假设你有⼀个最⼤⻓度限制)，因为需要抓住重要的信息
- 但是，通常使⽤ F1(结合了准确率和召回率)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202012702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Neural summarization (2015 - present)
- 2015: Rush et al. publish the first seq2seq summarization paper
- 单⽂档摘要摘要是⼀项翻译任务！
- 因此我们可以使⽤标准的 seq2seq + attention NMT ⽅法
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202032247.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> A Neural Attention Model for Abstractive Sentence Summarization, Rush et al, 2015 https://arxiv.org/pdf/1509.00685.pdf

- ⾃2015年以来，有了更多的发展
  - 使其更容易复制
    - 也防⽌太多的复制
  - 分层/多层次的注意⼒机制
  - 更多的 全局/⾼级 的内容选择
  - 使⽤ RL 直接最⼤化 ROUGE 或者其他离散⽬标（例如⻓度）
  - 复兴 pre-neural 想法(例如图算法的内容选择)，把它们变成神经系统
## Neural summarization: copy mechanisms
- Seq2seq+attention systems 善于⽣成流畅的输出，但是不擅⻓正确的复制细节(如罕⻅字)
- 复制机制使⽤注意⼒机制，使seq2seq系统很容易从输⼊复制单词和短语到输出
  - 显然这是⾮常有⽤的摘要
  - 允许复制和创造给了我们⼀个混合了抽取/抽象式的⽅法
- There are several papers proposing copy mechanism variants:
  - Language as a Latent Variable: Discrete Generative Models for Sentence Compression, Miao et al, 2016 https://arxiv.org/pdf/1609.07317.pdf
  - Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond, Nallapati et al, 2016 https://arxiv.org/pdf/1602.06023.pdf
  - Incorporating Copying Mechanism in Sequence-to-Sequence Learning, Gu et al, 2016 https://arxiv.org/pdf/1603.06393.pdf
  - etc

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320214352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202148259.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> Get To The Point: Summarization with Pointer-Generator Networks, See et al, 2017 https://arxiv.org/pdf/1704.04368.pdf

- 复制机制的⼤问题
  - 他们复制太多
    - 主要是⻓短语，有时甚⾄整个句⼦
  - ⼀个原本应该是抽象的摘要系统，会崩溃为⼀个主要是抽取的系统
- 另⼀个问题
  - 他们不善于整体内容的选择，特别是如果输⼊⽂档很⻓的情况下
  - 没有选择内容的总体战略
## Neural summarization: better content selection
- 回忆：pre-neural摘要是不同阶段的内容选择和表⾯实现(即⽂本⽣成)
- 标准seq2seq + attention 的摘要系统，这两个阶段是混合在⼀起的
  - 每⼀步的译码器(即表⾯实现)，我们也能进⾏词级别的内容选择(注意⼒)
  - 这是不好的：没有全局内容选择策略
- ⼀个解决办法：⾃下⽽上的汇总
## Bottom-up summarization
- 内容选择阶段：使⽤⼀个神经序列标注模型来将单词标注为 include / don’t-include
- ⾃下⽽上的注意⼒阶段：seq2seq + attention 系统不能处理 don’t-include 的单词（使⽤ mask ）

简单但是⾮常有效！
- 更好的整体内容选择策略
- 减少⻓序列的复制(即更摘要的输出)
  - 因为⻓序列中包含了很多 don’t-include 的单词，所以模型必须学会跳过这些单词并将那些include 的单词进⾏摘要与组合
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320225092.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Neural summarization via Reinforcement Learning
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202302973.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 使⽤ RL 直接优化 ROUGE-L
  - 相⽐之下，标准的最⼤似然(ML)训练不能直接优化ROUGE-L，因为它是⼀个不可微函数
- 有趣的发现
  - 使⽤RL代替ML取得更⾼的ROUGE分数，但是⼈类判断的得分越低
- 混合模型最好
> A Deep Reinforced Model for Abstractive Summarization, Paulus et al, 2017 https://arxiv.org/pdf/1705.04304.pdf
Blog post: https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/

## Dialogue
“对话”包括各种各样的设置
- ⾯向任务的对话
  - 辅助 (如客户服务、给予建议，回答问题，帮助⽤户完成任务，如购买或预订)
  - 合作 (两个代理通过对话在⼀起解决⼀个任务)
  - 对抗 (两个代理通过对话完成⼀个任务)
- 社会对话
  - 闲聊 (为了好玩或公司)
  - 治疗/精神健康

## Pre- and post-neural dialogue
- 由于开放式⾃由NLG的难度，pre-neural对话系统经常使⽤预定义的模板，或从语料库中检索⼀个适当的反应的反应
- 摘要过去的研究，⾃2015年以来有很多论⽂将seq2seq⽅法应⽤到对话，从⽽导致⾃由对话系统兴趣重燃
- ⼀些早期seq2seq对话⽂章包括
  - A Neural Conversational Model, Vinyals et al, 2015
    - https://arxiv.org/pdf/1506.05869.pdf
  - Neural Responding Machine for Short-Text Conversation, Shang et al, 2015
    - https://www.aclweb.org/anthology/P15-1152
## Seq2seq-based dialogue
然⽽，很快他们就明⽩简单的应⽤标准seq2seq +attention 的⽅法在对话(闲聊)任务中有严重的普遍缺陷
- ⼀般性/⽆聊的反应
- ⽆关的反应(与上下⽂不够相关)
- 重复
- 缺乏上下⽂(不记得谈话历史)
- 缺乏⼀致的⻆⾊⼈格
## Irrelevant response problem
- 问题：seq2seq经常产⽣与⽤户⽆关的话语
  - 要么因为它是通⽤的(例如,“我不知道”)
  - 或因为改变话题为⽆关的⼀些事情
- ⼀个解决⽅案：不是去优化输⼊ S 到回答 T 的映射来最⼤化给定 S 的 T 的条件概率，⽽是去优化输⼊S 和回复 T 之间的最⼤互信息Maximum Mutual Information (MMI)，从⽽抑制模型去选择那些本来就很⼤概率的通⽤句⼦


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202502704.png#pic_center)

## Genericness / boring response problem
- 简单的测试时修复
  - 直接在Beam搜索中增⼤罕⻅字的概率
  - 使⽤抽样解码算法⽽不是Beam搜索
- 条件修复
  - ⽤⼀些额外的内容训练解码器(如抽样⼀些内容词并处理)
  - 训练 retrieve-and-refine 模型⽽不是 generate-from-scratch 模型
    - 即从语料库采样⼈类话语并编辑以适应当前的场景
    - 这通常产⽣更加多样化/⼈类/有趣的话语！
## Repetition problem
简单的解决⽅案
- 直接在 Beam 搜索中禁⽌重复n-grams
  - 通常⾮常有效
- 更复杂的解决⽅案
  - 在seq2seq中训练⼀个覆盖机制，这是客观的，可以防⽌注意⼒机制多次注意相同的单词
  - 定义训练⽬标以阻⽌重复
    - 如果这是⼀个不可微函数⽣成的输出，然后将需要⼀些技术例如RL来训练
## Lack of consistent persona problem
- 2016年，李等⼈提出了⼀个seq2seq对话模式，学会将两个对话伙伴的⻆⾊编码为嵌
  - ⽣成的话语是以嵌⼊为条件的
- 最近有⼀个闲聊的数据集称为PersonaChat，包括每⼀次会话的⻆⾊(描述个⼈特质的5个句⼦的集合)
  - 这提供了⼀种简单的⽅式，让研究⼈员构建 persona-conditional 对话代理
>A Persona-Based Neural Conversation Model, Li et al 2016, https://arxiv.org/pdf/1603.06155.pdf
Personalizing Dialogue Agents: I have a dog, do you have pets too?, Zhang et al, 2018 https://arxiv.org/pdf/1801.07243.pdf

## Negotiation dialogue
2017年，Lewis et al收集谈判对话数据集
- 两个代理协商谈判对话(通过⾃然语⾔)如何分配⼀组项⽬
- 代理对项⽬有不同的估值函数
- 代理⼈会⼀直交谈直到达成协议
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202655336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


>Deal or No Deal? End-to-End Learning for Negotiation Dialogues, Lewis et al, 2017 https://arxiv.org/pdf/1706.05125.pdf

- 他们发现⽤标准的最⼤似然(ML)来训练seq2seq系统的产⽣了流利但是缺乏策略的对话代理
- 和Paulus等的摘要论⽂⼀样，他们使⽤强化学习来优化离散奖励(代理⾃⼰在训练⾃⼰)
- RL 的基于⽬的的⽬标函数与 ML ⽬标函数相结合
- 潜在的陷阱：如果两两对话时，代理优化的只是RL⽬标，他们可能会偏离英语
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202718842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

在测试时，模型通过计算 rollouts，选择可能的反应：模拟剩余的谈话和预期的回报
- 2018年，Yarats 等提出了另⼀个谈判任务的对话模型，将策略和NLG⽅⾯分开
- 每个话语$x_t$ 都有⼀个对应的离散潜在变量$z_t$
- $z_t$学习成为⼀个很好的预测对话中的未来事件的预测器(未来的消息，策略的最终收获)，但不是$x_t$本身的预测器
- 这意味着 $z_t$学会代表$x_t$ 对对话的影响，⽽不是$x_t$ 的话
- 因此 $z_t$将任务的策略⽅⾯从 NLG⽅⾯分离出来
- 这对可控制性、可解释性和更容易学习策略等是有⽤的

>Hierarchical Text Generation and Planning for Strategic Dialogue, Yarats et al, 2018 https://arxiv.org/pdf/1712.05846.pdf

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213202905961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Conversational question answering: CoQA
- ⼀个来⾃斯坦福NLP的新数据集
- 任务：回答关于以⼀段对话为上下⽂的⽂本的问题
- 答案必须写摘要地(不是复制)
- QA / 阅读理解任务，和对话任务

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320292631.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


> CoQA: a Conversational Question Answering Challenge, Reddy et al, 2018 https://arxiv.org/pdf/1808.07042.pdf

## Storytelling
- 神经讲故事的⼤部分⼯作使⽤某种提示
  - 给定图像⽣成的故事情节段落
  - 给定⼀个简短的写作提示⽣成⼀个故事
  - 给定迄今为⽌的故事，⽣成故事的下⼀个句⼦（故事续写）
    - 这和前两个不同，因为我们不关⼼系统在⼏个⽣成的句⼦上的性能
- 神经故事⻜速发展
  - 第⼀个故事研讨会于2018年举⾏
  - 它举⾏⽐赛(使⽤五张图⽚的序列⽣成⼀个故事)
## Generating a story from an image
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203005349.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

有趣的是，这并不是直接的监督图像标题。没有配对的数据可以学习。
> Generating Stories about Images, https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed

- 问题：如何解决缺乏并⾏数据的问题
- 回答：使⽤⼀个通⽤的 sentence-encoding space
- Skip-thought 向量是⼀种通⽤的句⼦嵌⼊⽅法
- 想法类似于我们如何学通过预测周围的⽂字来学习单词的嵌⼊
- 使⽤ COCO (图⽚标题数据集)，学习从图像到其标题的 Skip-thought 编码的映射
- 使⽤⽬标样式语料库(Taylor Swift lyrics)，训练RNN-LM， 将Skip-thought向量解码为原⽂
- 把两个放在⼀起

> Skip-Thought Vectors, Kiros 2015, https://arxiv.org/pdf/1506.06726v1.pdf

## Generating a story from a writing prompt
- 2018年，Fan 等发布了⼀个新故事⽣成数据集 collected from Reddit’s WritingPrompts subreddit.
- 每个故事都有⼀个相关的简短写作提示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203106364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

Fan 等也提出了⼀个复杂的 seq2seq prompt-to-story 模型
> Hierarchical Neural Story Generation, Fan et al, 2018 https://arxiv.org/pdf/1805.04833.pdf

- convolutional-based
  - 这使它的速度⽐RNN-based seq2seq
- 封闭的多头多尺度的self-attention
  - self-attention 对于捕获远程上下⽂⽽⾔⼗分重要
  - ⻔控允许更有选择性的注意机制
  - 不同的注意⼒头在不同的尺度上注意不同的东⻄——这意味着有不同的注意机制⽤于检索细粒度和粗粒度的信息
- 模型融合
  - 预训练⼀个seq2seq模型，然后训练第⼆个 seq2seq 模型访问的第⼀个 model 的隐状态
  - 想法是，第⼀seq2seq模型学习通⽤LM，第⼆个model学习基于提示的条件

结果令⼈印象深刻
- 与提示相关
- 多样化，并不普通
- 在⽂体上戏剧性

但是
- 主要是氛围/描述性/场景设定，很少是事件/情节
- ⽣成更⻓时，⼤多数停留在同样的想法并没有产⽣新的想法——⼀致性问题

## Challenges in storytelling
由神经LM⽣成的故事听起来流畅…但是是曲折的，荒谬的，情节不连贯的

缺失的是什么？

LMs对单词序列进⾏建模。故事是事件序列
- 为了讲⼀个故事，我们需要理解和模拟
  - 事件和它们之间的因果关系结构
  - ⼈物，他们的个性、动机、历史、和其他⼈物之间的关系
  - 世界(谁、是什么和为什么)
  - 叙事结构(如说明 冲突 解决)
  - 良好的叙事原则(不要引⼊⼀个故事元素然后从未使⽤它)
## Event2event Story Generation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203221496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203227282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> Event Representations for Automated Story Generation with Deep Neural Nets, Martin et al, 2018
https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17046/15769

## Structured Story Generation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203244412.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> Strategies for Structuring Story Generation, Fan et al, 2019 https://arxiv.org/pdf/1902.01109.pdf

## Tracking events, entities, state, etc.
- 旁注：在神经NLU(⾃然语⾔理解)领域，已经有⼤量关于跟踪事件/实体/状态的⼯作
  - 例如，Yejin Choi’s group* 很多⼯作在这⼀领域
- 将这些⽅法应⽤到 NLG是更加困难的
  - 如果你缩⼩范围，则更可控的
  - 不采⽤⾃然语⾔⽣成开放域的故事，⽽是跟踪状态
  - ⽣成⼀个配⽅(考虑到因素)，跟踪因素的状态
## Tracking world state while generating a recipe
- 过程神经⽹络：给定因素，⽣成配⽅指示
- 显式跟踪所有因素的状态，并利⽤这些知识来决定下⼀步要采取什么⾏动
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203319207.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> Simulating Action Dynamics with Neural Process Networks, Bosselut et al, 2018 https://arxiv.org/pdf/1711.05313.pdf

## Poetry generation: Hafez
- Hafez：Ghazvininejad et al 的诗歌系统
- 主要思想：使⽤⼀个有限状态受体(FSA)来定义所有可能的序列，服从希望满⾜的节奏约束
- 然后使⽤FSA约束RNN-LM的输出

例如
- 莎⼠⽐亚的⼗四⾏诗是14⾏的iambic pentameter
- 所以莎⼠⽐亚的⼗四⾏诗的FSA是 $((01)^5)^{14}$
- 在Beam搜索解码中，只有探索属于FSA的假设

> Generating Topical Poetry, Ghazvininejad et al, 2016 http://www.aclweb.org/anthology/D16-1126
Hafez: an Interactive Poetry Generation System, Ghazvininejad et al, 2017 http://www.aclweb.org/anthology/P17-4008

- 全系统
- ⽤户提供主题字
- 得到⼀个与主题相关的词的集合
- 识别局部词语押韵。这将是每⼀⾏的结束
- 使⽤受制于FSA的RNN-LM⽣成这⾸诗
- RNN-LM向后(⾃右向左)。这是必要的,因为每⼀⾏的最后⼀个词是固定的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203454753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

在后续的⼀篇论⽂中，作者制作了系统交互和⽤户可控
控制⽅法很简单：在Beam搜索中，增⼤具有期望特征的单词的分数
## Poetry generation: Deep-speare
更多的诗歌⽣成的端到端⽅法(lau等)

三个组件
- 语⾔模型
- pentameter model
- rhyme model 韵律模型……

作为⼀个多任务学习问题共同学习
- 作者发现 meter 和押韵是相对容易的，但⽣成的诗歌上有些缺乏“情感和可读性”
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203526856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

> Deep-speare: A joint neural model of poetic language, meter and rhyme, Lau et al, 2018 http://aclweb.org/anthology/P18-1181

## Non-autoregressive generation for NMT
> Non-Autoregressive Neural Machine Translation, Gu et al, 2018 https://arxiv.org/pdf/1711.02281.pdf

- 2018年,顾等发表了“Non-autoregressive 神经机器翻译”模型
- 意义：它不是根据之前的每个单词，从左到右产⽣翻译
- 它并⾏⽣成翻译
- 这具有明显的效率优势，但从⽂本⽣成的⻆度来看也很有趣
- 架构是基于Transformer 的；最⼤的区别是，解码器可以运⾏在测试时并⾏

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203558610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# Section 3: NLG evaluation
## Automatic evaluation metrics for NLG
基于词重叠的指标(BLEU，ROUGE，METROR，F1，等等)
- 我们知道他们不适合机器翻译
- 对于摘要⽽⾔是更差的评价标准，因为摘要⽐机器翻译更开放
  - 不幸的是，与抽象摘要系统相⽐，提取摘要系统更受ROUGE⻘睐
- 对于对话甚⾄更糟，这⽐摘要更开放
  - 类似的例⼦还有故事⽣成
## Word overlap metrics are not good for dialogue
>How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for
Dialogue Response Generation, Liu et al, 2017 https://arxiv.org/pdf/1603.08023.pdf

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021320364211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

上图展示了 BLEU-2、Embedding average 和⼈类评价的相关性都不⾼
> Why We Need New Evaluation Metrics for NLG, Novikova et al, 2017 https://arxiv.org/pdf/1707.06875.pdf

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203656711.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Automatic evaluation metrics for NLG
- 困惑度？
  - 捕捉你的LM有多强⼤，但是不会告诉你关于⽣成的任何事情(例如，如果你的困惑度是未改变的，解码算法是不好的)
- 词嵌⼊基础指标？
  - 主要思想：⽐较词嵌⼊的相似度(或词嵌⼊的均值)，⽽不仅仅是重叠的单词。以更灵活的⽅式捕获语义。
  - 不幸的是,仍然没有与类似对话的开放式任务的⼈类判断，产⽣很好的联系
- 我们没有⾃动指标充分捕捉整体质量(即代表⼈类的质量判断)
- 但我们可以定义更多的集中⾃动度量来捕捉⽣成⽂本的特定⽅⾯
  - 流利性(使⽤训练好的LM计算概率)
  - 正确的⻛格(使⽤⽬标语料库上训练好的LM的概率)
  - 多样性(罕⻅的⽤词，n-grams 的独特性)
  - 相关输⼊(语义相似性度量)
  - 简单的⻓度和重复
  - 特定于任务的指标，如摘要的压缩率
- 虽然这些不衡量整体质量，他们可以帮助我们跟踪⼀些我们关⼼的重要品质
## Human evaluation
- ⼈类的判断被认为是⻩⾦标准
- 当然，我们知道⼈类评价是缓慢⽽昂贵的
- 但这些问题？
- 假如你获得⼈类的评估：⼈类评估解决你所有的问题吗？
- 不！
- 进⾏⼈类有效评估⾮常困难
- ⼈类
  - 是不⼀致的
  - 可能是不合逻辑的
  - 失去注意⼒
  - 误解了你的问题
  - 不能总是解释为什么他们会这样做
## Detailed human eval of controllable chatbots
- 在聊天机器⼈项⽬上⼯作的个⼈经验（PersonaChat）
- 我们研究了可控性（特别是控制所产⽣的话语，如重复，特异性，回应相关性和问题询问）

> What makes a good conversation? How controllable attributes affect human judgments, See et al, 2019 https://arxiv.org/pdf/1902.08654.pdf

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203819838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 如何要求⼈的质量判断？
- 我们尝试了简单的整体质量（多项选择）问题，例如：
  - 这次对话有多好？
  - 这个⽤户有多吸引⼈？
  - 这些⽤户中哪⼀个给出了更好的响应？
    - 您想再次与该⽤户交谈吗？
    - 您认为该⽤户是⼈还是机器⼈？

主要问题：
- 必然⾮常主观
- 回答者有不同的期望；这会影响他们的判断
- 对问题的灾难性误解（例如“聊天机器⼈⾮常吸引⼈，因为它总是回写”）
- 总体质量取决于许多潜在因素；他们应该如何被称重和/或⽐较？

最终，我们设计了⼀个详细的⼈类评价体系分离的重要因素，有助于整体chatbot质量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213203858156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

发现：
- 控制重复对于所有⼈类判断都⾮常重要
- 提出更多问题可以提⾼参与度
- 控制特异性（较少的通⽤话语）提⾼了聊天机器⼈的吸引⼒，趣味性和感知的听⼒能⼒。
- 但是，⼈类评估⼈员对⻛险的容忍度较低（例如⽆意义或⾮流利的输出）与较不通⽤的机器⼈相关联
- 总体度量“吸引⼒”（即享受）很容易最⼤化 - 我们的机器⼈达到了近乎⼈性化的表现
- 整体度量“⼈性化”（即图灵测试）根本不容易最⼤化 - 所有机器⼈远远低于⼈类表现
- ⼈性化与会话质量不⼀样！
- ⼈类是次优的会话主义者：他们在有趣，流利，倾听上得分很低，并且问的问题太少

## Possible new avenues for NLG eval?
- 语料库级别的评价指标
  - 度量应独⽴应⽤于测试集的每个示例，或整个语料库的函数
  - 例如，如果对话模型对测试集中的每⼀个例⼦回答相同的通⽤答案，它应该被惩罚
- 评估衡量多样性安全权衡的评估指标
- 免费的⼈类评估
  - 游戏化：使任务（例如与聊天机器⼈交谈）变得有趣，这样⼈类就可以为免费提供监督和隐式评估，作为评估指标
- 对抗性鉴别器作为评估指标
  - 测试NLG系统是否能愚弄经过训练能够区分⼈类⽂本和AI⽣成的⽂本的识别器
# Section 4: Thoughts on NLG research, current trends, and the future
## Exciting current trends in NLG
- 将离散潜在变量纳⼊NLG
  - 可以帮助在真正需要它的任务中建模结构，例如讲故事，任务导向对话等
- 严格的从左到右⽣成的替代⽅案
  - 并⾏⽣成，迭代细化，⾃上⽽下⽣成较⻓的⽂本
- 替代teacher forcing的最⼤可能性培训
  - 更全⾯的句⼦级别的⽬标函数（⽽不是单词级别）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213204007753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Neural NLG community is rapidly maturing
- 在NLP+深度学习的早期，社区主要将成功的⾮机动⻋交通⽅法转移到NLG任务中。
- 现在，越来越多的创新NLG技术出现，针对⾮NMT⽣成环境。
- 越来越多（神经）NLG研讨会和竞赛，特别关注开放式NLG
  - NeuralGen workshop
  - Storytelling workshop
  - Alexa challenge
  - ConvAI2 NeurIPS challenge
- 这些对于组织社区提⾼再现性、标准化评估特别有⽤
- 最⼤障碍是评估
## 8 things I’ve learnt from working in NLG
1. 任务越开放，⼀切就越困难
  - 约束有时是受欢迎的
2. 针对特定改进的⽬标⽐旨在提⾼整体⽣成质量更易于管理
3. 如果你使⽤⼀个LM作为NLG：改进LM（即困惑）最有可能提⾼⽣成质量
  - 但这并不是提⾼⽣成质量的唯⼀途径
4. 多看看你的输出
5. 你需要⼀个⾃动度量，即使它是不受影响的
  - 您可能需要⼏个⾃动度量
6. 如果你做了⼈⼯评估，让问题尽可能的集中
7. 在今天的NLP + 深度学习和 NLG中，再现性是⼀个巨⼤的问题。
  - 请公开发布所有⽣成的输出以及您的论⽂
8. 在NLG⼯作可能很令⼈沮丧，但也很有趣
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213204055253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
