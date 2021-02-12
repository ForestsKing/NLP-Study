# 1.Lecture Plan
- Final project types and details; assessment revisited
- Finding research topics; a couple of examples
- Finding data
- Review of gated neural sequence models
- A couple of MT topics
- Doing your research
- Presenting your results and evaluation
## This lecture is still relevant ... Even if doing DFP
- At a lofty level
  - 了解⼀些关于做研究的知识是有好处的
- 我们将接触到:
  - 基线
  - 基准
  - 评估
  - 错误分析
  - 论⽂写作
这也是 默认最终项⽬ 的⼀⼤特点
# 2. Finding Research Topics
所有科学的两个基本出发点
- [钉⼦]从⼀个(领域)感兴趣的问题开始，并试图找到⽐⽬前已知的/使⽤的更好的⽅法来解决它。
- [锤⼦]从⼀个感兴趣的技术⽅法开始，找出扩展或改进它或应⽤它的新⽅法的好⽅法
## Project types
这不是⼀个详尽的列表，但⼤多数项⽬都是其中之⼀
- 找到感兴趣的应⽤程序/任务，探索如何有效地接近/解决它，通常应⽤现有的神经⽹络模型
- 实现了⼀个复杂的神经结构，并在⼀些数据上展示了它的性能
- 提出⼀种新的或变异的神经⽹络模型，并探讨其经验上的成功
- 分析项⽬。分析⼀个模型的⾏为：它如何表示语⾔知识，或者它能处理什么样的现象，或者它犯了什么样的错误
稀有的理论项⽬：显示模型类型、数据或数据表示的⼀些有趣的、重要的属性
## How to find an interesting place to start?
- Look at ACL anthology for NLP papers
- Also look at the online proceedings of major ML conferences
  - NeurIPS, ICML, ICLR •
- Look at past cs224n project
  - See the class website
- Look at online preprint servers, especially
- Even better: look for an interesting problem in the world
- ArxivSanity Preserver by Stanford grad Andrej Karpathy of cs231n
- Great new site –a much needed resource for this – lots of NLP tasks • Not always correct, though
## Finding a topic
> “If you see a research area where many people are working, go somewhere else.”

## Must-haves (for most* custom final projects)
- 合适的数据
  - 通常⽬标:10000 +标记的例⼦⾥程碑
- 可⾏的任务
- ⾃动评估指标
- NLP是项⽬的核⼼
# 3. Finding data
- 有些⼈会为⼀个项⽬收集他们⾃⼰的数据
  - 你可能有⼀个使⽤“⽆监督”数据的项⽬
  - 你可以注释少量的数据
  - 你可以找到⼀个⽹站，有效地提供注释，如喜欢，明星，评级等
- 有些⼈使⽤现有的研究项⽬或公司的数据
  - 如果你可以提供提交、报告等数据样本
- ⼤多数⼈使⽤现有的，由以前的研究⼈员建⽴的数据集
  - 你有⼀个快速的开始，有明显的前期⼯作和基线
## linguistic data consortium
语⾔数据联盟
- https://catalog.ldc.upenn.edu/
- Stanford licenses data; you can get access by signing up at:
  - https://linguistics.stanford.edu/resources/resources-corpora
- Treebanks, named entities, coreference data,lots of newswire, lots of speech with transcription, parallel MT data
  - Look at their catalog
  - Don’t use for non Stanford purposes!
## Machine translation
- lhttp://statmt.org
- 特别要注意各种 WMT 共享任务
## Dependency parsing: Universal Dependencies
https://universaldependencies.org
## Many, many more
现在⽹上有很多其他的数据集可以⽤于各种各样的⽬的
- 看Kaggle
- 看研究论⽂
= 看数据集列表
  - https://machinelearningmastery.com/datasets-natural-languageprocessing/
  - https://github.com/niderhoff/nlp-datasets
# 4. One more look at gated recurrent units and MT
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212165953673.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Backpropagation through Time
梯度消失问题⼗分严重
- 当梯度趋近于 0 时，我们⽆法判断
  - 数据中det 和 t+n 之间不再存在依赖关系
  - 参数设置错误（梯度消失条件）
- 这是原始转换函数的问题吗？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170022660.png#pic_center)

- 有了它，时间导数就会消失
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170041559.png#pic_center)

## Gated Recurrent Unit
- 这意味着错误必须通过所有中间节点反向传播
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170100464.png#pic_center)

- 或许我们可以创建快捷连接
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170116308.png#pic_center)


我们可以创建⾃适应的快捷连接
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021217013340.png#pic_center)

- 候选更新
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021217020627.png#pic_center)

- 更新⻔
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170213512.png#pic_center)

- $\Theta$表示逐元素的乘法
让⽹络⾃适应地修剪不必要的连接
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170406262.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170420744.png#pic_center)

- 候选更新
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170448747.png#pic_center)

- 重置⻔
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102121704571.png#pic_center)

- 更新⻔
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170504545.png#pic_center)

## 将RNN单元想象为⼀个微型计算机
tanh-RNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170519727.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


GRU
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170530617.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⻔控循环单位更现实
- 注意，在思想和注意⼒上有⼀些重叠

两个最⼴泛使⽤的⻔控循环单位：GRU和LSTM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170553679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## The LSTM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170626157.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- (绿⾊)LSTM⻔的所有操作都可以被遗忘/忽略，⽽不是把所有的东⻄都塞到其他所有东⻄上⾯
- (橙⾊)下⼀步的⾮线性更新就像⼀个RNN
- (紫⾊)这部分是核⼼（ResNets也是如此）不是乘，⽽是将⾮线性的东⻄和 $c_{t-1}$相加得到 $c_t$。$c_t, \ c_{t-1}$之间存在线性联络
# 5. The large output vocabulary problem in NMT (or all NLG)
Softmax 计算代价昂贵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170735118.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## The word generation problem
- 词汇⽣成问题
  - 词汇量通常适中:50K
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170756421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Possible approaches for output
- Hierarchical softmax : tree-structured vocabulary
- Noise-contrastive estimation : binary classification
- Train on a subset of the vocabulary at a time; test on a smart on the set of possible translations
  - 每次在词汇表的⼦集上进⾏训练，测试时⾃适应的选择词汇表的⼦集
  - Jean, Cho, Memisevic, Bengio. ACL2015
- Use attention to work out what you are translating
  - You can do something simple like dictionary lookup
  - 直接复制原句中的⽣词： “复制”模型
- More ideas we will get to : Word pieces; char. models
## MT Evaluation –an example of eval
- ⼈⼯(最好的!?)
  - Adequacy and Fluency 充分性和流畅性(5或7尺度)
  - 错误分类
  - 翻译排名⽐较（例如⼈⼯判断两个翻译哪⼀个更好）
- 在使⽤MT作为⼦组件的应⽤程序中进⾏测试
  - 如问答从外语⽂件
    - ⽆法测试翻译的很多⽅⾯(例如,跨语⾔IR)
- ⾃动度量
  - BLEU (双语评价替⼿)
  - Others like TER, METEOR, ……
## BLEU Evaluation Metric
- N-gram 精度(得分在0和1之间)
  - 参考译⽂中机器译⽂的 N-gram 的百分⽐是多少?
    - ⼀个n-gram是由n个单词组成的序列
  - 在⼀定的n-gram⽔平上不允许两次匹配相同的参考译⽂部分(两个MT单词airport只有在两个参考单词airport时才正确；不能通过输⼊“the the the the the”来作弊)
  - 也要⽤ unigrams 来计算单位的精度，等等
- 简洁惩罚 BP
  - 不能只输⼊⼀个单词“the”(精确度1.0!)
- ⼈们认为要“玩弄”这个系统是相当困难的。例如找到⼀种⽅法来改变机器的输出，使BLEU上升，但质量不会下降。
- BLEU是⼀个加权的⼏何平均值，加上⼀个简洁的惩罚因⼦
- 注意：只在语料库级起作⽤(0会杀死它)；句⼦级有⼀个平滑的变体
- 下图是 n-grams 1-4 的BLEU计算公式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212170954886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



## Initial results showed that BLEU predicts human judgments well

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212171007685.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Automatic evaluation of MT
- ⼈们开始优化系统最⼤化BLEU分数
  - BLEU分数迅速提⾼
  - BLEU和⼈类判断质量之间的关系⼀直下降
  - MT BLEU分数接近⼈类翻译但是他们的真实质量仍然远低于⼈类翻译
- 想出⾃动MT评估已经成为⾃⼰的研究领域
  - 有许多建议:TER, METEOR, MaxSim, SEPIA，我们⾃⼰的RTE-MT
T  - ERpA 是⼀个具有代表性的，好处理⼀些词的选择变化的度量
  - MT研究需要⼀些⾃动的度量，以允许快速的开发和评估
# 6. Doing your research example: Straightforward Class Project: Apply NNets to Task
- 定义任务
  - 示例：总结
- 定义数据集
  - 搜索学术数据集
    - 他们已经有基线
    - 例如 Newsroom Summarization Dataset https://summari.es
  - 定义你⾃⼰的数据(更难，需要新的基线)
    - 允许连接到你的研究
    - 新问题提供了新的机会
    - 有创意:Twitter、博客、新闻等等。有许多整洁的⽹站为新任务提供了创造性的机会
- 数据集卫⽣
  - 开始的时候，分离devtest and test
    - 接下来讨论更多
- 定义您的度量(s)
  - 在线搜索此任务的已建⽴的度量
  - 摘要: Rouge (Recall-Oriented Understudy for GistingEvaluation) ，它定义了⼈⼯摘要的n-gram重叠
  - ⼈⼯评价仍然更适合于摘要；你可以做⼀个⼩规模的⼈类计算
- 建⽴基线
  - ⾸先实现最简单的模型(通常对unigrams、bigrams 或平均字向量进⾏逻辑回归)
  - 在训练和开发中计算指标
  - 如果度量令⼈惊讶且没有错误，那么
    - 完成!问题太简单了。需要重启
- 实现现有的神经⽹络模型
  - 在训练和开发中计算指标
  - 分析输出和错误
  - 这⻔课的最低标准
- 永远要接近您的数据（除了最后的测试集）
  - 可视化数据集
  - 收集汇总统计信息
  - 查看错误
  - 分析不同的超参数如何影响性能
- 通过良好的实验设置，尝试不同的模型和模型变体，达到快速迭代的⽬的
  - Fixed window neural model
  - Recurrent neural network
  - Recursive neural network
  - Convolutional neural network
  - Attention-basedmodel
## Pots of data
- 许多公开可⽤的数据集都是使⽤train/dev/test结构发布的。我们都在荣誉系统上，只在开发完成时才运⾏测试集
- 这样的分割假设有⼀个相当⼤的数据集
- 如果没有开发集或者您想要⼀个单独的调优集，那么您可以通过分割训练数据来创建⼀个调优集，尽管您必须权衡它的⼤⼩/有⽤性与训练集⼤⼩的减少
- 拥有⼀个固定的测试集，确保所有系统都使⽤相同的⻩⾦数据进⾏评估。这通常是好的，但是如果测试集具有不寻常的属性，从⽽扭曲了任务的进度，那么就会出现问题。
## Training models and pots of data
- 训练时,模型过拟合
  - 该模型正确地描述了您所训练的特定数据中发⽣的情况，但是模式还不够通⽤，不适合应⽤于新数据
  - 监控和避免问题过度拟合的⽅法是使⽤独⽴的验证和测试集…

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212171351151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 您在⼀个训练集上构建(评价/训练)⼀个模型。
- 通常，然后在另⼀个独⽴的数据集上设置进⼀步的超参数，即调优集
  - 调优集是⽤来调整超参数的训练集
- 在开发集(开发测试集或验证集)上度量进度
  - 如果您经常这样做，就会过度适应开发集，所以最好有第⼆个开发集，即dev2set
- 只有最后,你评估和最终数据在⼀个测试集
  - ⾮常少地使⽤最终测试集……理想情况下只使⽤⼀次
- 培训、调优、开发和测试集需要完全不同
- 在训练所使⽤的数据集上进⾏测试是⽆效的
  - 您将得到⼀个错误的良好性能。我们通常训练时会过拟合
- 您需要⼀个独⽴的调优
  - 如果调优与train相同，则⽆法正确设置超参数
- 如果你⼀直运⾏在相同的评价集，你开始在评价集上过拟合
  - 实际上，你是在对评估集进⾏“训练”……你在学习那些对特定的评估集有⽤和没⽤的东⻄，并利⽤这些信息
- 要获得系统性能的有效度量，您需要另⼀个未经训练的独⽴测试集，即 dev2 和最终测试

我们需要意识到，每⼀次通过评估结果的变化⽽完成的调整，都是对数据集的拟合过程。我们需要对数据集的过拟合，但是不可以在独⽴测试集上过拟合，否则就失去了测试集的意义
## Getting your neural network to train
- 从积极的态度开始
  - 神经⽹络想要学习
    - 如果⽹络没有学习，你就是在做⼀些事情来阻⽌它成功地学习
- 认清残酷的现实
  - 有很多事情会导致神经⽹络完全不学习或者学习不好
  - 找到并修复它们(“调试和调优”)通常需要更多的时间，⽽不是实现您的模型
- 很难算出这些东⻄是什么
  - 但是经验、实验和经验法则会有所帮助！
## Models are sensitive to learning rates
From Andrej Karpathy, CS231n course notes
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021217151366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Models are sensitive to initialization
From Michael Nielsen http://neuralnetworksanddeeplearning.com/chap3.html
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212171539201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Training a (gated) RNN
- 使⽤LSTM或GRU：它使您的⽣活变得更加简单！
- 初始化递归矩阵为正交矩阵
- ⽤⼀个可感知的(⼩的)⽐例初始化其他矩阵
- 初始化忘记⻔偏差为1：默认记住
- 使⽤⾃适应学习速率算法：Adam, AdaDelta，…
- 梯度范数的裁剪：1-5似乎是⼀个合理的阈值，当与Adam 或 AdaDelta⼀起使⽤
- 要么只使⽤ dropout vertically，要么研究使⽤Bayesian dropout(Gal和gahramani -不在PyTorch中原⽣⽀持)
- 要有耐⼼！优化需要时间
## Experimental strategy
- 增量地⼯作！
- 从⼀个⾮常简单的模型开始
- 让它开始⼯作⼀个接⼀个地添加修饰物，让模型使⽤它们中的每⼀个(或者放弃它们)
- 最初运⾏在少量数据上
  - 你会更容易在⼀个⼩的数据集中看到bug
  - 像8个例⼦这样的东⻄很好
  - 通常合成数据对这很有⽤
  - 确保你能得到100%的数据
    - 否则你的模型肯定要么不够强⼤，要么是破碎的
- 在⼤型数据集中运⾏
  - 模型优化后的训练数据仍应接近100%
    - 否则，您可能想要考虑⼀种更强⼤的模式来过拟合训练数据
    - 对训练数据的过拟合在进⾏深度学习时并不可怕
      - 这些模型通常善于⼀般化，因为分布式表示共享统计强度，和对训练数据的过度拟合⽆关
- 但是，现在仍然需要良好的泛化性能
  - 对模型进⾏正则化，直到它不与dev数据过拟合为⽌
    - 像L2正则化这样的策略是有⽤的
    - 但通常Dropout是成功的秘诀
## Details matter!
- 查看您的数据，收集汇总统计信息
- 查看您的模型的输出，进⾏错误分析
- 调优超参数对于神经⽹络⼏乎所有的成功都⾮常重要
