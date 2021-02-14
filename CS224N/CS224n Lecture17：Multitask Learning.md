## The Natural Language Decathlon: Multitask Learning as Question Answering
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234607839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## The Limits of Single-task Learning
- 鉴于{dataset，task，model，metric}，近年来性能得到了很⼤改善
- 只要$|dataset|>1000 \tines C$ ，我们就可以得到当前的最优结果 (C是输出类别的个数)
- 对于更⼀般的 Al，我们需要在单个模型中继续学习
- 模型通常从随机开始，仅部分预训练

## Pre-training and sharing knowledge is great!
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234709210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Why has weight & model sharing not happened as much in NLP?
- NLP需要多种推理：逻辑，语⾔，情感，视觉，++
- 需要短期和⻓期记忆
- NLP被分为中间任务和单独任务以取得进展
  - 在每个社区中追逐基准
- ⼀个⽆⼈监督的任务可以解决所有问题吗？不可以
- 语⾔显然需要监督
## Why a unified multi-task model for NLP?
- 多任务学习是⼀般NLP系统的阻碍
- 统⼀模型可以决定如何转移知识（领域适应，权重分享，转移和零射击学习）
- 统⼀的多任务模型可以
  - 更容易适应新任务
  - 简化部署到⽣产的时间
  - 降低标准，让更多⼈解决新任务
  - 潜在地转向持续学习
## How to express many NLP tasks in the same framework?
- 序列标记
  - 命名实体识别，aspect specific sentiment
- ⽂字分类
  - 对话状态跟踪，情绪分类
- Seq2seq
  - 机器翻译，总结，问答
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234814950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## The Natural Language Decathlon (decaNLP)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234826949.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 把 10 项不同的任务都写成了 QA 的形式，进⾏训练与测试
## Multitask Learning as Question Answering
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213234842829.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- Meta-Supervised learning 元监督学习 ：$From \{x,y\}\ to\ \{x,t,y\}\ (t\  is\  the\  task)$
- 使⽤问题 q 作为任务 t 的⾃然描述，以使模型使⽤语⾔信息来连接任务
- y 是 q 的答案，x 是回答 q 所必需的上下⽂
## Designing a model for decaNLP
需求：
- 没有任务特定的模块或参数，因为我们假设任务ID是未提供的
- 必须能够在内部进⾏调整以执⾏不同的任务
- 应该为看不⻅的任务留下零射击推断的可能性
## A Multitask Question Answering Network for decaNLP
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235019241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021323502610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 以⼀段上下⽂开始
- 问⼀个问题
- ⼀次⽣成答案的⼀个单词，通过
  - 指向上下⽂
  - 指向问题
  - 或者从额外的词汇表中选择⼀个单词
- 每个输出单词的指针切换都在这三个选项中切换
## Multitask Question Answering Network (MQAN)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235052726.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- For code and leaderboard see www.decaNLP.com
- 固定的 GloVe 词嵌⼊ + 字符级的 n-gram 嵌⼊ Linear Shared BiLSTM with skip connection
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235109133.png#pic_center)

- 从⼀个序列到另⼀个序列的注意⼒总结，并通过跳过连接再次返回
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235119471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 分离BiLSTM以减少维数，两个变压器层，另⼀个BiLSTM
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235130258.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- ⾃回归解码器使⽤固定的 GloVe 和字符 n-gram 嵌⼊，两个变压器层和⼀个LSTM层来参加编码器最后三层的输出
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235143137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- LSTM解码器状态⽤于计算上下⽂与问题中的被⽤作指针注意⼒分布问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235153980.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 对上下⽂和问题的关注会影响两个开关：
  - gamma决定是复制还是从外部词汇表中选择
  - lambda决定是从上下⽂还是在问题中复制

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235209237.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235215960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235230610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- S2S 是 seq2seq
- +SelfAtt = plus self attention
- +CoAtt = plus coattention
- +QPtr = plus question pointer == MQAN
- Transformer 层在单任务和多任务设置中有 收益
- 多任务训练⼀开始会获得很差的效果（⼲扰和遗忘），但是如果顺序训练这些任务，将很快就会好起来
- QA和SRL有很强的关联性
- 指向问题⾄关重要
- 多任务处理有助于实现零射击
- 组合的单任务模型和单个多任务模型之间存在差距
## Training Strategies: Fully Joint
简单的全联合训练策略
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235308291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235314262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Training Strategies: Anti-Curriculum Pre-training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235324831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235328866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235333135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235338283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235347446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 困难：在单任务设置中收敛多少次迭代
- 带红⾊的任务：预训练阶段包含的任务
- QA 的 Anti-curriculum 反课程预训练改进了完全联合培训
- 但MT仍然很糟糕
## Closing the Gap: Some Recent Experiments
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235405352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235412157.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## Where MQAN Points
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235423109.png#pic_center)

- 答案从上下⽂或问题中正确的复制
- 没有混淆模型应该执⾏哪个任务或使⽤哪个输出空间
## Pretraining on decaNLP improves final performance
- 例如额外的 IWSLT language pairs
- 或者是新的类似 NER 的任务
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235442961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## Zero-Shot Domain Adaptation of pretrained MQAN:
- 在 Amazon and Yelp reviews 上获得了 80% 的 精确率
- 在 SNLI 上获得了 62% （参数微调的版本获得了 87% 的精确率，⽐使⽤随机初始化的⾼ 2%）
## Zero-Shot Classification
- 问题指针使得我们可以处理问题的改变（例如，将标签转换为满意/⽀持和消极/悲伤/不⽀持）⽽⽆需任何额外的微调
- 使模型⽆需训练即可响应新任务
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021021323551639.png#pic_center)


## decaNLP: A Benchmark for Generalized NLP
- 为多个NLP任务训练单问题回答模型
- 解决⽅案
  - 更⼀般的语⾔理解
  - 多任务学习
  - 领域适应
  - 迁移学习
  - 权重分享，预训练，微调（对于NLP的ImageNet-CNN？）
  - 零射击学习
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235545420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210213235550647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

https://einstein.ai