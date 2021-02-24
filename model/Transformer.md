# 0 代码：
- [Transformer](https://github.com/ForestsKing/NLP-Study/blob/master/demo/Transformer.ipynb)
- [BERT](https://github.com/ForestsKing/NLP-Study/blob/master/demo/BERT.ipynb)

# 1 Transformer
## 1.1 Self-Attention
### 1.1.1 基本原理
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103212595.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103256829.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103522933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103545502.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103603724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103632766.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103647521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
### 1.1.2 矩阵表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103733233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103752645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103816251.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103840388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103912257.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
### 1.1.3 多头注意力
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224103950523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224104015318.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224104029736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 1.2 Positional Encoding
在self-attention中并没有位置信息，所以需要添加上位置信息。
在词嵌入后添加one-hot向量$p^i$。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224104320554.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224104419413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 1.3 Transformer
Transformer 模型主要分为两大部分，分别是 Encoder 和 Decoder。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224110602658.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224111522678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

编码组件部分由一堆编码器（encoder）构成（论文中是将6个编码器叠在一起）。解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224110852558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
所有的编码器在结构上都是相同的，但它们没有共享参数。每个解码器都可以分解成两个子层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224110942636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
从编码器输入的句子首先会经过一个自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。我们将在稍后的文章中更深入地研究自注意力。

自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

在每个编码器中的每个子层（自注意力、前馈网络）的周围都有一个残差连接，并且都跟随着一个“层-归一化”步骤。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224111259459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224111209670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 2 ELMo
ELMo 是 Embedding from Language Model 的缩写，它通过无监督的方式对语言模型进行预训练来学习单词表示

它的思路是用深度的双向 Language Model 在大量未标注数据上训练语言模型，如下图所示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224115831822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102241159324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022411595527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

# 3 GPT
Generative Pre-training Transformer

GPT 得到的语言模型参数不是固定的，它会根据特定的任务进行调整（通常是微调），这样的到的句子表示能更好的适配特定任务。它的思想也很简单，使用单向 Transformer 学习一个语言模型，对句子进行无监督的 Embedding，然后根据具体任务对 Transformer 的参数进行微调。GPT 与 ELMo 有两个主要的区别：

- 模型架构不同：ELMo 是浅层的双向 RNN；GPT 是多层的 transformer encoder
- 针对下游任务的处理不同：ELMo 将词嵌入添加到特定任务中，作为附加功能；GPT 则针对所有任务微调相同的基本模型

## 3.1 无监督的 Pretraining
这里解释一下上面提到的单向 Transformer。在 Transformer 的文章中，提到了 Encoder 与 Decoder 使用的 Transformer Block 是不同的。在 Decoder Block 中，使用了 Masked Self-Attention，即句子中的每个词都只能对包括自己在内的前面所有词进行 Attention，这就是单向 Transformer。GPT 使用的 Transformer 结构就是将 **Encoder** 中的 **Self-Attention** 替换成了 **Masked Self-Attention**，具体结构如下图所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224122314963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
这里的位置编码没有使用传统 Transformer 固定编码的方式，而是动态学习的

## 3.2 监督的 Fine-Tuning
Pretraining 之后，我们还需要针对特定任务进行 Fine-Tuning。
## 3.3 改造OpenAI GPT用于下游NLP任务
针对不同任务，需要简单修改下输入数据的格式，例如对于相似度计算或问答，输入是两个序列，为了能够使用 GPT，我们需要一些特殊的技巧把两个输入序列变成一个输入序列
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224122943272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- Classification：对于分类问题，不需要做什么修改
- Entailment：对于推理问题，可以将先验与假设使用一个分隔符分开
- Similarity：对于相似度问题，由于模型是单向的，但相似度与顺序无关，所以要将两个句子顺序颠倒后，把两次输入的结果相加来做最后的推测
- Multiple-Choice：对于问答问题，则是将上下文、问题放在一起与答案分隔开，然后进行预测

# 4 BERT
Bidirectional **Encoder** Representation from **Transformer**

Google 在预训练 BERT 时让它同时进行两个任务：

1. 漏字填空（完型填空），学术点的说法是 Masked Language Model
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224112812607.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
2. 判断第 2 个句子在原始本文中是否跟第 1 个句子相接（Next Sentence Prediction）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224112911143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
## 4.1 Masked Language Model
在 BERT 中，Masked LM（Masked Language Model）构建了语言模型，简单来说，就是随机遮盖或替换一句话里面的任意字或词，然后让模型通过上下文预测那一个被遮盖或替换的部分，之后做 Loss 的时候也只计算被遮盖部分的 Loss，这其实是一个很容易理解的任务，实际操作如下：

- 随机把一句话中 15% 的 token（字或词）替换成以下内容：
  - 这些 token 有 80% 的几率被替换成 [MASK]，例如 my dog is hairy$\rightarrow$my dog is [MASK]
  - 有 10% 的几率被替换成任意一个其它的 token，例如 my dog is hairy$\rightarrow$my dog is apple
  - 有 10% 的几率原封不动，例如 my dog is hairy$\rightarrow$my dog is hairy
- 之后让模型预测和还原被遮盖掉或替换掉的部分，计算损失的时候，只计算在第 1 步里被随机遮盖或替换的部分，其余部分不做损失，其余部分无论输出什么东西，都无所谓

这样做的好处是，BERT 并不知道 [MASK] 替换的是哪一个词，而且任何一个词都有可能是被替换掉的，比如它看到的 apple 可能是被替换的词。这样强迫模型在编码当前时刻词的时候不能太依赖当前的词，而要考虑它的上下文，甚至根据上下文进行 "纠错"。比如上面的例子中，模型在编码 apple 时，根据上下文 my dog is，应该把 apple 编码成 hairy 的语义而不是 apple 的语义
## 4.2 Next Sentence Prediction
我们首先拿到属于上下文的一对句子，也就是两个句子，之后我们要在这两个句子中加一些特殊的 token：

> [CLS]上一句话[SEP]下一句话[SEP]

也就是在句子开头加一个 [CLS]，在两句话之间和句末加 [SEP]，具体地如下图所示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224113413151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

可以看到，上图中的两句话明显是连续的。如果现在有这么一句话 

> [CLS]我的狗很可爱[SEP]企鹅不擅长飞行[SEP]

可见这两句话就不是连续的。在实际训练中，我们会让这两种情况出现的数量为 1:1

- Token Embedding 就是正常的词向量，即 PyTorch 中的 nn.Embedding()
- Segment Embedding 的作用是用 embedding 的信息让模型分开上下句，我们给上句的 token 全 0，下句的 token 全 1，让模型得以判断上下句的起止位置
- Position Embedding 和 Transformer 中的不一样，不是三角函数，而是学习出来的

BERT 预训练阶段实际上是将上述两个任务结合起来，同时进行，然后将所有的 Loss 相加

## 4.3 Fine-Tuning
### 4.3.1 classification
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224113708430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
如果现在的任务是 classification，首先在输入句子的开头加一个代表分类的符号 [CLS]，然后将该位置的 output，丢给 Linear Classifier，让其 predict 一个 class 即可。整个过程中 Linear Classifier 的参数是需要从头开始学习的，而 BERT 中的参数微调就可以了

### 4.3.2 Slot Filling
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224113905333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
将句子中各个字对应位置的 output 分别送入不同的 Linear，预测出该字的标签。其实这本质上还是个分类问题，只不过是对每个字都要预测一个类别

### 4.3.3 NLI
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224113946335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
即给定一个前提，然后给出一个假设，模型要判断出这个假设是 正确、错误还是不知道。这本质上是一个三分类的问题，和 Case 1 差不多，对 [CLS] 的 output 进行预测即可

### 4.3.4 QA
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224114213395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

举例来说，如上图，将一篇文章，和一个问题（这里的例子比较简单，答案一定会出现在文章中）送入模型中，模型会输出两个数 s,e，这两个数表示，这个问题的答案，落在文章的第 s 个词到第 e 个词。具体流程我们可以看下面这幅图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224114139253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
首先将问题和文章通过 [SEP] 分隔，送入 BERT 之后，得到上图中黄色的输出。此时我们还要训练两个 vector，即上图中橙色和蓝色的向量。首先将橙色和所有的黄色向量进行 dot product，然后通过 softmax，看哪一个输出的值最大，例如上图中$d_2$对应的输出概率最大，那我们就认为 s=2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224114401354.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
同样地，我们用蓝色的向量和所有黄色向量进行 dot product，最终预测得$d_3$的概率最大，因此 e=3。最终，答案就是 s=2,e=3

# 5 ELMo vs. GPT vs. BERT
ELMo 和 GPT 最大的问题就是传统的语言模型是单向的 ——我们根据之前的历史来预测当前词。但是我们不能利用后面的信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224115115871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
> 预训练模型架构的差异。 BERT使用双向Transformer。 OpenAI GPT使用从左到右的Transformer。 ELMo使用经过独立训练的从左到右和从右到左的LSTM的串联来生成用于下游任务的功能。在这三个之中，只有BERT表示在所有层中同时受左右上下文的制约。除了架构差异之外，BERT和OpenAI GPT是微调方法，而ELMo是基于特性的方法。