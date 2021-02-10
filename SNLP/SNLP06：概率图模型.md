# 1 概述

- 图模型：

  ![<img src="D:\picture\image-20210206105746773.png" alt="image-20210206105746773" style="zoom:80%;" />](https://img-blog.csdnimg.cn/20210210120147124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 概率图模型演变：
  ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-jhT15HrS-1612916381528)(D:\picture\image-20210206110001360.png)\]](https://img-blog.csdnimg.cn/20210210120207957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


- 生成式模型（产生式模型）与区分式模型（判别式模型）的本质区别在于观测序列 x 和状态序列 y 之间的决定关系。前者假设 y 决定 x ，后者假设 x 决定 y。

  - 生成式模型以“状态（输出）序列 y 按照一定规律生成观测（输入）序列 x ”为假设，针对联合分布 p(x,y) 进行建模，并且通过估计使生成概率最大的生成序列来获取 y。

    典型的生成式模型有：n 元语法模型，HMM，朴素的贝叶斯分类器，概率上下文无关文法

  - 判别式模型认为 y 由 x 决定，直接对后验概率 p(y | x) 进行建模，它从 x 中提取特征，学习模型参数，使得条件概率符合一定形式的最优。

    典型的判别式模型有：最大熵模型，条件随机场，支持向量机，最大熵马尔可夫模型，感知机 

# 2 贝叶斯网络

- 一个贝叶斯网络就是一个有向无环图，结点表示随机变量，可以是可观测量、隐含变量、未知参量或假设等；结点之间的有向边表示条件依存关系，箭头指向的结点依存于箭头发出的结点（父结点）。两个结点没有连接关系表示两个随机变量能够在某些特定情况下条件独立，而两个结点有连接关系表示两个随机变量在任何条件下都不存在条件独立。条件独立是贝叶斯网络所依赖的一个核心概念。每一个结点都与一个概率函数相关，概率函数的输入是该结点的父结点所表示的随机变量的一组特定值，输出为当前结点表示的随机变量的概率值。概率函数值的大小实际上表达的是结点之间依存关系的强度。假设父结点有n个布尔变量，概率函数可表示成由 $2^n$个条目构成的二维表，每个条目是其父结点各变量可能的取值（"T"或"F"）与当前结点真值的组合。
  - 表示
  - 推断
    - 精确推理方法（变量消除法，团树法）
    - 近似推理方法（重要性抽样法，随机马尔可夫链蒙特卡罗模拟法，循环信念传播法，泛化理念传播法）
  - 学习
    - 参数学习（最大似然估计法，最后后验概率法，期望最大化方法（EM），贝叶斯估计法）
    - 近似学习

- 例如，如果一篇文章是关于南海岛屿的新闻（将这 一事件记作"News"），文章可能包含介绍南海岛屿历史的内容（这一事件记作"History"），但一般不会有太多介绍旅游风光的内容（将事件"有介绍旅游风光的内容"作"Sightseeing"）。们可以构造一个简单的贝叶斯网络。

  ![<img src="D:\picture\image-20210206112145001.png" alt="image-20210206112145001" style="zoom:80%;" />](https://img-blog.csdnimg.cn/20210210120225496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


  "文章是关于南海岛屿的新闻"这一事件直接影响"有介绍旅游b内容"这一事件。如果分别用 N、H、S表示这三个事件，每个变量都有两种可能"T"（表示"有、是"或"包含"）和"F"（表示"没有"、"不是"或"不含"），于是可以对过件之间的关系用贝叶斯网络建模。

  三个事件的联合概率函数为：
$$
  P(H,S,N)=P(H|S,N)\times P(S|N)\times P(N)
$$
  基于这个模型，如果一篇文章中含有海南岛历史相关的内容，该文章是关于南海新闻的可能性有多大
$$
  P(N=T|H=T)=\frac{P(H=T,N=T)}{P(H=T)}\\=\frac{\sum_{S \in \{T,F\}}P(H=T,S,N=T)}{\sum_{N,S \in \{T,F\}}P(H=T,S,N)}\\=\frac{0.008_{TTT}+0.054_{TFT}}{0.008_{TTT}+0.054_{TFT}+0.256_{TTF}+0.24_{TFF}}\\
  =11.11\%
$$

# 3 马尔可夫模型

- 如果一个系统有 N 个状态 $S=s_1, s_2,…, s_N,$ 随着时间的推移，该系统从某一状态转移到另一状态。如果用 qt 表示系统在时间 t 的状态变量，那么，t 时刻的状态取值为$S_j(1\leq j\leq N) $的概率取决于前 t-1 个时刻 (1, 2, …, t-1) 的状态，该概率为：
  $$
  P(q_t=s_j|q_{t-1}=s_i,q_{t-2}=s_k,…)
  $$

- 一阶马尔可夫链：如果在特定情况下，系统在时间 t 的状态只与其在时间 t-1 的状态相关：

$$
  P(q_t=s_j|q_{t-1}=s_i,q_{t-2}=s_k,…)=P(q_t=s_j|q_{t-1}=s_i)
$$

- 马尔可夫模型：如果独立于时间 t 的随机过程，即所谓的不动性假设，状态与时间无关，那么:
  $$
  P(q_t=s_j|q_{t-1}=s_i)=a_{ij},1\leq i,j\leq N
  $$
  
- 例题：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210120251403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 马尔可夫链可以表示成状态图（转移弧上有概率的非确定的有限状态自动机)
  
  - 零概率的转移弧省略
  
  - 每个节点上所有发出弧的概率之和等于 1
  
  - 状态序列 $S_1, …, S_T $的概率
    $$
    P(q_1,q_2,…,q_T)=P(q_1)P(q_2|q_1)P(q_3|q_1,q_2)…P(q_T|q_1,q_2,…,q_{T-1})\\=P(q_1)P(q_2|q_1)P(q_3|q_2)…P(q_T|q_{T-1})\\=\pi_{q_1}\prod_{t=1}^{T-1}a_{q_tq_{t-1}}
    $$
    其中$\pi_{q_1}=P(q_1)$
  
  ![- <img src="D:\picture\image-20210207152244195.png" alt="image-20210207152244195" style="zoom:50%;" />](https://img-blog.csdnimg.cn/20210210120313911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- n元语法模型就是n-1阶马尔可夫模型

# 4 隐马尔可夫模型

![<img src="D:\picture\image-20210207152615555.png" alt="image-20210207152615555" style="zoom:67%;" />](https://img-blog.csdnimg.cn/20210210120327319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



- HMM 的组成
  - 模型中的状态数为 N
  - 从每一个状态可能输出的不同的符号数 M
  - 状态转移概率矩阵 $A=a_{ij}$
  - 从状态 $S_j $观察到某一特定符号 $v_k$的概率分布矩阵为：$B=b_j(k)$
  - 初始状态的概率分布为：$\pi = \{\pi_i\}$
  - 为了方便，一般将 HMM 记为：$\mu=(S,K,A,B,\pi)$或者$\mu=(A,B,\pi)$ 用以指出模型的参数集合，S为状态的集合，K为输出符号的集合。

- 给定模型 $\mu=(A,B,\pi)$，产生观察序列 $O＝O_1O_2 …O_T$ :
  (1)令 t =1;
  (2)根据初始状态分布$\pi_i$ 选择初始状态 $q_1=s_i$;
  (3)根据状态 $S_i $的输出概率分布 $b_i(k)$, 输出$O_t=v_k$ ;
  (4)根据状态转移概率$a_{ij}$ ，转移到新状态$q_{t+1}=s_j$ ;
  (5) t = t+1, 如果 t < T, 重复步骤 (3) (4), 否则结束。

- 三个问题：
  - 估计问题：在给定模型$\mu=(A,B,\pi)$ 和观察序列 $O＝O_1O_2 …O_T$ 的情况下，怎样快速计算概率$ p(O|\mu)$？
  - 序列问题：在给定模型 $\mu=(A,B,\pi)$ 和观察序列 $O＝O_1O_2 …O_T$ 的情况下，如何选择在一定意义下“最优”的状态序列 $Q＝q_1q_2 …q_T$ ，使得该状态序列“最好地解释”观察序列？
  - 训练问题\参数估计问题：给定一个观察序列$O＝O_1O_2 …O_T$，如何根据最大似然估计来求模型的参数值？即如何调节模型$\mu=(A,B,\pi)$的参数，使得$ p(O|\mu)$最大？