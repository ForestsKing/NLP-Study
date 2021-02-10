# 1 概率论基本概念
## 1.1 概率

- $P(A) \geq 0$
- $P(\Omega) = 1$
- $P( \cup A_i) = \sum_{i=0}^{\infty} A_i$

## 1.2 最大似然估计

$$
q_N(s_k) = \frac {n_N(s_k)}{N}
$$
其中$q_N(s_k)$是$s_k$发生的频率，$N$为实验次数，$n_N(s_k)$是$s_k$发生的次数。当$N$足够大时，频率近似为概率。

## 1.3 条件概率

$$
P(A | B)=\frac{P(A \cap B)}{P(B)}
$$

## 1.4 贝叶斯法则

易知：
$$
P(B | A)=\frac{P(B \cap A)}{P(A)}=\frac{P(A|B)P(B)}{P(A)}
$$


全概率公式：
$$
P(A)=\sum_{i=1}^n P(A|B_i)P(B_i)
$$
贝叶斯法则：
$$
P(B_j|A)=\frac{P(A|B_j)P(B_j)}{P(A)}=\frac{P(A|B_j)P(B_j)}{\sum_{i=1}^n P(A|B_i)P(B_i)}
$$

## 1.5 随机变量

- 随机变量：$p_i=P(X=a_i)$
- 分布函数：$F(x)=P(x \leq X)$

## 1.6 二项式分布

$$
p_i=(\begin{matrix}n \\ i \end{matrix})p^i(1-p)^{n-i}
$$

## 1.7 联合概率分布和条件概率分布

- 联合概率分布：$p_{ij}=P(X_1=a_i, X_2=b_j)$
- 条件概率分布：
  $$
  P(X_1=a_i| X_2=b_j)=\frac{P(X_1=a_i, X_2=b_j)}   {P(X_2=b_j)}=\frac{p_{ij}}{P(X_2=b_j)}=\frac{p_{ij}}{\sum_k p_{kj}}
  $$

## 1.8 贝叶斯决策理论

$$
P(w_i | x)=\frac{p(x|w_i)p(w_i)}{\sum_{j=1}^c p(x|w_j)p(w_j)}
$$

其中$x$为待分类的对象，$w_i$是类别。

- $P(w_i|x)=max\ P(w_j|x)$，那么$x\in w_i$

- $p(x|w_i)P(w_i)=max\ p(x|w_j)P(w_j)$，那么$x\in w_i$

- 若只有两类，$l(x)=\frac{p(x|w_1)}{p(x|w_2)}>\frac{p(w_2)}{p(w_1)}$，那么$x\in w_1$，否则$x\in w_2$。

  $l(x)$为似然比，$\frac{p(w_2)}{p(w_1)}$为似然比阈值。

## 1.9 期望和方差

- 期望：$E(X)=\sum_{k=1}^\infty x_k p_k$
- 方差：$var(X)=E((X-E(X))^2)=E(X^2)-E^2(X)$
- 标准差：$\sqrt{var(X)}$

# 2 信息论基本概念

## 2.1 熵

$$
H(X)=H(p)=-\sum p(x)\ log_2p(x)
$$

- 约定 $0log_20=0$
- 通常熵的单位为二进制位比特(bit)
- 熵又称为自信息。熵也可以被视为描述一个随机变量的不确定性的数量。一个随机变量的熵越大，它的不确定性越大。那么，正确估计其值的可能性就越小。越不确定的随机变量越需要大的信息量用以确定其值。
- 使熵值最大的概率分布最真实的反应了事件的分布情况。

## 2.2 联合熵与条件熵

- 联合熵：描述一对随机变量平均所需要的信息量
  $$
  H(X,Y)=-\sum_{x\in X}\sum_{y\in Y} p(x,y)logp(x,y)
  $$
  
- 条件熵：
  $$
  H(Y|X)=-\sum_{x\in X}\sum_{y\in Y} p(x,y)logp(y|x)
  $$
  
- 连锁规则：
  $$
  H(X,Y)=H(X)+H(Y|X)
  $$
  
- 熵率：对于一条长度为n的信息，每一个字符或字的熵
  $$
  H_{rate}=\frac{1}{n}H(X_{1n})=-\frac{1}{n}\sum_{x_{1n}}p(x_{1n})logp(x_{1n})
  $$
  其中变量$X_{1n}$表示随机变量序列$(X_1,X_2,X_3,…,X_n)$，$x_{1n}=(x_1,x_2,x_3,…,x_n)$表示随机变量的具体取值。有时将$x_{1n}$写成$x_{1}^n$ 。

- 语言$L=(X_i)$的熵率：
  $$
  H_{rate}(L)=lim_{n\rightarrow\infty}\frac{1}{n}H(X_1,X_2,…,X_n)
  $$
  

## 2.3 互信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210112850314.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)



- 互信息：是在知道了Y的值以后X的不确定性的减少量，即Y的值透露了多少关于X的信息量。互信息值越大，表示两个汉字之间的结合越紧密，越可能成词。反之，断开的可能性越大。
  $$
  I(X;Y)=H(X)-H(X|Y)=\sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)p(y)}
  $$

- 连锁规则：
  $$
  I(X;Y|Z)=I((X;Y)|Z)=H(X|Z)-H(X|Y,Z)
  $$

- 双字耦合度:效果比互信息好
  设 $c_i$，$c_{i+1}$是两个连续出现的汉字，统计样本中$c_i$，$c_{i+1}$连续出现在一个词中的次数和连续出现的总次数，二者之比就是$c_i$，$c_{i+1}$的双字耦合度。

- 两个完全相互依赖的变量之间的额互信息不是常量，而是取决于他两的熵 $I(X;X)=H(X)$

## 2.4 相对熵

- 又称交叉熵、KL距离。规定$0log(0/q)=0, \ plog(p/0)=\infty$
  $$
  D(p||q)=\sum_{x\in X}p(x)log\frac{p(x)}{q(x)}
  $$

- 表示成期望值：
  $$
  D(p||q)=E_P(log\frac{p(x)}{q(x)})
  $$
  
- 相对熵常被用以衡量两个随机分布的差距。当两个随机分布相同时，其相对熵为0。当两个随机分布的差别增加时，其相对熵也增加。

- 互信息实际上就是衡量一个联合分布于独立性差距多大的测度：
  $$
  I(X;Y)=D(p(x,y) || p(x)p(y))
  $$

- 连锁规则：
  $$
  D(p(y|x)||q(y|x))=\sum_{x}p(x)\sum_yp(y|x)log\frac{p(y|x)}{q(y|x)}
  $$

  $$
  D(p(x,y)||q(x,y))=D(p(x)||q(x))+D(p(y|x)||q(y|x))
  $$

## 2.5 交叉熵

- 如果一个随机变量 X ~ p(x)，q(x)为用于近似 p(x) 的概率分布，那么，随机变量 X 和模型 q 之间的交叉熵定义为：
  $$
  H(X,q)=H(X)+D(p||q)=-\sum_{x}p(x)logq(x)=E_p(log\frac{1}{q(x)})
  $$

- 对于语言L = (X) ~ p(x) 与其模型 q 的交叉熵定义为：
  $$
  H(L,q)=-lim_{n\rightarrow\infty}\frac{1}{n}\sum_{x_1^n}p(x_1^n)logq(x_1^n)
  $$
  其中$x_1^n=x_1,x_2,…,x_n$为L的词序列，$p(x_1^n)$为$x_1^n$的概率（理论值），$q(x_1^n)$为模型q对$x_1^n$的概率估计值。
  $$
  H(L,q)=-lim_{n\rightarrow\infty}\frac{1}{n}logq(x_1^n)
  $$

  $$
  H(L,q)\approx-\frac{1}{N}logq(x_1^N)
  $$

  

- 用以衡量估计模型与真实概率分布之间的差异，越小越好。

## 2.6 困惑度

在设计语言模型时，我们通常用困惑度来代替交叉熵衡量语言模型的好坏。给定语言L的样本$l_1^n=l_1,l_2,…,l_n$，L的困惑度$PP_q$定义为：
$$
PP_q=2^{H(L,q)}\approx2^{-\frac{1}{n}logq(l_1^n)}=[q(l_1^n)]^{-\frac{1}{n}}
$$

## 2.7 噪声信道模型

![<img src="D:\picture\image-20210202122328671.png" alt="image-20210202122328671" style="zoom: 67%;" />](https://img-blog.csdnimg.cn/20210210112955807.png#pic_center)


- 一方面要通过压缩消除所有的冗余，另一方面又要通过增加一定的可控冗余以保障输入信号经过噪声信道后可以很好地恢复原状。
- 信道容量 $C=max_{p(X)}I(X,Y)$

- $p(I)$语言模型，$p(O|I)$信道概率（翻译模型）
  $$
  I=argmax\ p(I|O)=argmax\ \frac{p(I)p(O|I)}{p(O)}=argmax\ p(I)p(O|I）
  $$

# 3 支持向量机

## 3.1 线性分类

- 两类问题
  $$
  f(x)=<w·x>+b=\sum_{i=1}^nw_ix_i+b
  $$
  方程式$<w·x>+b=0$定义的超平面将输入空间X分成两半，一半正类，一半负类。

- 多类问题
  $$
  c(x)=argmax(<w_i·x>+b)
  $$
  给每个类关联一个超平面，然后将新点x赋予超平面离其最远的那一类。

## 3.2 线性不可分

- 首先使用一个非线性映射函数将数据变换到一个特征空间F，然后在这个空间上使用线性分类器
  $$
  f(x)=\sum_{i=1}^Nw_i\varphi_i(x)+b
  $$
  $\varphi:x\rightarrow F$是从输入空间X到输出空间F的映射。

- 决策规则（分类函数）可以用测试点和训练点的内积来表示：
  $$
  f(x)=\sum_{i=1}^l\alpha_iy_i<\varphi(x_i),\varphi(x)>+b
  $$
  其中$l$为样本数目，$\alpha_i$为正值函数，$y_i$为类别标记

## 3.3 构造核函数

- 核是一个函数K，对所有的$x,z\in X$满足：
  $$
  K(x,z)=<\varphi(x)·\varphi(z)>
  $$
  这里的$\varphi$是从X到（内积）特征空间F的映射。

- $K(x,z)$是核函数的充分必要条件是矩阵
  $$
  K=(K(x_i,x_j))_{i,j=1}^n
  $$
  是半正定的（即特征值非负）

- 只要一个核函数满足Mercer条件，他就对应某一空间中的内积。