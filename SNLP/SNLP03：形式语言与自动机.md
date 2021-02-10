# 1 基本概念

## 1.1 图

- 有向图
- 无向图
- 连通图
- 回路

## 1.2 树

- 森林：无回路的无向图
- 树：无回路的连通无向图
- 根树：有一个根节点的树
- 根节点、叶节点、中间结点
- 父节点、子节点、兄弟节点

## 1.3 字符串

- 字符串：假设$\Sigma$是字符的有限集合，一般称作字符表，它的每一个元素称为字符。由$\Sigma$中字符相连而成的有限序列称为$\Sigma$上的字符串。不包括任何字符的字符串称为空串，记作$\varepsilon$，包括空串在内的$ \Sigma $上的字符串全体记为$\Sigma^*$
- 字符串连接：$\Sigma=\{a,b,c\},x=abc,y=cba$则$xy=abccba,x^2=abcabc$
- 符号串集合的乘积：$A=\{aa,bb\},B=\{cc,dd,ee\} $则$AB=\{aacc,aadd,aaee,bbcc,bbdd,bbee\},A^2=\{aaaa,aabb,bbaa,bbbb\}$
- 闭包运算：$V=\{a,b\}$则$V^*=\{\varepsilon,a,b,aa,ab,bb,ba,aaa,…\},\ V^+=\{a,b,aa,ab,bb,ba,aaa,…\}$
- 用$|x|$表示字符串长度。

# 2 形式语言

## 2.1 概述

描述语言三种途径：

- 穷举法：只适合句子数目有限的语言。

- 语法描述：生成语言中合格的句子。（形式语法）

- 自动机：对输入的句子进行检验。（自动机）

## 2.2 形式语法的定义

- 形式语法：形式语法是一个4元组 $G=(N, \Sigma, P, S)$, 其中$ N$ 是非终结符的有限集合(有时也叫变量集或句法种类集)；$\Sigma$ 是终结符的有限集合，$N\cap\Sigma=\Phi$；$V=N\cup\Sigma$ 称总词汇表；$P$是一组重写规则的有限集合：$P={\alpha\rightarrow\beta}$，其中，$\alpha$，$\beta$ 是$ V $中元素构成的串，但 $\alpha$中至少应含有一个非终结符号；$S\in N$，称为句子符或初始符。

- 推导：按非平凡方式派生（+，至少一次），派生（*，可以零次）

  ![<img src="D:\picture\image-20210204093803139.png" alt="image-20210204093803139" style="zoom: 50%;" />](https://img-blog.csdnimg.cn/2021021011342721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


  最左推导、最右推导（规范推导）

- 句子：$ G=(N, \Sigma, P, S) $
  - $S$ 是一个句子形式；
  - 如果 $\gamma\beta\alpha$ 是一个句子形式，且$ \beta\rightarrow\delta$是$P $的产生式，则$\gamma\delta\alpha$也是一个句子形式；

- 语言：文法 $G $的不含非终结符的句子形式称为 $G $生成的句子。由文法 $G $生成的语言，记作$ L(G)$，指 $G $生成的所有句子的集合。即：$L(G)=\{x | x\in \Sigma, S \Rightarrow x \}$

## 2.3 形式语法的类型

- 正则文法（3型文法）：文法 $G=(N, \Sigma, P, S)$ 的$ P$ 中的规则满足如下形式：$A \rightarrow Bx$，或 $A \rightarrow x$，其中 $A, B \in N$， $x\in\Sigma$(左线性正则文法)。如果 $A \rightarrow xB$，则该文法称为右线性正则文法。
- 上下文无关文法CFG（2型文法）： $P$中的规则满足如下形式：$A \rightarrow \alpha$，其中$A\in N，\alpha \in (N \cup \Sigma )^*$。
- 上下文有关文法CSG（1型文法）：$ P $中的规则满足如下形式: $\alpha A \beta \rightarrow \alpha \gamma \beta$, 其 中 $A \in N, \alpha, \beta , \gamma \in (N \cup \Sigma )^*$，且$ \gamma$至少包含一个字符。
- 无约束文法（0型文法）：$ P $中的规则满足如下形式：$\alpha\rightarrow\beta$，其中$\alpha \in (N \cup \Sigma )^+,\beta \in (N \cup \Sigma )^*$
- $L(G_3) \subseteq L(G_2) \subseteq L(G_1) \subseteq L(G_0)$

## 2.4 CFG识别句子的派生树表示

- 构造步骤：CFG $G=(N, \Sigma, P, S)$ 产生一个句子的派生树由如下
  - 对于任意$x \in N \cup \Sigma $给一个标记作为节点, $S$ 作为树的根节点。
  - 如果一个节点的标记为 $A$，并且它至少有一个除它自身以外的后裔，则 $A \in N$。
  - 如果一个节点的标记为 $A$，它的 $k ( k > 0) $个直接后裔节点按从左到右的次序依次标记为$ A_1, A_2, …, A_k$，则$ A \rightarrow A_1,A_2,…A_k $一定是 $P $中的一个产生式。
- 二义性文法：一个文法$ G$，如果存在某个句子有不只一棵分析树与之对应，那么称这个文法是二义的。

# 3 自动机理论

|形式语言|自动机|
| ---- | ---- |
| 3型文法 | 有限自动机(FA) |
| 2型文法 | 下推自动机(PDA) |
| 1型文法 | 线性界限自动机 |
| 0型文法 | 图灵机 |

## 3.1 有限自动机
### 3.1.1 确定的有限自动机(DFA)

- 确定的有限自动机$ M $是一个五元组：
  $$
  M = (\Sigma, Q, \delta, q_0, F)
  $$
  - $\Sigma$是输入符号的有穷集合；
  - $Q$是状态的有限集合；
  - $q_0\in Q$是初始状态；
  - $F$ 是终止状态集合，$F \subseteq Q$； 
  - $\delta$ 是$Q$与$\Sigma$ 的直积$ Q\times\Sigma $到$Q$ (下一个状态) 的映射。它支配着有限状态控制的行为，有时也称为状态转移函数。

- 终止状态用双圈表示，起始状态用有“开始”标记的箭头表示

- 如果一个句子 $x $使得有限自动机$ M $有 $\delta(q_0, x) = p,\  p \in F$，那么，称句子$ x $被$ M$ 接受。由 $M $定义的语言 $T(M)$ 就是被$ M$ 接受的句子的全集。即：
  $$
  T(M) = \{ x | \delta(q_0, x) \in F \}
  $$

### 3.1.2 不确定的有限自动机(NFA)

- 不确定的有限自动机$ M $是一个五元组：
  $$
  M = (\Sigma, Q, \delta, q_0, F)
  $$
  - $\Sigma$是输入符号的有穷集合；
  - $Q$是状态的有限集合；
  - $q_0\in Q$是初始状态；
  - $F$ 是终止状态集合，$F \subseteq Q$； 
  - $\delta$ 是$Q$与$\Sigma$ 的直积$ Q\times\Sigma $到$Q$ 的幂集 $2^Q$的映射。

- NFA 与 DFA 的唯一区别是：在 NFA 中 $\delta(q, a)$ 是一个状态集合，而在 DFA 中 $\delta(q, a)$ 是一个状态。

- 如果存在一个状态$p$ 有 $p \in \delta(q_0, x)$，且$p\in F$那么，称句子$ x $被NFA $ M$ 接受。由NFA $M $定义的语言 $T(M)$ 就是被NFA $ M$ 接受的句子的全集。即：
  $$
  T(M) = \{ x |p \in \delta(q_0, x)且p \in F \}
  $$

设 L 是一个被 NFA 所接受的句子的集合，则存在一个 DFA，它能够接受 L 。

## 3.2 正则文法与自动机的关系

- 若$ G = (V_N,V_T, P, S )$是一个正则文法，则存在一个有限自动机FA $M=(\Sigma, Q, \delta, q_0, F)$，使得：$T(M) = L(G)$。
  - 令 $\Sigma ＝V_T, Q＝V_N\cup \{T\}，q_0＝S$，其中$T $是一个新增加的非终结符。
  - 如果在 P 中有产生式 $S \rightarrow \epsilon$，则 $F＝\{S, T\}$，否则$ F=\{T\}$。
  - 如果在 P 中有产生式 $B \rightarrow a， B\in V_N ，a \in V_T$, 则 $T \in  \delta(B, a)$。
  - 如果在 P 中有产生式 $B \rightarrow aC， B,C\in V_N ，a \in V_T$, 则 $C \in  \delta(B, a)$。
  - 对于每一个 $a \in V_T$，有 $\delta(T, a)=\Phi$。

- 若 $M=(\Sigma, Q, \delta, q_0, F)$ 是一有限自动机，则存在正则文法 $ G = (V_N,V_T, P, S )$ 使 $L(G)=T(M)$。
  - 令 $V_N = Q，V_T = \Sigma，S =q_0 $。
  - 如果 $C \in  \delta(B, a)，B,C\in Q，a \in \Sigma$，则在 P 中有产生式$B \rightarrow aC$。
  - 如果$C \in  \delta(B, a)，C\in F$，则在P 中有产生式$B \rightarrow a$。

## 3.3 上下文无关文法与下推自动机

- 下推自动机（PDA）：PDA 可以看成是一个带有附加的下推存储器的有限自动机，下推存储器是一个栈。

- 一个不确定的PDA可以表达成一个7元组：
  $$
  M = (\Sigma, Q, \Gamma, \delta, q, Z_0, F)
  $$
  - $\Sigma$ 是输入符号的有穷集合；
  - $Q $是状态的有限集合； 
  - $q0 \in Q$ 是初始状态；
  - $\Gamma$ 为下推存储器符号的有穷集合；
  - $Z_0 \in \Gamma$ 为最初出现在下推存储器顶端的符号;
  - $F \subseteq Q$ 是终止状态集合
  - $\delta$ 是从 $Q \times (\Sigma\cup\{\varepsilon\})\times\Gamma $ 到 $ Q\times\Gamma ^* $ 子集的映射。
  
- 映射关系 $\delta(q, a, Z)=\{(q_1, \gamma_1), (q_2, \gamma_2),…,(q_m, \gamma_m)\}$其中，$ q_1, q_2, …,q_m \in Q,\  a\in\Sigma,\  Z\in\Gamma,\  \gamma_1, \gamma_2,…,\gamma_m\in\Gamma ^*$。该映射的意思是：当PDA处于状态 q，面临输入符号a 时，自动机将进入 $q_i, i = 1, 2, …, m $状态，并以$ \gamma_i $来代替下推存储器(栈)顶端符号Z，同时将输入头指向下一个字符 。当 Z 被 $ \gamma_i $取代时，$ \gamma_i $ 的符号按照从左到右的顺序依次从下向上推入到存储器。
  特殊情况下，当$\delta(q, \varepsilon, Z)=\{(q_1, \gamma_1), (q_2, \gamma_2),…,(q_m, \gamma_m)\}$时，输入头位置不动，只用于处理下推存储器内部的操作，叫作 “$\varepsilon$ 移动”。
  
- 设有序对 $(q,\gamma),q\in Q,\gamma \in \Gamma ^*$，对于$a \in (\Sigma\cup\{\varepsilon\}),\ \gamma,\beta\in\Gamma ^*,\ z\in F$如果 $(q',\beta)\in\gamma(q,a,z),\ q,q'\in Q$，则表达式
  
  ![<img src="D:\picture\image-20210204111319544.png" alt="image-20210204111319544" style="zoom:67%;" />](https://img-blog.csdnimg.cn/20210210113523911.png#pic_center)

  
  表示根据下推自动机的状态变换规则，输入 a 能使下推自动机 M 由格局 $(q, Z\gamma)$ 变换到格局 $(q, \beta\gamma)$，或称为合法转移。零次或多次合法转移记为
  
  ![<img src="D:\picture\image-20210204111335391.png" alt="image-20210204111335391" style="zoom:67%;" />](https://img-blog.csdnimg.cn/20210210113536518.png#pic_center)


- 被PDA接受的标准：

  - 终止状态接受标准

    ![<img src="D:\picture\image-20210204111618078.png" alt="image-20210204111618078" style="zoom: 67%;" />](https://img-blog.csdnimg.cn/20210210113603896.png#pic_center)

    对于输入句子x，如果PDA从初始状态$q_0$开始转换到终止状态q时，x正好被读完，则认为x被PDA M所接受，而不管这时下推存储器的内容如何。
  
- 空存储器接受标准
  
    ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-lGC2LsBk-1612916290287)(D:\picture\image-20210204111814764.png)\]](https://img-blog.csdnimg.cn/20210210113621185.png#pic_center)

  对于给定的输入句子x，当输入头指向x的末端时，如果下推存储器变为空，则认为x被PDA M所接受，而不管这时PDA的状态q是否在终止状态集F中。

- 例题：![<img src="D:\picture\image-20210204112912210.png" alt="image-20210204112912210" style="zoom:80%;" />](https://img-blog.csdnimg.cn/20210210113720732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)


## 3.4 图灵机
- 一个图灵机T可以表达为一个六元组：
  $$
  T=(\Sigma,Q,\Gamma,\delta,q_0,F)
  $$
  - $\Sigma$ 是输入/输出带上符号的有穷集合，不包含空白字符B；
  - $Q $是状态的有限集合； 
  - $q0 \in Q$ 是初始状态；
  - $\Gamma$ 为输入符号的有穷集合，包含空字符B，$\Sigma\subseteq\Gamma,\Gamma=\Sigma\cup\{B\}$；
  - $F \subseteq Q$ 是终止状态集合；
  - $\delta$ 是从 $ Q\times\Gamma$到 $Q \times (\Gamma-\{B\}\times\{R,L,S\} $ 子集的映射,其中R,L,S分别表示右移一格，左移一格，停止不动。
  
- 图灵机T的一个格局可以用三元组$(q,\alpha,i)$表示，其中，$q\in Q$，$\alpha$是输入/输出带上非空白部分，$\alpha\in(\Gamma-\{B\})^*$，$i$是整数，表示T的读/写头到$\alpha$左端（起始位置）的距离。图灵机T通过如下转移动作引起格局的变化：假设$(q,A_1,A_2,…,A_n,i)$是当前T的一个格局$(1\leq i\leq n+1)$

  - 如果$\delta(q,A_i)=(p,X,R),1\leq i\leq n$，那么 T 的读/写头在 i 位置写入符号 X ，并将读/写头向右移动一个位置。

    ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-yG8iJvr8-1612916290292)(D:\picture\image-20210204120445446.png)\]](https://img-blog.csdnimg.cn/20210210113823771.png#pic_center)


  - 如果$\delta(q,A_i)=(p,X,L),2\leq i\leq n$，那么 T 的读/写头在 i 位置写入符号 X ，并将读/写头向z左移动一个位置，但不超出输入带的左端位置。

    ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-tqtVr3N8-1612916290297)(D:\picture\image-20210204120500367.png)\]](https://img-blog.csdnimg.cn/20210210113835676.png#pic_center)


  - 如果$i=n+1$，读写头超出原字符串的右端，读到的是空白字符B，此时如果有$\delta(q,B)=(p,X,R)$，那么 

    ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-KilAv9Qy-1612916290300)(D:\picture\image-20210204120523988.png)\]](https://img-blog.csdnimg.cn/20210210113849150.png#pic_center)


    如果有$\delta(q,B)=(p,X,L)$，那么
    
    ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-XCYnX3P0-1612916290303)(D:\picture\image-20210204120536834.png)\]](https://img-blog.csdnimg.cn/20210210113903274.png#pic_center)


- 如果T的两个格局X和Y之间的基本移动（包括不移动）的次数是有限的，并且相互关联，则可记为

  ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-9NdnLWbd-1612916290305)(D:\picture\image-20210204120549388.png)\]](https://img-blog.csdnimg.cn/20210210113915501.png#pic_center)


- 图灵机T所接受的语言定义为:

  ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-epMxLW4G-1612916290306)(D:\picture\image-20210204120618068.png)\]](https://img-blog.csdnimg.cn/2021021011392926.png#pic_center)


  给定一个识别语言L和图灵机T，当输入句子被接受时，T就停机，否则，T可能不停机。


## 3.5 线性界限自动机
- 一个线性界限自动机M可以表达为一个六元组：
  $$
  M=(\Sigma,Q,\Gamma,\delta,q_0,F)
  $$
  - $\Sigma$ 是输入/输出带上符号的有穷集合，$\Sigma\subseteq\Gamma$，包含特殊字符 # 和 \$ ，分别表示输入链的左端和右端结束标志。
  - $Q $是状态的有限集合； 
  - $q0 \in Q$ 是初始状态；
  - $\Gamma$ 为输入/输出带上符号的有穷集合
  - $F \subseteq Q$ 是终止状态集合；
  - $\delta$ 是从 $ Q\times\Gamma$到 $Q \times \Gamma\times\{R,L\} $ 子集的映射
  
- 线性带限自动机是一个确定的单带图灵机，其读写头不能超越原输入带上字符串的初始和终止位置，即线性带限自动机的存储空间被输入符号串的长度所限制。

- 接受的语言：

  ![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-isA8Oprp-1612916290308)(D:\picture\image-20210204121339868.png)\]](https://img-blog.csdnimg.cn/2021021011395189.png#pic_center)


各类自动机的主要区别是它们能够使用的信息存储空间的差异：有限状态自动机只能用状态来存储信息；下推自动机除了可以用状态以外，还可以用下推存储器(栈)；线性带限自动机可以利用状态和输入/输出带本身。因为输入/输出带没有“先进后出”的限制，因此，其功能大于栈；而图灵机的存储空间没有任何限制。
# 4 自动机在自然语言处理中的应用

## 4.1 单词拼写检查

- 编辑距离：设 X 为拼写错误的字符串，其长度为m, Y 为 X 对应的正确的单词(答案)，其长度为 n。则 X 和 Y 的编辑距离 $ed(X[m], Y[n])$ 定义为：从字符串 X转换到 Y 需要的插入、删除、替换和交换两个相邻的基本单位(字符)的最小个数。

- 假设 $Z = z_1 z_2 … z_p $为字母表 A上的p 个字母构成的字符串，$Z[j] $表示含有$j (j \geq 1)$ 个字符的子串。$X[m]$ 为拼写错误的字符串，其长度为m，Y[n] 为与X串接近的字符串(一个候选)，其长度为n。则给定两个串X 和Y的编辑距离$ed(X[m], Y[n]) $可以通过循环计算出从字符串X 转换到Y 需要进行插入、删除、替换和交换两个相邻的字符操作的最少次数

  - 如果$ x_{i+1}= y_{j+1}$（两个串的最后一个字母相同），则
    $ed(X[i+1], Y[j+1]) = ed(X[i], Y[j])$

  - 如果$ x_i = y_{j+1}$，并且 $x_{i+1} = y_j$（最后两个字符需要交换位置），则
    $ed(X[i+1], Y[j+1]) = 1+min\{ed(X[i-1], Y[j-1]),ed(X[i], Y[j+1]),ed(X[i+1], Y[j])\}$

  - 其它情况下$ed(X[i+1], Y[j+1]) = 1+min\{ed(X[i], Y[j]),ed(X[i], Y[j+1]),ed(X[i+1], Y[j])\}$
    其中，$ed(X[0],Y[j])=j(0\leq j\leq n)$

    $ed(X[i],Y[0])=i(0\leq i\leq m)$

    $ed(X[-1],Y[j])=ed(X[i],Y[-1])=max(m,n)$（边界约定）

-  $cuted(X[m],Y[n])=mined_{l\leq i\leq u}(X[i],Y[n])$

$l=max(1,n-t),u=min(m,n+t)$

定阈值 t 有两个用途：确定截取 X 的范围；限定编辑距离。

- 例题：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210210114017670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)