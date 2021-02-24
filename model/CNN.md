# 0 代码：
- [CNN](https://github.com/ForestsKing/NLP-Study/blob/master/demo/CNN.ipynb)
# 1 基础知识
## 1D convolution for text
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134714255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

## 1D convolution for text with padding
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134732921.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 输⼊⻓度为 的词序列，假设单词维度为 4，即有 4 channels
- 卷积后将会得到 1 channel
## 3 channel 1D convolution with padding = 1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134754514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 多个channel则最终得到多个channel的输出，关注的⽂本潜在特征也不同
## conv1d, padded with max pooling over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134815134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 通过 max pooling over time，获得最⼤的激活值
## conv1d, padded with avepooling over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134839279.png#pic_center)

## In PyTorch

```python
batch_size= 16
word_embed_size= 4
seq_len= 7
input = torch.randn(batch_size, word_embed_size, seq_len)
conv1 = Conv1d(in_channels=word_embed_size, out_channels=3, kernel_size=3)
# can add: padding=1
hidden1 = conv1(input)
hidden2 = torch.max(hidden1, dim=2) # max pool
```


## Other less useful notions: stride = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211134942153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- stride 步⻓，减少计算量
## Less useful: local max pool, stride = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135007794.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)
- 每两⾏做 max pooling，被称为步⻓为2的局部最⼤池化
## conv1d, k-max pooling over time, k= 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135040720.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 记录每⼀个channel的所有时间的 top k的激活值，并且按原有顺序保留（上例中的-0.2 0.3）
## Other somewhat useful notions: dilation = 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211135059135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 扩张卷积
- 上例中，对1 3 5⾏进⾏卷积，通过两个filter得到两个channel的激活值
- 可以在第⼀步的卷积中将卷积核从3改为5，即可实现这样的效果，既保证了矩阵很⼩，⼜保证了⼀次卷积中看到更⼤范围的句⼦

**Summary**
- 在CNN中，⼀次能看⼀个句⼦的多少内容是很重要的概念
- 可以使⽤更⼤的filter、扩张卷积或者增⼤卷积深度（层数）

# 2 应用
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210211140524434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDg1NzY4OA==,size_16,color_FFFFFF,t_70#pic_center)

- 输⼊⻓度为 7 的⼀句话，每个词的维度是 5 ，即输⼊矩阵是
- 使⽤不同的filter_size : (2,3,4)，并且每个size都是⽤两个filter，获得两个channel的feature，即共计6个filter
- 对每个filter的feature进⾏1-max pooling后，拼接得到 6 维的向量，并使⽤softmax后再获得⼆分类结果