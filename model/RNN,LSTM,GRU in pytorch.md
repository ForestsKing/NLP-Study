# RNNCell
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity=‘tanh’)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223151657801.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- bias：默认为 True，如果为 false 则表示神经元不使用 bias 偏移参数。
- nonlinearity：默认为tanh，可选relu
## 输入

- input：[batch,input_size]
- hidden：[batch，hidden_size]
## 输出


 - $h'$：[batch,hidden_size]
## 参数
- RNNCell.weight_ih: [hidden_size, input_size]
- RNNCell.weight_hh: [hidden_size, hidden_size]
- RNNCell.bias_ih: [hidden_size]
- RNNCell.bias_hh: [hidden_size]

# RNN
torch.nn.RNN(args, kwargs)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223152254710.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- num_layers：循环神经网络的层数，默认值是 1。
- nonlinearity：默认为tanh，可选relu
- bias：默认为 True，如果为 false 则表示神经元不使用 bias 偏移参数。
- batch_first：如果设置为 True，则输入数据的维度中第一个维度就 是 batch 值，默认为 False。默认情况下第一个维度是序列的长度， 第二个维度才是 - - batch，第三个维度是特征数目。
- dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。默认为0。
- bidirectional : If True, becomes a bidirectional RNN. Default: False
## 输入

- input: [seq_len, batch, input_size]
- $h_0$: [(num_layers * num_directions, batch, hidden_size)]

## 输出

- out: [seq_len, batch, num_directions * hidden_size]
- $h_n$ : [num_layers * num_directions, batch, hidden_size]

## 参数

- RNN.weight_ih_l[k]: 第0层[hidden_size, input_size]，之后为[hidden_size, num_directions * hidden_size]
- RNN.weight_hh_l[k]: [hidden_size, hidden_size]
- RNN.bias_ih_l[k]: [hidden_size]
- RNN.bias_hh_l[k]: [hidden_size]

# LSTMCell
torch.nn.LSTMCell(input_size, hidden_size, bias=True)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022315252187.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- bias：默认为 True，如果为 false 则表示神经元不使用 $bias_{in}$ 和 $bias_{hh}$ 偏移参数。

## 输入
input,(h0,c0) ,后两个默认为全0

- input: [batch, input_size]
- $h_{0}$ : [batch, hidden_size]
- $c_{0}$ : [batch, hidden_size]

## 输出

- $h_{1}$: [batch, hidden_size]
- $c_{1}: [batch, hidden_size]

## 参数

- LSTMCell.weight_ih:包括(W_ii|W_if|W_ig|W_io), [4*hidden_size, input_size]
- LSTMCell.weight_hh: 包括((W_hi|W_hf|W_hg|W_ho)), [4*hidden_size, hidden_size]
- LSTMCell.bias_ih: 包括(b_ii|b_if|b_ig|b_io), [4*hidden_size]
- LSTMCell.bias_hh: 包括(b_hi|b_hf|b_hg|b_ho), [4*hidden_size]

# LSTM
torch.nn.LSTM(*args, **kwargs)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223152952994.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- num_layers：循环神经网络的层数，默认值是 1。
- bias：默认为 True，如果为 false 则表示神经元不使用  $bias_{ih}$ 和 $bias_{hh}$ 偏移参数。
- batch_first：如果设置为 True，则输入数据的维度中第一个维度就 是 batch 值，默认为 False。默认情况下第一个维度是序列的长度， 第二个维度才是 - - batch，第三个维度是特征数目。
- dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。默认为0。
- bidirectional : If True, becomes a bidirectional RNN. Default: False

## 输入
input,(h0,c0) ,后两个默认为全0

- input: [seq_len, batch, input_size]
- $h_{0}$: [num_layers* num_directions, batch, hidden_size]
- $c_{0}$: [num_layers* num_directions, batch, hidden_size]

## 输出

- output: [seq_len, batch, num_directions * hidden_size]
- $h_{n}$: [num_layers * num_directions, batch, hidden_size]
- $c_{n}$: [num_layers * num_directions, batch, hidden_size]

## 参数

- LSTM.weight_ih_l[k]: 包括(W_ii|W_if|W_ig|W_io), 第0层[4*hidden_size, input_size]，之后为[4*hidden_size, num_directions * hidden_size]
- LSTM.weight_hh_l[k]: 包括((W_hi|W_hf|W_hg|W_ho)), [4*hidden_size, hidden_size]
- LSTM.bias_ih_l[k]: 包括(b_ii|b_if|b_ig|b_io), [4*hidden_size]
- LSTM.bias_hh_l[k]: 包括(b_hi|b_hf|b_hg|b_ho), [4*hidden_size]

# GRUCell
torch.nn.GRUCell(input_size, hidden_size, bias=True)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223153305964.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- bias：默认为 True，如果为 false 则表示神经元不使用  $bias_{ih}$ 和 $bias_{hh}$ 偏移参数。

## 输入

input: [batch, input_size]
hidden: [batch, hidden_size]
## 输出

- $h'$：[batch,hidden_size]

## 参数
- GRUCell.weight_ih: [3*hidden_size, input_size]
- GRUCell.weight_hh: [3*hidden_size, hidden_size]
- GRUCell.bias_ih: [3*hidden_size]
- GRUCell.bias_hh: [3*hidden_size]

# GRU
torch.nn.GRU(*args,**kwargs)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210223153534515.png#pic_center)
- input_size：输入数据X的特征值的数目。
- hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
- num_layers：循环神经网络的层数，默认值是 1。
- bias：默认为 True，如果为 false 则表示神经元不使用  $bias_{ih}$ 和 $bias_{hh}$ 偏移参数。
- batch_first：如果设置为 True，则输入数据的维度中第一个维度就 是 batch 值，默认为 False。默认情况下第一个维度是序列的长度， 第二个维度才是 - - batch，第三个维度是特征数目。
- dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。默认为0。
- bidirectional : If True, becomes a bidirectional RNN. Default: False

## 输入

- input: [seq_len, batch, input_size]
- $h_{0}$: [num_layers* num_directions, batch, hidden_size]

## 输出
- output: [seq_len, batch, num_directions * hidden_size]

- $h_{n}$: [num_layers * num_directions, batch, hidden_size]

## 参数

- GRU.weight_ih_l[k]: 包括(W_ir|W_iz|W_in), 第0层[3*hidden_size, input_size]，之后为[3*hidden_size, num_directions * hidden_size]
- GRU.weight_hh_l[k]: 包括(W_hr|W_hz|W_hn), [3*hidden_size, hidden_size]
- GRU.bias_ih_l[k]: 包括(b_ir|b_iz|b_in), [3*hidden_size]
- GRU.bias_hh_l[k]: 包括(b_hr|b_hz|b_hn), [3*hidden_size]