## 重构LSTM
目的：在不直接调用pytorch的nn.LSTM()的情况下，用pytorch的内部操作直接写出一个LSTM。

### 内容：

1. ReLSTM.py

重构的一层LSTM。

2. ReLSTM copy.py

重构的多层LSTM

### 多层LSTM的思路：

首先设计单位，然后用`nn.Sequential()`将多层单位连接成一个整体。这一“多层”是课件ppt图示中的一个纵列，因此，需要保存每一层的隐藏层和记忆层以保持网络工作。这里使用两个数组完成：

```python
tensors:List[torch.Tensor] = list();
cells:List[torch.Tensor] = list();
```

其中，List[torch.Tensor]是从typing引入的标记，并无实际的编译作用。其起的作用是提示编辑器这是个torch.Tensor的数组，方便在调用数组元素的时候自动提示其成员函数。

其次，每次进行操作的时候，要把元素和上一层的隐藏层合到一起（最后一层除外）：

```python
return torch.cat([hidden_state[0],model],dim=1)
```

因此要设计三种神经元，一种放在最开头，`n_class`个->`n_class+n_hidden`个；一种放在中间，`n_class+n_hidden`个->`n_class+n_hidden`个，最后一种放在最后，`n_class+n_hidden`个->`n_class`个。三种神经元分别设定为`TextLSTM_1`，`TextLSTM_2`和`Text_LSTM_3`。三种神经元运算过程大同小异，只是输出和输入的时候需要调整。同时三种神经元输出前还要更新对应的tensors和cells中的元素。

最后注意：本模型运算量较大，建议上GPU。
