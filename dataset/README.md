NUSWIDE 的切分规则

当前实现中，保留文本模态的完整性，只对图像模态进行划分。

特征顺序

拼接后的 NUSWIDE 特征向量顺序如下：

图像视图特征，固定顺序为：
- CH [64]
- CM55 [225]
- CORR [144]
- EDH [73]
- WT [128]

文本标签特征：
- Tags1k [1000]

因此，完整特征布局为：

- 图像特征：634
- 文本特征：1000
- 总特征维数：1634

客户端分配方式

对于 NUSWIDE，client0 始终持有完整的文本特征块 Tags1k[1000]。
其余客户端对 5 个图像视图特征按如下方式划分。

2 个客户端
- client0 = TEXT [1000]
- client1 = CH + CM55 + CORR + EDH + WT [634]

3 个客户端
- client0 = TEXT [1000]
- client1 = CH + CM55 + EDH [362]
- client2 = CORR + WT [272]

5 个客户端
- client0 = TEXT [1000]
- client1 = CH + CM55 [289]
- client2 = CORR [144]
- client3 = EDH [73]
- client4 = WT [128]

4 个客户端
client0 = TEXT[1000]
client1 = CH + CM55 + EDH [362]
client2 = CORR [144]
client3 = WT [128]

目前，NUSWIDE 仅支持 client_num 取值为 {2, 3,4， 5}。

统一入口

项目内 NUSWIDE 的规范数据集名统一为：

- NUSWIDE

历史别名 NUSWIDET / NUSWIDEI 会在运行入口处自动归一化为 NUSWIDE，不再保留单独的数据切分或模型分支。

本地输出维度

模型使用的本地表示维度也遵循相同设计：

- 2 个客户端：[60, 40]
- 3 个客户端：[60, 24, 16]
- 5 个客户端：[60, 16, 8, 8, 8]

其中，文本客户端保持 60 维本地表示，而每个图像视图贡献 8 维表示。
