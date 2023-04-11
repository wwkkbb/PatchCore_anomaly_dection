# #
# # 导入必要的库和模块：
# #
# # python
#
# import pytorch_lightning as pl
# import torch
# from torch.utils.data import DataLoader, random_split
#
# # 继承 LightningModule 并实现其中的方法，以定义自己的神经网络模型：
#
#
# class MyModel(pl.LightningModule):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.layer1 = torch.nn.Linear(2 8 *28, 128)
#         self.layer2 = torch.nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = torch.relu(x)
#         x = self.layer2(x)
#         return x
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = torch.nn.functional.cross_entropy(y_hat, y)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
# #
# # 在这个例子中，我们创建了一个名为 MyModel 的类并继承了 LightningModule 类，同时实现了其中的 forward 方法、training_step 方法和 configure_optimizers 方法。其中，forward 方法定义了如何计算模型的前向传播，training_step 方法定义了如何计算损失函数和更新模型参数，configure_optimizers 方法定义了优化器的类型和超参数。
# #
# # 加载数据集并将其划分为训练集和验证集：
#
# dataset = MyDataset()
# train_set, val_set = random_split(dataset, [55000, 5000])
# train_loader = DataLoader(train_set, batch_size=64)
# val_loader = DataLoader(val_set, batch_size=64)
#
# # 在这个例子中，我们创建了一个名为 MyDataset 的类，并使用 PyTorch 的 DataLoader 类将数据集分配为训练集和验证集。
# #
# # 创建一个 Trainer 对象并使用 fit 方法开始训练模型：
#
# model = MyModel()
# trainer = pl.Trainer(max_epochs=10, gpus=1)
# trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
# #
# # 在这个例子中，我们首先创建了一个 MyModel 的对象，然后创建了一个 Trainer 对象并向其传递了许多参数，例如最大 epoch 数、GPU 数量及训练和验证数据加载器等。最后，我们调用 Trainer 对象的 fit 方法并将模型和数据加载器作为参数传递给它，以开始训练模型。
# #
# # 使用 test 或 predict 方法对模型进行测试或预测：
#
# test_set = MyTestset()
# test_loader = DataLoader(test_set, batch_size=64)
# result = trainer.test(model, test_dataloaders=test_loader)
#
# # 在这个例子中，我们创建了一个名为 MyTestset 的类，并使用 PyTorch 的 DataLoader 类将其转换为测试数据加载器。然后，我们调用 Trainer 对象的 test 方法并将模型和测试数据加载器传递给它，以执行测试和计算指标。
# #
# # 以上就是调用 PyTorch Lightning 的基本流程。需要注意的是，PyTorch Lightning 提供了丰富的 API 扩展和个性化配置选项，可以根据具体情况灵活使用。

from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载数据集并进行标准化处理
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 使用 SparseRandomProjection 进行数据降维
srp = SparseRandomProjection(n_components=2, eps=0.9)
X_srp = srp.fit_transform(X_std)

# 输出降维后的数据信息
print("原始数据形状：", X_std.shape)
print("降维后数据形状：", X_srp.shape)