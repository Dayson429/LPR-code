import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 定义 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=bias,
        )
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class Xception(nn.Module):
    def __init__(self, input_channel, num_classes=10):
        super(Xception, self).__init__()

        # Entry Flow
        self.entry_flow1 = nn.Sequential(
            nn.Conv2d(
                input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.entry_flow2 = nn.Sequential(
            SeparableConv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SeparableConv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.entry_flow2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.entry_flow3 = nn.Sequential(
            nn.ReLU(True),
            SeparableConv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            SeparableConv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.entry_flow3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        self.entry_flow4 = nn.Sequential(
            nn.ReLU(True),
            SeparableConv2d(256, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
            SeparableConv2d(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.entry_flow4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2)

        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            SeparableConv2d(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
            SeparableConv2d(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
            SeparableConv2d(728, 728, 3, 1),
            nn.BatchNorm2d(728),
        )

        # Exit Flow
        self.exit_flow1 = nn.Sequential(
            nn.ReLU(True),
            SeparableConv2d(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
            SeparableConv2d(728, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.exit_flow1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2)
        self.exit_flow2 = nn.Sequential(
            SeparableConv2d(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),
            SeparableConv2d(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        entry_out1 = self.entry_flow1(x)
        entry_out2 = self.entry_flow2(entry_out1) + self.entry_flow2_residual(
            entry_out1
        )
        entry_out3 = self.entry_flow3(entry_out2) + self.entry_flow3_residual(
            entry_out2
        )
        entry_out = self.entry_flow4(entry_out3) + self.entry_flow4_residual(entry_out3)

        middle_out = self.middle_flow(entry_out) + entry_out
        for i in range(7):
            middle_out = self.middle_flow(middle_out) + middle_out
        exit_out1 = self.exit_flow1(middle_out) + self.exit_flow1_residual(middle_out)
        exit_out2 = self.exit_flow2(exit_out1)
        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)
        output = self.linear(exit_avg_pool_flat)
        return output


# (1) 将[0,1]的PILImage 转换为[-1,1]的Tensor
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((32, 32)),  # 图像大小调整为 (w,h)=(32，32)
        transforms.ToTensor(),  # 将图像转换为张量 Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
# 测试集不需要进行数据增强
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # 图像大小调整为 (w,h)=(32，32)
        transforms.ToTensor(),  # 将图像转换为张量 Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

# (2) 加载 CIFAR10 数据集
batchsize = 128
# 加载 CIFAR10 数据集, 如果 root 路径加载失败, 则自动在线下载
# 加载 CIFAR10 训练数据集, 50000张训练图片
train_set = torchvision.datasets.CIFAR10(
    root="../dataset", train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize)
# 加载 CIFAR10 验证数据集, 10000张验证图片
test_set = torchvision.datasets.CIFAR10(
    root="../dataset", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)
# 创建生成器，用 next 获取一个批次的数据
valid_data_iter = iter(test_loader)  # _SingleProcessDataLoaderIter 对象
valid_images, valid_labels = next(
    valid_data_iter
)  # images: [batch,3,32,32], labels: [batch]
valid_size = valid_labels.size(0)  # 验证数据集大小，batch
print(valid_images.shape, valid_labels.shape)

# 定义类别名称，CIFAR10 数据集的 10个类别
classes = (
    "0",
    "1",
)

# (3) 构造 Xception 网络模型
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 检查是否有可用的 GPU
model = Xception(3, num_classes=2)  # 实例化 Xception 网络模型
model.to(device)  # 将网络分配到指定的device中
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 定义损失函数 CrossEntropy
optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())  # Adam 优化器

# (4) 训练 Xception 模型
epoch_list = []  # 记录训练轮次
loss_list = []  # 记录训练集的损失值
accu_list = []  # 记录验证集的准确率
num_epochs = 100  # 训练轮次
for epoch in range(num_epochs):  # 训练轮次 epoch
    running_loss = 0.0  # 每个轮次的累加损失值清零
    for step, data in enumerate(train_loader, start=0):  # 迭代器加载数据
        optimizer.zero_grad()  # 损失梯度清零

        inputs, labels = data  # inputs: [batch,3,32,32] labels: [batch]
        outputs = model(inputs.to(device))  # 正向传播
        loss = criterion(outputs, labels.to(device))  # 计算损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # 累加训练损失值
        running_loss += loss.item()
        # if step%100==99:  # 每 100 个 step 打印一次训练信息
        #     print("\t epoch {}, step {}: loss = {:.4f}".format(epoch, step, loss.item()))

    # 计算每个轮次的验证集准确率
    with torch.no_grad():  # 验证过程, 不计算损失函数梯度
        outputs_valid = model(
            valid_images.to(device)
        )  # 模型对验证集进行推理, [batch, 10]
    pred_labels = torch.max(outputs_valid, dim=1)[1]  # 预测类别, [batch]
    accuracy = (
        torch.eq(pred_labels, valid_labels.to(device)).sum().item() / valid_size * 100
    )  # 计算准确率
    print(
        "Epoch {}: train loss={:.4f}, accuracy={:.2f}%".format(
            epoch, running_loss, accuracy
        )
    )

    # 记录训练过程的统计数据
    epoch_list.append(epoch)  # 记录迭代次数
    loss_list.append(running_loss)  # 记录训练集的损失函数
    accu_list.append(accuracy)  # 记录验证集的准确率


# 绘制训练集损失函数和验证集准确率
plt.subplot(2, 1, 1)
plt.plot(epoch_list, loss_list, label="train loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="best")

plt.subplot(2, 1, 2)
plt.plot(epoch_list, accu_list, label="accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.pause(0.1)
plt.show()  # 显示图像
# 保存为CSV
import csv

with open("botnet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "accuracy"])
    for i in range(len(epoch_list)):
        writer.writerow([epoch_list[i], loss_list[i], accu_list[i]])


# (5) 保存 Xception 网络模型
save_path = "../models/Xception_Cifar1"
model_cpu = model.cpu()  # 将模型移动到 CPU
model_path = save_path + ".pth"  # 模型文件路径
torch.save(model.state_dict(), model_path)  # 保存模型权值

# (7) 模型检测
correct = 0
total = 0
for data in test_loader:  # 迭代器加载测试数据集
    imgs, labels = data  # torch.Size([batch,3,32,32) torch.Size([batch])
    # print(imgs.shape, labels.shape)
    outputs = model(imgs.to(device))  # 正向传播, 模型推理, [batch, 10]
    labels_pred = torch.max(outputs, dim=1)[1]  # 模型预测的类别 [batch]
    # _, labels_pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += torch.eq(labels_pred, labels.to(device)).sum().item()
accuracy = 100.0 * correct / total
print("Test samples: {}".format(total))
print("Test accuracy={:.2f}%".format(accuracy))


# (8) 提取测试集图片进行模型推理
batch = 8  # 批次大小
data_set = torchvision.datasets.CIFAR10(
    root="../dataset", train=False, download=False, transform=None
)
plt.figure(figsize=(9, 6))
for i in range(batch):
    imgPIL = data_set[i][0]  # 提取 PIL 图片
    label = data_set[i][1]  # 提取 图片标签
    # 预处理/模型推理/后处理
    imgTrans = transform(imgPIL)  # 预处理变换, torch.Size([3,32,32])
    imgBatch = torch.unsqueeze(imgTrans, 0)  # 转为批处理，torch.Size([batch=1,3,32,32])
    outputs = model(imgBatch.to(device))  # 模型推理, 返回 [batch=1, 10]
    indexes = torch.max(outputs, dim=1)[1]  # 注意 [batch=1], device = 'device
    index = indexes[0].item()  # 预测类别，整数
    # 绘制第 i 张图片
    imgNP = np.array(imgPIL)  # PIL -> Numpy
    out_text = "label:{}/model:{}".format(classes[label], classes[index])
    plt.subplot(2, 4, i + 1)
    plt.imshow(imgNP)
    plt.title(out_text)
    plt.axis("off")
plt.tight_layout()
plt.show()

# (9) 读取图像文件进行模型推理
from PIL import Image

filePath = "../images/img_plane_01.jpg"  # 数据文件的地址和文件名
imgPIL = Image.open(filePath)  # PIL 读取图像文件, <class 'PIL.Image.Image'>

# 预处理/模型推理/后处理
imgTrans = transform["test"](imgPIL)  # 预处理变换, torch.Size([3, 32, 32])
imgBatch = torch.unsqueeze(imgTrans, 0)  # 转为批处理，torch.Size([batch=1, 3, 32, 32])
outputs = model(imgBatch.to(device))  # 模型推理, 返回 [batch=1, 10]
indexes = torch.max(outputs, dim=1)[1]  # 注意 [batch=1], device = 'device
percentages = nn.functional.softmax(outputs, dim=1)[0] * 100
index = indexes[0].item()  # 预测类别，整数
percent = percentages[index].item()  # 预测类别的概率，浮点数

# 绘制第 i 张图片
imgNP = np.array(imgPIL)  # PIL -> Numpy
out_text = "Prediction:{}, {}, {:.2f}%".format(index, classes[index], percent)
print(out_text)
plt.imshow(imgNP)
plt.title(out_text)
plt.axis("off")
plt.tight_layout()
plt.show()
