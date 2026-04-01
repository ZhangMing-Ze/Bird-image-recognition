import time
import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm  # 导入tqdm库

# 日志记录
writer = SummaryWriter("logs")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 设置数据目录
data_dir = r"D:\homework"

# 加载训练数据集
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)

# 拆分训练集和验证集
train_size = int(0.8 * len(train_dataset))  # 80% 用于训练
val_size = len(train_dataset) - train_size  # 20% 用于验证
train_set, val_set = random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 64
train_dataloader = DataLoader(dataset=train_set, pin_memory=True, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=val_set, pin_memory=True, batch_size=batch_size)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()  # 切换到训练模式
    correct = 0
    size = len(dataloader.dataset)
    avg_loss = 0
    # 使用tqdm来显示进度条
    loop = tqdm(dataloader, total=len(dataloader), unit="batch", desc="Training Progress")

    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()  # 累加损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # 更新进度条
        loop.set_postfix(loss=avg_loss / (batch + 1), accuracy=correct / ((batch + 1) * len(X)) * 100)

    train_loss = avg_loss / size  # 平均训练损失
    train_accuracy = correct / size * 100  # 计算训练准确率
    return train_loss, train_accuracy


def validate(dataloader, model, loss_fn, device):
    model.eval()  # 切换到评估模式
    correct = 0
    size = len(dataloader.dataset)
    avg_loss = 0
    with torch.no_grad():  # 不计算梯度
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss.item()  # 累加损失
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss = avg_loss / size  # 平均验证损失
    val_accuracy = correct / size * 100  # 计算验证准确率
    print(f"Validation Accuracy: {val_accuracy:.1f}%, Validation Loss: {val_loss:.4f}")
    return val_loss, val_accuracy


if __name__ == '__main__':
    # 创建用于保存模型的输出目录
    save_root = "output_bird165_2/"
    os.makedirs(save_root, exist_ok=True)  # 如果目录不存在，则创建

    # 加载预训练的ResNet18模型
    finetune_net = resnet18(weights=ResNet18_Weights.DEFAULT)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 165)  # 171个类别
    nn.init.xavier_normal_(finetune_net.fc.weight)

    parms_1x = [value for name, value in finetune_net.named_parameters()
                if name not in ["fc.weight", "fc.bias"]]
    parms_10x = [value for name, value in finetune_net.named_parameters()
                 if name in ["fc.weight", "fc.bias"]]

    finetune_net = finetune_net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam([{
        'params': parms_1x
    }, {
        'params': parms_10x,
        'lr': learning_rate * 10
    }], lr=learning_rate)

    epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        # 训练过程
        train_loss, train_accuracy = train(train_dataloader, finetune_net, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Training Accuracy: {train_accuracy:.1f}%, Training Loss: {train_loss:.4f}")

        # 验证过程
        val_loss, val_accuracy = validate(valid_dataloader, finetune_net, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 记录到TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, t)
        writer.add_scalars("Accuracy", {"train": train_accuracy, "val": val_accuracy}, t)

    # 保存模型
    torch.save(finetune_net.state_dict(), os.path.join(save_root, "finetuned_resnet18_bird165_2.pth"))
    writer.close()
import torch.nn as nn
