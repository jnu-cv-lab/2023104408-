# -*- coding: utf-8 -*-
"""
深度学习：CNN用于图像分类
支持数据集：MNIST / CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 任务1：环境准备 ==================
print("=" * 60)
print("任务1：环境准备")

# 检查PyTorch版本和GPU支持
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("使用CPU训练")
    device = torch.device('cpu')

# 简单的张量操作测试
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
z = x + y
print(f"张量加法测试: \n{x} + \n{y} = \n{z}")

# ================== 参数配置 ==================
# 选择数据集: 'MNIST' 或 'CIFAR10'
DATASET = 'MNIST'  # 改成 'CIFAR10' 即可切换

if DATASET == 'MNIST':
    input_channels = 1
    img_size = 28
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
else:  # CIFAR-10
    input_channels = 3
    img_size = 32
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

print(f"\n数据集: {DATASET}")
print(f"图像尺寸: {img_size}x{img_size}, 通道数: {input_channels}")
print(f"类别数: {num_classes}")

# ================== 任务2：加载图像数据集 ==================
print("\n" + "=" * 60)
print("任务2：加载图像数据集")

# 数据预处理
if DATASET == 'MNIST':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 将训练集划分为训练集和验证集 (80%训练, 20%验证)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 显示样本图像
def show_samples(dataset, num_samples=8, dataset_name=DATASET):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flat
    for i in range(num_samples):
        img, label = dataset[i]
        # 反归一化显示
        if dataset_name == 'MNIST':
            img_display = img.squeeze().numpy()
            cmap = 'gray'
        else:
            img_display = np.transpose(img.numpy(), (1, 2, 0))
            img_display = img_display * 0.5 + 0.5  # 反归一化
            cmap = None
        axes[i].imshow(img_display, cmap=cmap)
        axes[i].set_title(f'真实类别: {label}')
        axes[i].axis('off')
    plt.suptitle(f"{dataset_name} 样本图像")
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150)
    plt.show()

print("\n显示训练集样本图像:")
show_samples(train_dataset, num_samples=8)

# ================== 任务3：定义CNN模型 ==================
print("\n" + "=" * 60)
print("任务3：定义CNN模型")

class CNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, img_size):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半

            # 第三个卷积块 (为CIFAR-10增加更多特征)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 计算全连接层输入维度
        self._to_linear = None
        self._get_conv_output(img_size, input_channels)

        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_conv_output(self, img_size, input_channels):
        """计算卷积层输出特征图的维度"""
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_size, img_size)
            dummy = self.conv_layers(dummy)
            self._to_linear = dummy.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 创建模型
model = CNNClassifier(input_channels, num_classes, img_size).to(device)
print(f"模型结构:\n{model}")
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ================== 任务4：训练模型 ==================
print("\n" + "=" * 60)
print("任务4：训练模型")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练历史
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print(f"开始训练 {num_epochs} 个epoch...")

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ================== 任务5 & 6：测试模型 ==================
print("\n" + "=" * 60)
print("任务6：测试模型")

model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader)
test_acc = 100.0 * correct / total

print(f"\n最终测试结果:")
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.2f}%")

# 显示测试图像预测结果
def show_test_predictions(test_dataset, test_loader, model, num_samples=8):
    model.eval()
    images_shown = []
    labels_shown = []
    preds_shown = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(len(images)):
                if len(images_shown) < num_samples:
                    images_shown.append(images[i].cpu())
                    labels_shown.append(labels[i].item())
                    preds_shown.append(predicted[i].item())

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flat
    for i in range(num_samples):
        img = images_shown[i]
        if DATASET == 'MNIST':
            img_display = img.squeeze().numpy()
            cmap = 'gray'
        else:
            img_display = np.transpose(img.numpy(), (1, 2, 0))
            img_display = img_display * 0.5 + 0.5
            cmap = None
        axes[i].imshow(img_display, cmap=cmap)
        color = 'green' if labels_shown[i] == preds_shown[i] else 'red'
        axes[i].set_title(f'真实: {labels_shown[i]} | 预测: {preds_shown[i]}', color=color)
        axes[i].axis('off')
    plt.suptitle(f"{DATASET} 测试集预测结果 (绿色正确, 红色错误)")
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150)
    plt.show()

show_test_predictions(test_dataset, test_loader, model, num_samples=8)

# ================== 任务7：绘制训练曲线 ==================
print("\n" + "=" * 60)
print("任务7：绘制训练曲线")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss曲线
ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
ax1.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss曲线')
ax1.legend()
ax1.grid(True)

# Accuracy曲线
ax2.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Training Accuracy')
ax2.plot(range(1, num_epochs + 1), val_accuracies, 'r-', label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy曲线')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("实验完成！")
print(f"最终测试准确率: {test_acc:.2f}%")