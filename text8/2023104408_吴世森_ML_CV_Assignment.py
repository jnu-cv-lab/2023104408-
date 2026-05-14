# -*- coding: utf-8 -*-
"""
传统机器学习方法用于手写数字图像分类
数据集：sklearn digits (8x8 手写数字)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 任务1：数据准备 ==================
print("=" * 50)
print("任务1：数据准备")
digits = load_digits()
print(f"数据集形状: {digits.images.shape}")
print(f"图像数量: {len(digits.images)}")
print(f"每张图像大小: {digits.images[0].shape}")
print(f"特征向量维度: {digits.data.shape[1]}")
print(f"类别数量: {len(np.unique(digits.target))}")
print(f"类别标签: {np.unique(digits.target)}")

# 可视化几张图像
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'数字: {digits.target[i]}')
    ax.axis('off')
plt.suptitle("手写数字数据集示例")
plt.tight_layout()
plt.savefig('digits_samples.png', dpi=150)
plt.show()

# ================== 任务2：数据划分 ==================
print("\n" + "=" * 50)
print("任务2：数据划分")
X = digits.data  # 特征矩阵 (1797, 64)
y = digits.target  # 标签 (1797,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} 张图像")
print(f"测试集大小: {X_test.shape[0]} 张图像")
print(f"测试集比例: {len(X_test)/len(X):.1%}")
print("\n说明：")
print("- 训练集：用于训练模型，让模型学习从特征到标签的映射关系")
print("- 测试集：用于评估模型的泛化能力，模型从未见过测试集的数据")

# ================== 任务3：特征表示 ==================
print("\n" + "=" * 50)
print("任务3：特征表示")
print(f"一张 8x8 的图像被展平为 64 维向量")
print(f"示例：第一张图像展平后的前10个像素值: {X[0][:10]}")
print("\n说明：")
print("- 转换方式：将图像的每一行首尾相接，形成一个长向量")
print("- 为什么需要转换：传统机器学习算法通常输入为二维表格数据（样本×特征）")
print("- 原始像素优点：计算简单，保留了像素级别的亮度信息")
print("- 原始像素局限：缺乏空间不变性，无法处理平移、旋转等变换")

# ================== 任务4：模型训练 ==================
print("\n" + "=" * 50)
print("任务4：模型训练与评估")

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "朴素贝叶斯": GaussianNB(),
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', gamma='scale', random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:10} 准确率: {acc:.4f}")

# ================== 任务5：结果比较 ==================
print("\n" + "=" * 50)
print("任务5：结果比较")
print("-" * 40)
print(f"{'模型':<15} {'测试准确率':<15}")
print("-" * 40)
for name, acc in results.items():
    print(f"{name:<15} {acc:.4f}")
print("-" * 40)

# 找出最高和最低准确率
best_model = max(results, key=results.get)
worst_model = min(results, key=results.get)
print(f"\n准确率最高的模型: {best_model} ({results[best_model]:.4f})")
print(f"准确率最低的模型: {worst_model} ({results[worst_model]:.4f})")

# ================== 任务6：错误样本分析 ==================
print("\n" + "=" * 50)
print("任务6：错误样本分析（以随机森林为例）")

best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=digits.target_names, 
            yticklabels=digits.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('随机森林混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 找出错误分类的样本
errors = np.where(y_test != y_pred_best)[0]
print(f"\n测试集中错误分类的样本数量: {len(errors)}")
print(f"准确率: {accuracy_score(y_test, y_pred_best):.4f}")

# 显示部分错误样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, idx in enumerate(errors[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax.set_title(f'真实:{y_test[idx]} -> 预测:{y_pred_best[idx]}')
    ax.axis('off')
plt.suptitle("随机森林的错误分类样本")
plt.tight_layout()
plt.savefig('error_samples.png', dpi=150)
plt.show()

# 分析哪些数字容易混淆
print("\n各类别分类报告：")
print(classification_report(y_test, y_pred_best, target_names=[str(i) for i in range(10)]))

# 找出最容易被混淆的数字对
confusion = confusion_matrix(y_test, y_pred_best)
print("\n最容易混淆的数字对（真实标签 -> 预测标签）：")
for i in range(10):
    for j in range(10):
        if i != j and confusion[i][j] > 0:
            print(f"  数字 {i} 被误判为 {j}: {confusion[i][j]} 次")

print("\n分析：")
print("- 从混淆矩阵可以看出，数字4和9、7和2、3和5等容易被混淆")
print("- 原因：这些数字的手写形状相似，在8x8低分辨率下特征不够明显")
print("- 例如：4和9在书写潦草时，上半部分都可能是封闭或半封闭的圆形")

print("\n" + "=" * 50)
print("实验完成！")