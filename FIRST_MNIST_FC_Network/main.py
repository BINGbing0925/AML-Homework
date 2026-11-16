from model import FullyConnectedNetwork
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# MNIST 数据集路径
DATA_DIR = Path(__file__).resolve().parent / "MNIST"


# ==========================
# 1. 读取 MNIST 图片
# ==========================
def load_mnist_images(file_path):
    with gzip.open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError("图片文件格式不正确")

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)

    return images


# ==========================
# 2. 读取 MNIST 标签
# ==========================
def load_mnist_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("标签文件格式不正确")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


# ==========================
# 3. 读取数据
# ==========================
X_all = load_mnist_images(
    DATA_DIR / "train-images-idx3-ubyte.gz"
)

y_all = load_mnist_labels(
    DATA_DIR / "train-labels-idx1-ubyte.gz"
)

X_test_raw = load_mnist_images(
    DATA_DIR / "t10k-images-idx3-ubyte.gz"
)

y_test = load_mnist_labels(
    DATA_DIR / "t10k-labels-idx1-ubyte.gz"
)


# ==========================
# 4. 数据预处理
# ==========================

# 将图片像素值归一化到 0~1
X_all = X_all.astype(np.float32) / 255.0
X_test_raw = X_test_raw.astype(np.float32) / 255.0

# 将 28×28 图片转换成 784 维向量
X_all = X_all.reshape(-1, 784)
X_test_raw = X_test_raw.reshape(-1, 784)


# ==========================
# 5. 划分训练集和验证集
# ==========================

np.random.seed(42)

indices = np.random.permutation(len(X_all))

train_indices = indices[:55000]
valid_indices = indices[55000:]

X_train = X_all[train_indices]
y_train = y_all[train_indices]

X_valid = X_all[valid_indices]
y_valid = y_all[valid_indices]

X_test = X_test_raw


# ==========================
# 6. 输出数据集信息
# ==========================

print("训练集图片：", X_train.shape)
print("训练集标签：", y_train.shape)

print("验证集图片：", X_valid.shape)
print("验证集标签：", y_valid.shape)

print("测试集图片：", X_test.shape)
print("测试集标签：", y_test.shape)


# ==========================
# 7. 显示部分 MNIST 图片
# ==========================

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================
# 正式训练全连接神经网络
# ==========================

# 创建模型
model = FullyConnectedNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10
)

# 训练参数
epochs = 10
batch_size = 128
learning_rate = 0.1

# 保存训练过程中的指标
train_losses = []
valid_losses = []

train_accuracies = []
valid_accuracies = []

best_valid_acc = 0.0

# 开始训练
for epoch in range(epochs):

    # 打乱训练数据
    indices = np.random.permutation(len(X_train))

    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0.0

    # 按照 batch_size 划分小批量
    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # 前向传播
        probs = model.forward(X_batch)

        # 计算损失
        loss = model.compute_loss(probs, y_batch)

        # 反向传播
        model.backward(y_batch)

        # 更新参数
        model.update(learning_rate)

        epoch_loss += loss * len(X_batch)

    # 计算平均训练损失
    epoch_loss /= len(X_train)

    # 计算训练集准确率
    train_acc = model.accuracy(X_train, y_train)

    # 计算验证集损失
    valid_probs = model.forward(X_valid)
    valid_loss = model.compute_loss(valid_probs, y_valid)

    # 计算验证集准确率
    valid_acc = model.accuracy(X_valid, y_valid)

    # 保存结果
    train_losses.append(epoch_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    # 保存验证集准确率最高的模型参数
    if valid_acc > best_valid_acc:

        best_valid_acc = valid_acc

        best_params = {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy()
        }

    # 输出当前 epoch 的结果
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Valid Loss: {valid_loss:.4f} "
        f"Valid Acc: {valid_acc:.4f}"
    )


# ==========================
# 恢复验证集准确率最高的模型
# ==========================

model.W1 = best_params["W1"]
model.b1 = best_params["b1"]
model.W2 = best_params["W2"]
model.b2 = best_params["b2"]


# ==========================
# 在官方测试集上进行最终评估
# ==========================

test_acc = model.accuracy(X_test, y_test)

print("\n========== 最终测试结果 ==========")
print("最佳验证集准确率：", best_valid_acc)
print("测试集准确率：", test_acc)


# ==========================
# 绘制损失变化曲线
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid()

plt.savefig("results/loss_curve.png", dpi=300)
plt.show()


# ==========================
# 绘制准确率变化曲线
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(valid_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid()

plt.savefig("results/accuracy_curve.png", dpi=300)
plt.show()
