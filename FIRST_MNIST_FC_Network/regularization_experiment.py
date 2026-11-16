
import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt
from pathlib import Path

from model import FullyConnectedNetwork


# =========================
# 1. 读取 MNIST 数据集
# =========================

DATA_DIR = Path(__file__).resolve().parent / "MNIST"
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


def load_images(filename):
    with gzip.open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)

    return images.astype(np.float32) / 255.0


def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


X_train_all = load_images(DATA_DIR / "train-images-idx3-ubyte.gz")
y_train_all = load_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")

X_test = load_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
y_test = load_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

# 训练集和验证集划分
X_train = X_train_all[:55000]
y_train = y_train_all[:55000]

X_valid = X_train_all[55000:]
y_valid = y_train_all[55000:]

print("训练集：", X_train.shape)
print("验证集：", X_valid.shape)
print("测试集：", X_test.shape)


# =========================
# 2. 定义 L2 正则化模型
# =========================

class L2Network(FullyConnectedNetwork):

    def __init__(self, lambda_reg=0.0001):
        super().__init__(
            input_size=784,
            hidden_size=128,
            output_size=10
        )

        self.lambda_reg = lambda_reg

    def compute_loss(self, probs, y):

        # 原始交叉熵损失
        ce_loss = super().compute_loss(probs, y)

        # L2 正则化惩罚项
        l2_loss = 0.5 * self.lambda_reg * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
        )

        return ce_loss + l2_loss

    def backward(self, y):

        # 先计算原始反向传播梯度
        super().backward(y)

        # 在权重梯度中加入 L2 正则化项
        self.grads["dW1"] += self.lambda_reg * self.W1
        self.grads["dW2"] += self.lambda_reg * self.W2


# =========================
# 3. 定义交叉熵损失计算函数
# =========================

def cross_entropy_loss(probs, y):

    batch_size = len(y)

    correct_probs = probs[np.arange(batch_size), y]
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)

    return -np.mean(np.log(correct_probs))


# =========================
# 4. 定义训练函数
# =========================

def train_model(use_l2=False, epochs=10, batch_size=128):

    if use_l2:
        name = "L2"
        model = L2Network(lambda_reg=0.0001)
    else:
        name = "Baseline"
        model = FullyConnectedNetwork(
            input_size=784,
            hidden_size=128,
            output_size=10
        )

    print("\n开始训练模型：", name)

    learning_rate = 0.1

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    best_valid_acc = 0.0
    best_params = None

    for epoch in range(epochs):

        # 打乱训练数据
        indices = np.random.permutation(len(X_train))

        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        total_loss = 0.0

        # Mini-batch SGD
        for start in range(0, len(X_epoch), batch_size):

            end = start + batch_size

            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            probs = model.forward(X_batch)

            loss = model.compute_loss(probs, y_batch)

            model.backward(y_batch)

            model.update(learning_rate)

            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)

        # 训练集准确率
        train_acc = model.accuracy(X_train, y_train)

        # 验证集交叉熵损失
        valid_probs = model.forward(X_valid)
        valid_loss = cross_entropy_loss(valid_probs, y_valid)

        # 验证集准确率
        valid_acc = model.accuracy(X_valid, y_valid)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(
            f"{name} Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Valid Loss: {valid_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

        # 保存最佳模型
        if valid_acc > best_valid_acc:

            best_valid_acc = valid_acc

            best_params = {
                "W1": model.W1.copy(),
                "b1": model.b1.copy(),
                "W2": model.W2.copy(),
                "b2": model.b2.copy()
            }

    # 恢复最佳模型
    model.W1 = best_params["W1"]
    model.b1 = best_params["b1"]
    model.W2 = best_params["W2"]
    model.b2 = best_params["b2"]

    # 测试集评估
    test_acc = model.accuracy(X_test, y_test)

    print(name, "最佳验证集准确率：", round(best_valid_acc, 4))
    print(name, "测试集准确率：", round(test_acc, 4))

    return history, best_valid_acc, test_acc


# =========================
# 5. 运行基础模型和 L2 模型
# =========================

baseline_history, baseline_valid, baseline_test = train_model(
    use_l2=False,
    epochs=10,
    batch_size=128
)

l2_history, l2_valid, l2_test = train_model(
    use_l2=True,
    epochs=10,
    batch_size=128
)


# =========================
# 6. 绘制验证集损失对比曲线
# =========================

plt.figure(figsize=(9, 6))

plt.plot(
    baseline_history["valid_loss"],
    marker="o",
    label="Baseline"
)

plt.plot(
    l2_history["valid_loss"],
    marker="s",
    label="L2 Regularization"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Cross-Entropy Loss")
plt.title("Baseline vs L2 - Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "l2_loss_comparison.png")
plt.show()


# =========================
# 7. 绘制验证集准确率对比曲线
# =========================

plt.figure(figsize=(9, 6))

plt.plot(
    baseline_history["valid_acc"],
    marker="o",
    label="Baseline"
)

plt.plot(
    l2_history["valid_acc"],
    marker="s",
    label="L2 Regularization"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Baseline vs L2 - Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "l2_accuracy_comparison.png")
plt.show()


# =========================
# 8. 输出实验结果
# =========================

print("\n========== L2 正则化实验结果 ==========")

print(
    f"基础模型：验证集准确率 = {baseline_valid:.4f}, "
    f"测试集准确率 = {baseline_test:.4f}"
)

print(
    f"L2 模型：验证集准确率 = {l2_valid:.4f}, "
    f"测试集准确率 = {l2_test:.4f}"
)

print("\n实验完成！")
