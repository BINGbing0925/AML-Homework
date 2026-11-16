
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

# 划分训练集和验证集
X_train = X_train_all[:55000]
y_train = y_train_all[:55000]

X_valid = X_train_all[55000:]
y_valid = y_train_all[55000:]

print("训练集：", X_train.shape)
print("验证集：", X_valid.shape)
print("测试集：", X_test.shape)


# =========================
# 2. 定义 Dropout 网络
# =========================

class DropoutNetwork(FullyConnectedNetwork):

    def __init__(self, dropout_rate=0.2):

        super().__init__(
            input_size=784,
            hidden_size=128,
            output_size=10
        )

        self.dropout_rate = dropout_rate

        # 独立随机数生成器，不影响训练数据的随机打乱
        self.dropout_rng = np.random.RandomState(123)

    def forward(self, X, training=False):

        # 第一层：全连接 + ReLU
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)

        dropout_mask = None

        # 仅在训练阶段使用 Dropout
        if training:

            dropout_mask = (
                self.dropout_rng.rand(*A1.shape)
                >= self.dropout_rate
            ).astype(float)

            # Inverted Dropout
            A1 = A1 * dropout_mask / (1.0 - self.dropout_rate)

        # 第二层：全连接 + Softmax
        Z2 = np.dot(A1, self.W2) + self.b2
        probs = self.softmax(Z2)

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "probs": probs,
            "dropout_mask": dropout_mask
        }

        return probs

    def backward(self, y):

        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        probs = self.cache["probs"]
        dropout_mask = self.cache["dropout_mask"]

        batch_size = len(y)

        # Softmax + 交叉熵梯度
        dZ2 = probs.copy()
        dZ2[np.arange(batch_size), y] -= 1
        dZ2 /= batch_size

        # 第二层梯度
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 向前一层传播梯度
        dA1 = np.dot(dZ2, self.W2.T)

        # 反向传播时应用相同的 Dropout 掩码
        if dropout_mask is not None:
            dA1 = dA1 * dropout_mask / (1.0 - self.dropout_rate)

        # ReLU 反向传播
        dZ1 = dA1 * self.relu_backward(Z1)

        # 第一层梯度
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }


# =========================
# 3. 定义交叉熵损失
# =========================

def cross_entropy_loss(probs, y):

    batch_size = len(y)

    correct_probs = probs[np.arange(batch_size), y]
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)

    return -np.mean(np.log(correct_probs))


# =========================
# 4. 定义训练函数
# =========================

def train_model(use_dropout=False, epochs=10, batch_size=128):

    if use_dropout:

        name = "Dropout"

        model = DropoutNetwork(dropout_rate=0.2)

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

        # 打乱训练集
        indices = np.random.permutation(len(X_train))

        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        total_loss = 0.0

        # Mini-batch SGD
        for start in range(0, len(X_epoch), batch_size):

            end = start + batch_size

            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            # 训练阶段开启 Dropout
            if use_dropout:
                probs = model.forward(X_batch, training=True)
            else:
                probs = model.forward(X_batch)

            loss = model.compute_loss(probs, y_batch)

            model.backward(y_batch)

            model.update(learning_rate)

            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)

        # 训练集评估：关闭 Dropout
        train_acc = model.accuracy(X_train, y_train)

        # 验证集评估：关闭 Dropout
        valid_probs = model.forward(X_valid)
        valid_loss = cross_entropy_loss(valid_probs, y_valid)

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

    # 恢复验证集表现最好的参数
    model.W1 = best_params["W1"]
    model.b1 = best_params["b1"]
    model.W2 = best_params["W2"]
    model.b2 = best_params["b2"]

    # 测试时关闭 Dropout
    test_acc = model.accuracy(X_test, y_test)

    print(name, "最佳验证集准确率：", round(best_valid_acc, 4))
    print(name, "测试集准确率：", round(test_acc, 4))

    return history, best_valid_acc, test_acc


# =========================
# 5. 运行基础模型和 Dropout 模型
# =========================

baseline_history, baseline_valid, baseline_test = train_model(
    use_dropout=False,
    epochs=10,
    batch_size=128
)

dropout_history, dropout_valid, dropout_test = train_model(
    use_dropout=True,
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
    dropout_history["valid_loss"],
    marker="s",
    label="Dropout"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Cross-Entropy Loss")
plt.title("Baseline vs Dropout - Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "dropout_loss_comparison.png")
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
    dropout_history["valid_acc"],
    marker="s",
    label="Dropout"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Baseline vs Dropout - Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "dropout_accuracy_comparison.png")
plt.show()


# =========================
# 8. 输出最终实验结果
# =========================

print("\n========== Dropout 实验结果 ==========")

print(
    f"基础模型：验证集准确率 = {baseline_valid:.4f}, "
    f"测试集准确率 = {baseline_test:.4f}"
)

print(
    f"Dropout模型：验证集准确率 = {dropout_valid:.4f}, "
    f"测试集准确率 = {dropout_test:.4f}"
)

print("\n实验完成！")
