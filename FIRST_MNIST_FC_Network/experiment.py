
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
# 2. 定义优化算法
# =========================

class Optimizer:

    def __init__(self, method, learning_rate=0.001):
        self.method = method
        self.lr = learning_rate
        self.state = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def update(self, model):
        self.t += 1

        params = {
            "W1": model.W1,
            "b1": model.b1,
            "W2": model.W2,
            "b2": model.b2
        }

        grads = model.grads

        for name in params:

            param = params[name]
            grad = grads["d" + name]

            if self.method == "SGD":
                param -= self.lr * grad

            elif self.method == "Momentum":

                if name not in self.state:
                    self.state[name] = np.zeros_like(param)

                self.state[name] = (
                    0.9 * self.state[name] + grad
                )

                param -= self.lr * self.state[name]

            elif self.method == "Adam":

                if name not in self.state:
                    self.state[name] = {
                        "m": np.zeros_like(param),
                        "v": np.zeros_like(param)
                    }

                m = self.state[name]["m"]
                v = self.state[name]["v"]

                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                self.state[name]["m"] = m
                self.state[name]["v"] = v

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (
                    np.sqrt(v_hat) + self.epsilon
                )


# =========================
# 3. 定义训练函数
# =========================

def train_model(method, learning_rate, epochs=10, batch_size=128):

    print("\n开始训练算法：", method)

    # 每种算法使用相同的初始参数
    model = FullyConnectedNetwork(
        input_size=784,
        hidden_size=128,
        output_size=10
    )

    optimizer = Optimizer(method, learning_rate)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    best_valid_acc = 0.0
    best_params = None

    for epoch in range(epochs):

        indices = np.random.permutation(len(X_train))

        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        total_loss = 0.0

        for start in range(0, len(X_epoch), batch_size):

            end = start + batch_size

            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            probs = model.forward(X_batch)

            loss = model.compute_loss(probs, y_batch)

            model.backward(y_batch)

            optimizer.update(model)

            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)

        train_acc = model.accuracy(X_train, y_train)

        valid_probs = model.forward(X_valid)
        valid_loss = model.compute_loss(valid_probs, y_valid)
        valid_acc = model.accuracy(X_valid, y_valid)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(
            f"{method} Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Valid Loss: {valid_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

        # 保存验证集表现最好的模型
        if valid_acc > best_valid_acc:

            best_valid_acc = valid_acc

            best_params = {
                "W1": model.W1.copy(),
                "b1": model.b1.copy(),
                "W2": model.W2.copy(),
                "b2": model.b2.copy()
            }

    # 恢复最佳模型参数
    model.W1 = best_params["W1"]
    model.b1 = best_params["b1"]
    model.W2 = best_params["W2"]
    model.b2 = best_params["b2"]

    test_acc = model.accuracy(X_test, y_test)

    print(method, "最佳验证集准确率：", round(best_valid_acc, 4))
    print(method, "测试集准确率：", round(test_acc, 4))

    return history, best_valid_acc, test_acc


# =========================
# 4. 比较三种优化算法
# =========================

experiments = {
    "SGD": 0.1,
    "Momentum": 0.1,
    "Adam": 0.001
}

results = {}

for method, lr in experiments.items():

    history, valid_acc, test_acc = train_model(
        method=method,
        learning_rate=lr,
        epochs=10,
        batch_size=128
    )

    results[method] = {
        "history": history,
        "valid_acc": valid_acc,
        "test_acc": test_acc
    }


# =========================
# 5. 绘制损失对比曲线
# =========================

plt.figure(figsize=(9, 6))

for method in experiments:

    plt.plot(
        results[method]["history"]["valid_loss"],
        marker="o",
        label=method
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Comparison of Optimizers - Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "optimizer_loss_comparison.png")
plt.show()


# =========================
# 6. 绘制准确率对比曲线
# =========================

plt.figure(figsize=(9, 6))

for method in experiments:

    plt.plot(
        results[method]["history"]["valid_acc"],
        marker="o",
        label=method
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Comparison of Optimizers - Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULT_DIR / "optimizer_accuracy_comparison.png")
plt.show()


# =========================
# 7. 输出最终实验结果
# =========================

print("\n========== 优化算法对比结果 ==========")

for method in experiments:

    print(
        f"{method}: "
        f"验证集准确率 = {results[method]['valid_acc']:.4f}, "
        f"测试集准确率 = {results[method]['test_acc']:.4f}"
    )

print("\n实验完成！")
