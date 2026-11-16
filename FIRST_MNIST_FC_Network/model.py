import numpy as np


class FullyConnectedNetwork:

    def __init__(self, input_size=784, hidden_size=128, output_size=10):

        # 初始化随机数
        np.random.seed(42)

        # 第一层权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # 第二层权重和偏置
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 保存前向传播中的中间结果
        self.cache = {}

    # ==========================
    # 1. ReLU 激活函数
    # ==========================
    def relu(self, x):
        return np.maximum(0, x)

    # ReLU 的导数
    def relu_backward(self, x):
        return (x > 0).astype(float)

    # ==========================
    # 2. Softmax 函数
    # ==========================
    def softmax(self, x):

        # 减去最大值，避免指数计算溢出
        x = x - np.max(x, axis=1, keepdims=True)

        exp_x = np.exp(x)

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # ==========================
    # 3. 前向传播
    # ==========================
    def forward(self, X):

        # 第一层：输入层 -> 隐藏层
        Z1 = np.dot(X, self.W1) + self.b1

        # ReLU 激活
        A1 = self.relu(Z1)

        # 第二层：隐藏层 -> 输出层
        Z2 = np.dot(A1, self.W2) + self.b2

        # Softmax 输出概率
        probs = self.softmax(Z2)

        # 保存中间结果，供反向传播使用
        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "probs": probs
        }

        return probs

    # ==========================
    # 4. 交叉熵损失函数
    # ==========================
    def compute_loss(self, probs, y):

        batch_size = len(y)

        # 取出每张图片真实类别对应的预测概率
        correct_probs = probs[np.arange(batch_size), y]

        # 防止 log(0)
        correct_probs = np.clip(correct_probs, 1e-12, 1.0)

        # 计算平均交叉熵损失
        loss = -np.mean(np.log(correct_probs))

        return loss

    # ==========================
    # 5. 反向传播
    # ==========================
    def backward(self, y):
        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        probs = self.cache["probs"]
        batch_size = len(y)
        # 输出层梯度
        dZ2 = probs.copy()
        # 将真实标签对应的位置减去 1
        dZ2[np.arange(batch_size), y] -= 1
        # 求平均梯度
        dZ2 /= batch_size
        # 第二层权重和偏置梯度
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        # 反向传播到隐藏层
        dA1 = np.dot(dZ2, self.W2.T)
        # 经过 ReLU 导数
        dZ1 = dA1 * self.relu_backward(Z1)
        # 第一层权重和偏置梯度
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        # 保存梯度
        self.grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

    # ==========================
    # 6. 使用 SGD 更新参数
    # ==========================
    def update(self, learning_rate=0.1):

        self.W1 -= learning_rate * self.grads["dW1"]
        self.b1 -= learning_rate * self.grads["db1"]

        self.W2 -= learning_rate * self.grads["dW2"]
        self.b2 -= learning_rate * self.grads["db2"]

    # ==========================
    # 7. 预测类别
    # ==========================
    def predict(self, X):

        probs = self.forward(X)

        return np.argmax(probs, axis=1)

    # ==========================
    # 8. 计算准确率
    # ==========================
    def accuracy(self, X, y):

        predictions = self.predict(X)

        return np.mean(predictions == y)
