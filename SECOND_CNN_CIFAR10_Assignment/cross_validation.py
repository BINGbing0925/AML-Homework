
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CIFAR10CNN


# ==============================
# 1. 基本配置
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATES = [0.001, 0.0005]

NUM_FOLDS = 3
SEED = 42
NUM_WORKERS = 0

DROPOUT = 0.0
NORMALIZATION = "batch"
KERNEL_SIZE = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# 2. 设置随机种子
# ==============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================
# 3. 加载 CIFAR-10 数据集
# ==============================

def load_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    print("\n正在加载 CIFAR-10 官方训练集……")

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )

    print("训练集图片数量：", len(dataset))

    return dataset


# ==============================
# 4. 构建分层三折交叉验证
# ==============================

def create_stratified_folds(dataset):

    set_seed(SEED)

    labels = np.array(dataset.targets)

    class_indices = []

    # 每个类别分别划分为三份
    for class_id in range(10):

        indices = np.where(labels == class_id)[0]

        np.random.shuffle(indices)

        class_indices.append(
            np.array_split(indices, NUM_FOLDS)
        )

    folds = []

    for fold_id in range(NUM_FOLDS):

        val_indices = []

        for class_id in range(10):
            val_indices.extend(
                class_indices[class_id][fold_id].tolist()
            )

        val_indices = np.array(val_indices)

        all_indices = np.arange(len(dataset))

        train_indices = np.setdiff1d(
            all_indices,
            val_indices
        )

        folds.append((train_indices, val_indices))

    return folds


# ==============================
# 5. 模型训练
# ==============================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ==============================
# 6. 模型验证
# ==============================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ==============================
# 7. 单个学习率的交叉验证
# ==============================

def run_cross_validation(dataset, folds, learning_rate):

    print("\n" + "=" * 65)
    print("开始实验：Learning Rate =", learning_rate)
    print("=" * 65)

    fold_accuracies = []

    for fold_id, (train_indices, val_indices) in enumerate(
        folds, start=1
    ):

        print("\n----------------------------------------")
        print("Learning Rate:", learning_rate)
        print("Fold:", fold_id, "/", NUM_FOLDS)
        print("----------------------------------------")

        set_seed(SEED + fold_id)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        model = CIFAR10CNN(
            num_classes=10,
            dropout=DROPOUT,
            norm=NORMALIZATION,
            kernel_size=KERNEL_SIZE
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        best_val_acc = 0.0
        best_epoch = 0
        final_val_acc = 0.0

        for epoch in range(EPOCHS):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion
            )

            final_val_acc = val_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1

            print(
                f"Fold [{fold_id}/{NUM_FOLDS}] "
                f"Epoch [{epoch + 1}/{EPOCHS}]"
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.2f}%"
            )

            print(
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_acc:.2f}%"
            )

        fold_accuracies.append(final_val_acc)

        print(
            f"Fold {fold_id} 最终验证准确率："
            f"{final_val_acc:.2f}%"
        )

        print(
            f"Fold {fold_id} 最佳验证准确率："
            f"{best_val_acc:.2f}%"
        )

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    print("\n学习率", learning_rate, "的交叉验证结果：")
    print("各折最终验证准确率：", fold_accuracies)
    print(f"平均验证准确率：{mean_acc:.2f}%")
    print(f"标准差：{std_acc:.2f}%")

    return {
        "learning_rate": learning_rate,
        "fold_1_acc": fold_accuracies[0],
        "fold_2_acc": fold_accuracies[1],
        "fold_3_acc": fold_accuracies[2],
        "mean_val_acc": mean_acc,
        "std_val_acc": std_acc
    }


# ==============================
# 8. 保存实验结果
# ==============================

def save_results(results):

    csv_path = os.path.join(
        RESULT_DIR,
        "cross_validation_results.csv"
    )

    txt_path = os.path.join(
        RESULT_DIR,
        "cross_validation_results.txt"
    )

    fieldnames = [
        "learning_rate",
        "fold_1_acc",
        "fold_2_acc",
        "fold_3_acc",
        "mean_val_acc",
        "std_val_acc"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    best_result = max(
        results,
        key=lambda x: x["mean_val_acc"]
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("CIFAR-10 三折交叉验证实验结果\n")
        f.write("=" * 50 + "\n\n")

        for result in results:

            f.write(
                f"Learning Rate: "
                f"{result['learning_rate']}\n"
            )

            f.write(
                f"Fold 1: {result['fold_1_acc']:.2f}%\n"
            )

            f.write(
                f"Fold 2: {result['fold_2_acc']:.2f}%\n"
            )

            f.write(
                f"Fold 3: {result['fold_3_acc']:.2f}%\n"
            )

            f.write(
                f"Mean Accuracy: "
                f"{result['mean_val_acc']:.2f}%\n"
            )

            f.write(
                f"Standard Deviation: "
                f"{result['std_val_acc']:.2f}%\n\n"
            )

        f.write(
            "最佳学习率："
            f"{best_result['learning_rate']}\n"
        )

        f.write(
            "最佳平均验证准确率："
            f"{best_result['mean_val_acc']:.2f}%\n"
        )

    print("\n实验结果 CSV 已保存：", csv_path)
    print("实验结果 TXT 已保存：", txt_path)


# ==============================
# 9. 主程序
# ==============================

def main():

    print("=" * 65)
    print("CIFAR-10 三折交叉验证实验")
    print("=" * 65)

    print("运行设备：", DEVICE)
    print("交叉验证折数：", NUM_FOLDS)
    print("训练轮数：", EPOCHS)
    print("学习率：", LEARNING_RATES)
    print("Dropout：", DROPOUT)
    print("Normalization：", NORMALIZATION)

    dataset = load_dataset()

    folds = create_stratified_folds(dataset)

    print("\n三折数据划分完成：")

    for i, (train_indices, val_indices) in enumerate(
        folds, start=1
    ):

        print(
            f"Fold {i}: "
            f"训练集 {len(train_indices)} 张，"
            f"验证集 {len(val_indices)} 张"
        )

    results = []

    for learning_rate in LEARNING_RATES:

        result = run_cross_validation(
            dataset,
            folds,
            learning_rate
        )

        results.append(result)

    print("\n" + "=" * 65)
    print("交叉验证实验结果汇总")
    print("=" * 65)

    print(
        f"{'Learning Rate':<18}"
        f"{'Mean Val Acc':<18}"
        f"{'Std':<15}"
    )

    for result in results:

        print(
            f"{result['learning_rate']:<18}"
            f"{result['mean_val_acc']:.2f}%"
            f"{'':<12}"
            f"{result['std_val_acc']:.2f}%"
        )

    best_result = max(
        results,
        key=lambda x: x["mean_val_acc"]
    )

    print("\n最佳学习率：", best_result["learning_rate"])

    print(
        "最佳平均验证准确率：",
        f"{best_result['mean_val_acc']:.2f}%"
    )

    save_results(results)

    print("\n三折交叉验证实验完成！")


if __name__ == "__main__":
    main()
