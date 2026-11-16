
import os

# 处理 Windows 环境下 OpenMP 重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CIFAR10CNN
from train import train_one_epoch, evaluate


# ============================================================
# 1. 实验参数
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT = 0.0
NORMALIZATION = "batch"
KERNEL_SIZE = 3
SEED = 42
NUM_WORKERS = 0

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 创建数据加载器
# ============================================================

def get_augmentation_loaders(
        use_augmentation,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED
):
    """
    加载 CIFAR-10 数据集。

    use_augmentation=False:
        不使用数据增强。

    use_augmentation=True:
        使用随机裁剪和随机水平翻转。

    两组实验使用完全相同的训练集、验证集划分。
    """

    # CIFAR-10 标准均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 普通预处理
    normal_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 数据增强预处理
    augmentation_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 训练集使用增强或普通预处理
    if use_augmentation:
        train_transform = augmentation_transform
    else:
        train_transform = normal_transform

    # 分别创建两个数据集对象。
    # 它们读取同一批官方训练图片，但使用不同的 transform。
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    eval_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=normal_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=normal_transform
    )

    # 固定随机种子，确保两组实验使用相同的数据划分
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    # 训练集可以使用数据增强
    train_subset = Subset(
        train_dataset,
        train_indices
    )

    # 验证集始终使用普通预处理
    val_subset = Subset(
        eval_train_dataset,
        val_indices
    )

    # 使用独立的随机数生成器控制训练数据打乱顺序
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("\n========== 数据集加载完成 ==========")
    print("数据增强：", "开启" if use_augmentation else "关闭")
    print("训练集数量：", len(train_subset))
    print("验证集数量：", len(val_subset))
    print("测试集数量：", len(test_dataset))

    return train_loader, val_loader, test_loader


# ============================================================
# 4. 单组实验
# ============================================================

def run_experiment(
        experiment_name,
        use_augmentation,
        device
):
    print("\n" + "=" * 60)
    print("开始实验：", experiment_name)
    print("=" * 60)

    # 每组实验使用相同的随机种子
    set_seed(SEED)

    # 加载数据
    train_loader, val_loader, test_loader = (
        get_augmentation_loaders(
            use_augmentation=use_augmentation,
            batch_size=BATCH_SIZE,
            seed=SEED
        )
    )

    # 创建模型
    model = CIFAR10CNN(
        num_classes=10,
        dropout=DROPOUT,
        norm=NORMALIZATION,
        kernel_size=KERNEL_SIZE
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0

    # 每组实验使用独立的模型文件
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        experiment_name + ".pth"
    )

    # --------------------------------------------------------
    # 模型训练
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
        )

        # 保存验证集准确率最高的模型
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print("已保存当前最佳模型。")

    # --------------------------------------------------------
    # 加载最佳模型
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    # --------------------------------------------------------
    # 测试集评估
    # --------------------------------------------------------

    test_loss, test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n实验完成：", experiment_name)
    print(f"最佳 Epoch：{best_epoch}")
    print(f"最佳验证准确率：{best_val_acc:.2f}%")
    print(f"测试 Loss：{test_loss:.4f}")
    print(f"测试 Accuracy：{test_acc:.2f}%")

    return {
        "name": experiment_name,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }


# ============================================================
# 5. 保存两组实验的训练记录
# ============================================================

def save_histories(results):

    for result in results:

        history = result["history"]

        file_path = os.path.join(
            RESULT_DIR,
            result["name"] + "_history.csv"
        )

        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Val Loss",
                "Val Accuracy"
            ])

            for i in range(EPOCHS):

                writer.writerow([
                    i + 1,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["val_loss"][i],
                    history["val_acc"][i]
                ])

        print("训练记录已保存：", file_path)


# ============================================================
# 6. 保存实验对比结果
# ============================================================

def save_comparison(results):

    csv_path = os.path.join(
        RESULT_DIR,
        "augmentation_comparison.csv"
    )

    txt_path = os.path.join(
        RESULT_DIR,
        "augmentation_comparison.txt"
    )

    # 保存 CSV
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Experiment",
            "Best Epoch",
            "Best Validation Accuracy",
            "Test Loss",
            "Test Accuracy"
        ])

        for result in results:

            writer.writerow([
                result["name"],
                result["best_epoch"],
                result["best_val_acc"],
                result["test_loss"],
                result["test_acc"]
            ])

    # 保存 TXT
    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("CIFAR-10 数据增强实验对比结果\n")
        f.write("=" * 50 + "\n\n")

        for result in results:

            f.write("实验名称：" + result["name"] + "\n")
            f.write(f"最佳 Epoch：{result['best_epoch']}\n")
            f.write(
                f"最佳验证准确率："
                f"{result['best_val_acc']:.2f}%\n"
            )
            f.write(
                f"测试 Loss：{result['test_loss']:.4f}\n"
            )
            f.write(
                f"测试 Accuracy：{result['test_acc']:.2f}%\n\n"
            )

    print("\n实验对比结果已保存。")


# ============================================================
# 7. 绘制实验对比曲线
# ============================================================

def plot_comparison(results):

    epochs = range(1, EPOCHS + 1)

    # 损失曲线
    plt.figure(figsize=(10, 5))

    for result in results:

        plt.plot(
            epochs,
            result["history"]["val_loss"],
            marker="o",
            label=result["name"]
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Data Augmentation - Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "augmentation_loss_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    # 准确率曲线
    plt.figure(figsize=(10, 5))

    for result in results:

        plt.plot(
            epochs,
            result["history"]["val_acc"],
            marker="o",
            label=result["name"]
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Data Augmentation - Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "augmentation_accuracy_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print("实验对比曲线已保存。")


# ============================================================
# 8. 主函数
# ============================================================

def main():

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CIFAR-10 数据增强对比实验")
    print("=" * 60)

    print("Device：", device)
    print("Batch Size：", BATCH_SIZE)
    print("Learning Rate：", LEARNING_RATE)
    print("Epochs：", EPOCHS)
    print("Dropout：", DROPOUT)
    print("Normalization：", NORMALIZATION)
    print("Kernel Size：", KERNEL_SIZE)

    # 实验一：不使用数据增强
    result_without = run_experiment(
        experiment_name="augmentation_without",
        use_augmentation=False,
        device=device
    )

    # 实验二：使用数据增强
    result_with = run_experiment(
        experiment_name="augmentation_with",
        use_augmentation=True,
        device=device
    )

    results = [
        result_without,
        result_with
    ]

    # 保存结果
    save_histories(results)
    save_comparison(results)
    plot_comparison(results)

    # --------------------------------------------------------
    # 最终对比
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("数据增强实验最终对比")
    print("=" * 60)

    print(
        f"不使用数据增强："
        f"Test Accuracy = {result_without['test_acc']:.2f}%"
    )

    print(
        f"使用数据增强："
        f"Test Accuracy = {result_with['test_acc']:.2f}%"
    )

    difference = (
        result_with["test_acc"]
        - result_without["test_acc"]
    )

    print(f"测试准确率变化：{difference:+.2f} 个百分点")

    print("\n所有实验完成！")


if __name__ == "__main__":
    main()
