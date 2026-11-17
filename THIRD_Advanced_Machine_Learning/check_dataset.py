import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# 1. 检查运行设备
device = torch.device("cpu")
print("当前运行设备：", device)
print("PyTorch版本：", torch.__version__)
print("Torchvision版本：", torchvision.__version__)


# 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])


# 3. 下载CIFAR-10训练集
print("正在加载CIFAR-10数据集，请稍等...")

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("数据集加载成功！")
print("训练集数量：", len(dataset))


# 4. 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


# 5. 读取一批图片
images, labels = next(iter(dataloader))

print("图片张量形状：", images.shape)
print("标签张量形状：", labels.shape)


# 6. 显示图片
grid = make_grid(images, nrow=4)

plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.title("CIFAR-10 Sample Images")
plt.axis("off")

plt.savefig(
    "cifar10_samples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("数据集测试完成！")
