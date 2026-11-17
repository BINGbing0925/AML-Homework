
import os

# 解决当前环境中遇到的 OpenMP 运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset


# ============================================================
# 1. 实验参数
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 32
EPOCHS = 10
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5

IMAGE_SIZE = 32
CHANNELS = 3

# 第一次先使用少量数据测试完整流程
# 测试成功后，再将其修改为 10000 或 50000
TRAIN_LIMIT = 5000

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs("./results/dcgan", exist_ok=True)
os.makedirs("./checkpoints/dcgan", exist_ok=True)

print("当前运行设备：", DEVICE)


# ============================================================
# 2. 加载 CIFAR-10 数据集
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 为了先测试 CPU 训练流程，暂时选取部分数据
if TRAIN_LIMIT is not None:
    subset_indices = list(range(min(TRAIN_LIMIT, len(dataset))))
    dataset = Subset(dataset, subset_indices)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

print("本次使用训练图片数量：", len(dataset))
print("每批图片数量：", BATCH_SIZE)
print("每轮训练批次数量：", len(dataloader))


# ============================================================
# 3. 定义生成器 Generator
# ============================================================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(

            # 输入：100 × 1 × 1
            nn.ConvTranspose2d(
                LATENT_DIM, 512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4 × 4 -> 8 × 8
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8 × 8 -> 16 × 16
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16 × 16 -> 32 × 32
            nn.ConvTranspose2d(
                128, CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 4. 定义判别器 Discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            # 32 × 32 -> 16 × 16
            nn.Conv2d(
                CHANNELS, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 × 16 -> 8 × 8
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 × 8 -> 4 × 4
            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 × 4 -> 1 × 1
            nn.Conv2d(
                256, 1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):
        return self.model(x).view(-1)


# ============================================================
# 5. 初始化网络
# ============================================================

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)

netG.apply(weights_init)
netD.apply(weights_init)

print("生成器参数量：", sum(p.numel() for p in netG.parameters()))
print("判别器参数量：", sum(p.numel() for p in netD.parameters()))


# ============================================================
# 6. 定义损失函数和优化器
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizerG = optim.Adam(
    netG.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)

optimizerD = optim.Adam(
    netD.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)


# ============================================================
# 7. 固定噪声，用于观察生成图片的变化
# ============================================================

fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)


# ============================================================
# 8. 保存生成图片
# ============================================================

def save_generated_images(epoch, is_final=False):

    netG.eval()

    with torch.no_grad():
        fake_images = netG(fixed_noise).cpu()

    # 将图片从 [-1, 1] 还原到 [0, 1]
    fake_images = (fake_images + 1) / 2

    grid = make_grid(
        fake_images,
        nrow=8,
        normalize=False
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")

    if is_final:
        filename = "./results/dcgan/final_generated.png"
    else:
        filename = f"./results/dcgan/epoch_{epoch}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    netG.train()

    print("生成图片已保存：", filename)


# ============================================================
# 9. 开始训练 DCGAN
# ============================================================

G_losses = []
D_losses = []

print("\n开始训练 DCGAN ...")

for epoch in range(EPOCHS):

    netG.train()
    netD.train()

    running_loss_G = 0.0
    running_loss_D = 0.0

    for batch_idx, (real_images, _) in enumerate(dataloader):

        real_images = real_images.to(DEVICE)
        current_batch_size = real_images.size(0)

        # ----------------------------------------------------
        # 训练判别器
        # ----------------------------------------------------

        netD.zero_grad()

        real_labels = torch.ones(
            current_batch_size,
            device=DEVICE
        )

        fake_labels = torch.zeros(
            current_batch_size,
            device=DEVICE
        )

        # 判别真实图片
        output_real = netD(real_images)

        loss_real = criterion(
            output_real,
            real_labels
        )

        # 生成假图片
        noise = torch.randn(
            current_batch_size,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        fake_images = netG(noise)

        # 判别生成图片
        output_fake = netD(fake_images.detach())

        loss_fake = criterion(
            output_fake,
            fake_labels
        )

        # 判别器总损失
        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizerD.step()

        # ----------------------------------------------------
        # 训练生成器
        # ----------------------------------------------------

        netG.zero_grad()

        output_fake = netD(fake_images)

        # 生成器希望判别器将生成图片识别为真实图片
        loss_G = criterion(
            output_fake,
            real_labels
        )

        loss_G.backward()
        optimizerG.step()

        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"D Loss: {loss_D.item():.4f} "
                f"G Loss: {loss_G.item():.4f}"
            )

    avg_loss_D = running_loss_D / len(dataloader)
    avg_loss_G = running_loss_G / len(dataloader)

    D_losses.append(avg_loss_D)
    G_losses.append(avg_loss_G)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] 完成 "
        f"D Loss: {avg_loss_D:.4f} "
        f"G Loss: {avg_loss_G:.4f}\n"
    )

    save_generated_images(epoch + 1)

    torch.save(
        netG.state_dict(),
        "./checkpoints/dcgan/generator.pth"
    )

    torch.save(
        netD.state_dict(),
        "./checkpoints/dcgan/discriminator.pth"
    )


# ============================================================
# 10. 保存最终生成图片
# ============================================================

save_generated_images(EPOCHS, is_final=True)


# ============================================================
# 11. 绘制生成器和判别器损失曲线
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    G_losses,
    label="Generator Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    D_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCGAN Training Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "./results/dcgan/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. 输出实验结果
# ============================================================

print("\n========== DCGAN 训练完成 ==========")

print("最终生成图片：results/dcgan/final_generated.png")
print("损失曲线：results/dcgan/loss_curve.png")
print("生成器权重：checkpoints/dcgan/generator.pth")
print("判别器权重：checkpoints/dcgan/discriminator.pth")

print("\n实验参数：")
print("训练图片数量：", len(dataset))
print("训练轮数：", EPOCHS)
print("Batch Size：", BATCH_SIZE)
print("学习率：", LR)

print("\n所有文件已保存！")
