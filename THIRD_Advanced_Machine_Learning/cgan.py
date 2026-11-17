
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# 1. 实验参数
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 10

LATENT_DIM = 100
NUM_CLASSES = 10
LABEL_EMBED_DIM = 50

LR = 0.0002
BETA1 = 0.5

TRAIN_LIMIT = 5000
IMAGE_SIZE = 32

DEVICE = torch.device("cpu")

RESULT_DIR = "./results/cgan"
CHECKPOINT_DIR = "./checkpoints/cgan"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# 3. 数据集加载
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

print("正在加载 CIFAR-10 数据集...")

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 与第一题保持相同的训练图片数量
# 使用数据集前5000张图片
indices = list(range(TRAIN_LIMIT))
train_dataset = Subset(dataset, indices)

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

print("数据集加载成功！")
print("训练图片数量：", len(train_dataset))
print("每批图片数量：", BATCH_SIZE)
print("每轮训练批次数量：", len(dataloader))


# ============================================================
# 4. 定义条件生成器 Generator
# ============================================================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # 类别嵌入层
        self.label_embedding = nn.Embedding(
            NUM_CLASSES,
            LABEL_EMBED_DIM
        )

        # 将噪声和类别特征映射到初始特征图
        self.fc = nn.Sequential(
            nn.Linear(
                LATENT_DIM + LABEL_EMBED_DIM,
                512 * 4 * 4
            ),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )

        # 逐步生成32×32 RGB图片
        self.main = nn.Sequential(

            # 4×4 -> 8×8
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8×8 -> 16×16
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16×16 -> 32×32
            nn.ConvTranspose2d(
                128, 3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        # 获取类别嵌入
        label_features = self.label_embedding(labels)

        # 拼接随机噪声与类别特征
        x = torch.cat(
            [noise, label_features],
            dim=1
        )

        # 生成初始特征图
        x = self.fc(x)

        x = x.view(-1, 512, 4, 4)

        # 生成图片
        image = self.main(x)

        return image


# ============================================================
# 5. 定义条件判别器 Discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # 将类别标签转换为10维条件特征图
        self.label_embedding = nn.Embedding(
            NUM_CLASSES,
            NUM_CLASSES
        )

        self.main = nn.Sequential(

            # 输入：RGB图片3通道 + 类别条件10通道
            nn.Conv2d(
                3 + NUM_CLASSES,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 16×16 -> 8×8
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8×8 -> 4×4
            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 4×4 -> 2×2
            nn.Conv2d(
                256, 512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 2×2 -> 1×1
            nn.Conv2d(
                512, 1,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, image, labels):

        batch_size = image.size(0)

        # 获取类别嵌入
        label_features = self.label_embedding(labels)

        # 将类别特征扩展为32×32特征图
        label_features = label_features.view(
            batch_size,
            NUM_CLASSES,
            1,
            1
        )

        label_features = label_features.expand(
            batch_size,
            NUM_CLASSES,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        # 图片与类别条件在通道维度拼接
        x = torch.cat(
            [image, label_features],
            dim=1
        )
        # 输出真假判断分数
        output = self.main(x)
        return output.view(-1)


# ============================================================
# 6. 权重初始化
# ============================================================

def weights_init(module):

    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(
            module.weight.data,
            0.0,
            0.02
        )

    elif "BatchNorm" in classname:
        nn.init.normal_(
            module.weight.data,
            1.0,
            0.02
        )
        nn.init.constant_(
            module.bias.data,
            0
        )


# ============================================================
# 7. 创建模型
# ============================================================

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

generator.apply(weights_init)
discriminator.apply(weights_init)

print("\n生成器参数量：",
      sum(p.numel() for p in generator.parameters()))

print("判别器参数量：",
      sum(p.numel() for p in discriminator.parameters()))


# ============================================================
# 8. 定义损失函数和优化器
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)


# ============================================================
# 9. 固定生成条件
# 每一行对应一个类别，每类生成8张图片
# ============================================================

fixed_labels = torch.arange(
    NUM_CLASSES,
    dtype=torch.long
).repeat_interleave(8).to(DEVICE)

fixed_noise = torch.randn(
    NUM_CLASSES * 8,
    LATENT_DIM,
    device=DEVICE
)


# ============================================================
# 10. 生成并保存图片
# ============================================================

def generate_images(epoch):

    generator.eval()

    with torch.no_grad():

        fake_images = generator(
            fixed_noise,
            fixed_labels
        )

    save_path = os.path.join(
        RESULT_DIR,
        f"epoch_{epoch}.png"
    )

    save_image(
        fake_images,
        save_path,
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

    print("条件生成图片已保存：", save_path)

    generator.train()


# ============================================================
# 11. 训练 DCGAN 的条件版本 cGAN
# ============================================================

G_losses = []
D_losses = []

print("\n开始训练 cGAN ...")

for epoch in range(EPOCHS):

    generator.train()
    discriminator.train()

    running_G_loss = 0.0
    running_D_loss = 0.0

    for batch_idx, (real_images, real_labels) in enumerate(dataloader):

        real_images = real_images.to(DEVICE)
        real_labels = real_labels.to(DEVICE)

        batch_size = real_images.size(0)

        real_targets = torch.ones(
            batch_size,
            device=DEVICE
        )

        fake_targets = torch.zeros(
            batch_size,
            device=DEVICE
        )

        # ----------------------------------------------------
        # 训练判别器
        # ----------------------------------------------------

        optimizer_D.zero_grad()

        # 真实图片与真实类别标签
        real_output = discriminator(
            real_images,
            real_labels
        )

        loss_real = criterion(
            real_output,
            real_targets
        )

        # 使用真实标签作为生成条件
        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=DEVICE
        )

        fake_images = generator(
            noise,
            real_labels
        )

        # 假图片与对应类别标签
        fake_output = discriminator(
            fake_images.detach(),
            real_labels
        )

        loss_fake = criterion(
            fake_output,
            fake_targets
        )

        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizer_D.step()

        # ----------------------------------------------------
        # 训练生成器
        # ----------------------------------------------------

        optimizer_G.zero_grad()

        # 重新生成图片
        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=DEVICE
        )

        fake_images = generator(
            noise,
            real_labels
        )

        fake_output = discriminator(
            fake_images,
            real_labels
        )

        # 希望判别器将生成图片判断为真实
        loss_G = criterion(
            fake_output,
            real_targets
        )

        loss_G.backward()
        optimizer_G.step()

        running_G_loss += loss_G.item()
        running_D_loss += loss_D.item()

        if (batch_idx + 1) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"D Loss: {loss_D.item():.4f} "
                f"G Loss: {loss_G.item():.4f}"
            )

    # 计算每轮平均损失
    avg_G_loss = running_G_loss / len(dataloader)
    avg_D_loss = running_D_loss / len(dataloader)

    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] 完成 "
        f"D Loss: {avg_D_loss:.4f} "
        f"G Loss: {avg_G_loss:.4f}\n"
    )

    # 每轮保存一次生成图片
    generate_images(epoch + 1)


# ============================================================
# 12. 保存最终生成图片
# ============================================================

generate_images("final")


# ============================================================
# 13. 绘制损失曲线
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, EPOCHS + 1),
    G_losses,
    label="Generator Loss",
    linewidth=2
)

plt.plot(
    range(1, EPOCHS + 1),
    D_losses,
    label="Discriminator Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("cGAN Training Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(RESULT_DIR, "loss_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("损失曲线已保存！")


# ============================================================
# 14. 保存模型权重
# ============================================================

torch.save(
    generator.state_dict(),
    os.path.join(CHECKPOINT_DIR, "generator.pth")
)

torch.save(
    discriminator.state_dict(),
    os.path.join(CHECKPOINT_DIR, "discriminator.pth")
)


# ============================================================
# 15. 输出实验结果
# ============================================================

print("\n========== cGAN 训练完成 ==========")

print("最终生成图片：", os.path.join(RESULT_DIR, "epoch_final.png"))
print("损失曲线：", os.path.join(RESULT_DIR, "loss_curve.png"))

print("生成器权重：",
      os.path.join(CHECKPOINT_DIR, "generator.pth"))

print("判别器权重：",
      os.path.join(CHECKPOINT_DIR, "discriminator.pth"))

print("\n实验参数：")
print("训练图片数量：", TRAIN_LIMIT)
print("训练轮数：", EPOCHS)
print("Batch Size：", BATCH_SIZE)
print("学习率：", LR)
print("随机噪声维度：", LATENT_DIM)
print("类别数量：", NUM_CLASSES)

print("\n所有文件已保存！")
