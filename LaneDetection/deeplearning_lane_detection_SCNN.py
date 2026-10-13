"""
教学版 SCNN：RGB → 可学习卷积 → 四方向空间消息传递 → 二分类分割。

这是小型、随机初始化的模型，不是原论文 VGG16 完整复现，也不兼容官方权重。
原论文与代码：https://github.com/XingangPan/SCNN
与传统算法的联系：固定梯度核变成可学习卷积，颜色/斜率规则变成像素监督，
SCNN 沿行列传播上下文，最后得到车道像素掩膜，不区分车道实例。

数据目录：data/train/images/*.jpg 和 data/train/masks/*.png，同名主干配对。
mask 必须为单通道：0=背景，1 或 255=车道；不支持 ignore 或实例标签。
另建 data/val，必须按道路/视频分组划分，避免相邻帧泄漏。

python deeplearning_lane_detection_SCNN.py smoke-test
python deeplearning_lane_detection_SCNN.py train --train-dir data/train --val-dir data/val
python deeplearning_lane_detection_SCNN.py predict --checkpoint outputs/scnn_best.pt --input img.jpg
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

if not hasattr(cv2, "imread"):
    raise ImportError("当前环境没有可用的 OpenCV，请在同一 Python 环境安装 opencv-python 和 torch")


class SpatialMessagePassing(nn.Module):
    """输入 [N,C,H,W]。每个方向共享一个卷积，顺序传递而非同时相加。

    下传示例：new_row[i] = row[i] + ReLU(conv(new_row[i-1]))。
    前一行已经吸收更上方的信息，因此依赖能跨越多行。
    使用列表和 cat，避免原地修改张量破坏 autograd。
    """
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        self.transforms = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False),
            nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False),
            nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False),
            nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False),
        ])
        for conv in self.transforms:
            nn.init.normal_(conv.weight, std=.02)

    @staticmethod
    def sweep(x, conv, axis, reverse=False):
        slices = list(torch.split(x, 1, dim=axis))
        if reverse:
            slices.reverse()
        result = [slices[0]]
        for current in slices[1:]:
            result.append(current + F.relu(conv(result[-1])))
        if reverse:
            result.reverse()
        return torch.cat(result, dim=axis)

    def forward(self, x):
        for conv, (axis, reverse) in zip(self.transforms,
                                         [(2, False), (2, True), (3, False), (3, True)]):
            x = self.sweep(x, conv, axis, reverse)
        return x


class LaneSCNN(nn.Module):
    def __init__(self, use_scnn=True):
        super().__init__()
        # 下采样到 1/8，减少逐行逐列消息传递开销。
        layers = []
        channels = 3
        for out_channels in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                           nn.ReLU(inplace=False)])
            channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.spatial = SpatialMessagePassing(64) if use_scnn else nn.Identity()
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        logits = self.head(self.spatial(self.encoder(x)))
        # 输出 [N,1,H,W] 原始 logits。训练损失内部处理 sigmoid。
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def image_tensor(bgr, size):
    """训练和推理必须共用预处理，OpenCV BGR 转 RGB，像素缩放至 [0,1]。"""
    rgb = cv2.cvtColor(cv2.resize(bgr, size), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()/255.


class LaneDataset(Dataset):
    def __init__(self, root, size):
        root = Path(root)
        self.size = size
        self.pairs = []
        for path in sorted((root/"images").glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            mask = root/"masks"/(path.stem+".png")
            if not mask.is_file():
                raise FileNotFoundError(f"缺少标签：{mask}")
            self.pairs.append((path, mask))
        if not self.pairs:
            raise ValueError(f"{root}/images 中没有训练图片")
        if len({p.stem for p, _ in self.pairs}) != len(self.pairs):
            raise ValueError("图片主干名称重复，会共用标签，请重命名")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise OSError(f"读取失败：{image_path} / {mask_path}")
        if mask.ndim != 2 or mask.shape != bgr.shape[:2]:
            raise ValueError(f"标签必须为单通道且与图片同尺寸：{mask_path}")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"只接受二分类标签 0/1 或 0/255：{mask_path}")
        # 标签不能使用双线性插值，否则会产生不存在的类别。
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        target = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return image_tensor(bgr, self.size), target


def segmentation_loss(logits, target, pos_weight):
    # 正样本较少：加权 BCE + Dice；过大的权重也会增加误检。
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2*(probability*target).sum(axes)+1)/(probability.sum(axes)+target.sum(axes)+1)
    return bce + 1-dice.mean()


def run_epoch(model, loader, device, pos_weight, optimizer=None):
    model.train(optimizer is not None)
    total_loss, count, intersection, union = 0., 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, targets, pos_weight)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()*len(images)
            count += len(images)
            predictions, truth = logits.detach().sigmoid() >= .5, targets.bool()
            intersection += (predictions & truth).sum().item()
            union += (predictions | truth).sum().item()
    # 只计算前景 IoU；全空预测且全空标签的集合记为 1。
    return total_loss/count, intersection/union if union else 1.


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size = (args.width, args.height)
    training, validation = LaneDataset(args.train_dir, size), LaneDataset(args.val_dir, size)
    train_paths = {p.resolve() for p, _ in training.pairs}
    if train_paths & {p.resolve() for p, _ in validation.pairs}:
        raise ValueError("训练集与验证集包含相同图片路径")
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    model = LaneSCNN(use_scnn=not args.no_scnn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor(args.pos_weight, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.
    for epoch in range(args.epochs):
        train_loss, train_iou = run_epoch(model, train_loader, device, pos_weight, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, device, pos_weight)
        print(f"epoch {epoch+1}: train loss={train_loss:.4f}, IoU={train_iou:.4f}; "
              f"val loss={val_loss:.4f}, IoU={val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "size": list(size),
                        "use_scnn": not args.no_scnn, "epoch": epoch+1,
                        "val_iou": val_iou}, output)
    print(f"最佳验证 IoU={best_iou:.4f}，权重：{output}")


def predict(args):
    from Traditional_Lane_Detection import save_image
    if not 0 < args.threshold < 1:
        raise ValueError("threshold 必须在 (0,1)")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = LaneSCNN(checkpoint["use_scnn"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(args.input)
    output = Path(args.output)
    mask_path = output.with_name(output.stem+"_mask.png")
    if Path(args.input).resolve() in {output.resolve(), mask_path.resolve()}:
        raise ValueError("输出不能覆盖输入图片")
    with torch.no_grad():
        inputs = image_tensor(frame, tuple(checkpoint["size"]))[None].to(args.device)
        probability = model(inputs).sigmoid()[0, 0].cpu().numpy()
    # 概率先恢复原尺寸，再阈值化。训练没有硬编码传统 ROI。
    probability = cv2.resize(probability, (frame.shape[1], frame.shape[0]))
    mask = probability >= args.threshold
    overlay = frame.copy()
    overlay[mask] = (.55*frame[mask]+.45*np.array([0, 255, 0])).astype(np.uint8)
    save_image(output, overlay)
    save_image(mask_path, mask.astype(np.uint8)*255)
    print(f"叠加图：{output}，二值标签：{mask_path}")


def smoke_test():
    """仅验证实现可训练；随机张量测试不能证明真实车道检测效果。"""
    torch.manual_seed(0)
    model = LaneSCNN()
    image = torch.rand(2, 3, 32, 64)
    target = torch.zeros(2, 1, 32, 64)
    target[:, :, :, 20:23] = 1
    logits = model(image)
    assert logits.shape == target.shape
    loss = segmentation_loss(logits, target, torch.tensor(5.))
    loss.backward()
    for conv in model.spatial.transforms:
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
        assert conv.weight.grad.abs().sum() > 0
    # 确认信息传播跨越多行，而不是只到相邻行。
    signal = torch.zeros(1, 1, 4, 3)
    signal[:, :, 0] = 1
    conv = nn.Conv2d(1, 1, 1, bias=False)
    nn.init.ones_(conv.weight)
    propagated = SpatialMessagePassing.sweep(signal, conv, 2)
    assert torch.all(propagated == 1)
    print(f"前向/四方向梯度/顺序传播检查通过，loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke-test")
    trainer = commands.add_parser("train")
    trainer.add_argument("--train-dir", required=True)
    trainer.add_argument("--val-dir", required=True)
    trainer.add_argument("--output", default="outputs/scnn_best.pt")
    trainer.add_argument("--epochs", type=int, default=20)
    trainer.add_argument("--batch-size", type=int, default=4)
    trainer.add_argument("--width", type=int, default=512)
    trainer.add_argument("--height", type=int, default=288)
    trainer.add_argument("--lr", type=float, default=1e-3)
    trainer.add_argument("--pos-weight", type=float, default=5.)
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--no-scnn", action="store_true", help="训练普通 CNN，用于消融对比")
    predictor = commands.add_parser("predict")
    predictor.add_argument("--checkpoint", required=True)
    predictor.add_argument("--input", default="img.jpg")
    predictor.add_argument("--output", default="outputs/scnn.jpg")
    predictor.add_argument("--threshold", type=float, default=.5)
    for command in (trainer, predictor):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.width, args.height) <= 0 or args.lr <= 0 or args.pos_weight <= 0:
            parser.error("训练参数必须为正数")
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        smoke_test()


if __name__ == "__main__":
    main()
