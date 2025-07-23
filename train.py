import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Baseline Training Script')
    parser.add_argument('--cfg', type=str, default='/home/user/pp/NEU-DET_YOLO_state_qmh/NEU-DET.yaml',
                        help='数据集配置文件 (.yaml) 路径')
    # parser.add_argument('--weights', type=str, default='yolo11n.pt',
    #                     help='预训练权重路径，如 yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练 epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图片大小 (px)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam'],
                        help='优化器类型 (SGD 或 Adam)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减 (weight decay)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='优化器动量 (momentum)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='实验结果保存目录')
    parser.add_argument('--name', type=str, default='baseline_yolo11',
                        help='实验名称，会在 project 下生成同名文件夹')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    # 这里直接使用本地Ultralytics库中自带的配置文件
    yolo11_baseline = 'ultralytics/cfg/models/11/yolo11.yaml'  # 原始YOLOv8模型（默认n）

    model = YOLO('./yolo11s.pt')


    # 开始训练
    model.train(
        data=args.cfg,                # 数据集配置文件路径
        imgsz=args.img_size,          # 输入图片尺寸
        epochs=args.epochs,           # 训练轮数
        batch=args.batch_size,        # 批次大小
        lr0=args.lr0,                 # 初始学习率
        weight_decay=args.weight_decay, # 权重衰减
        momentum=args.momentum,       # 优化器动量
        optimizer=args.optimizer,     # 优化器类型
        project=args.project,         # 实验结果保存主目录
        name=args.name,               # 实验名称
        exist_ok=True,                # 允许已存在的目录
        # 数据增强参数
        augment=True,                 # 启用数据增强
        auto_augment='randaugment', # 使用 RandAugment 数据增强
        mosaic=1.0,                   # mosaic 数据增强概率
        mixup=0.2,                    # mixup 数据增强概率
        hsv_h=0.015,                  # HSV 色调增强幅度
        hsv_s=0.7,                    # HSV 饱和度增强幅度
        hsv_v=0.4,                    # HSV 明度增强幅度
        translate=0.1,                # 平移增强幅度
        scale=0.5,                    # 尺度缩放增强幅度
        fliplr=0.5,                   # 水平翻转概率
        erasing=0.4,                   # 随机擦除概率
        warmup_epochs=3,               # 预热轮数
        close_mosaic=10,              # 关闭 mosaic 数据增强的轮数
        # 早停与 checkpoint
        patience=50,                  # 早停轮数
        save_period=10,               # 模型保存周期
        amp=True,                     # 启用自动混合精度
    )
    print(f"Training complete. Results saved to: {save_dir}")