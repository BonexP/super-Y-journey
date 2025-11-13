import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Baseline Training Script')
    parser.add_argument('--cfg', type=str, default='/home/user/PROJECT/FSWD/FSW-MERGE/data.yaml',
                        help='数据集配置文件 (.yaml) 路径 (默认: /home/user/PROJECT/FSWD/FSW-MERGE/data.yaml)')

    parser.add_argument('--model', type=str, default='./modified_yolo11s.yaml',
                        help='修改后的模型配置文件 (.yaml) 路径 (默认: ./modified_yolo11s.yaml)')
    # parser.add_argument('--weights', type=str, default='yolo11n.pt',
    #                     help='预训练权重路径，如 yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练 epochs (默认: 300)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (默认: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图片大小 (px) (默认: 640)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam'],
                        help='优化器类型 (SGD 或 Adam) (默认: Adam)')
    # 降低初始学习率
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率(默认:0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减 (weight decay) (默认: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='优化器动量 (momentum) (默认: 0.937)')

    # 数据增强总开关
    parser.add_argument('--augment', action='store_true',
                        help='启用数据增强 (默认: False)')

# 细分的数据增强参数（默认值为启用增强时的推荐值）
    parser.add_argument('--auto-augment', type=str, default='randaugment',
                        help='自动增强策略 (默认: randaugment, 仅在 --augment 启用时生效)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='mosaic 数据增强概率 (默认: 1.0, 仅在 --augment 启用时生效)')
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='mixup 数据增强概率 (默认: 0.2, 仅在 --augment 启用时生效)')
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV 色调增强幅度 (默认: 0.015, 仅在 --augment 启用时生效)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV 饱和度增强幅度 (默认: 0.7, 仅在 --augment 启用时生效)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV 明度增强幅度 (默认: 0.4, 仅在 --augment 启用时生效)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平移增强幅度 (默认: 0.1, 仅在 --augment 启用时生效)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='尺度缩放增强幅度 (默认: 0.5, 仅在 --augment 启用时生效)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='水平翻转概率 (默认: 0.5, 仅在 --augment 启用时生效)')
    parser.add_argument('--erasing', type=float, default=0.4,
                        help='随机擦除概率 (默认: 0.4, 仅在 --augment 启用时生效)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='预热轮数 (默认: 5)')
    parser.add_argument('--close-mosaic', type=int, default=10,
                        help='关闭 mosaic 数据增强的轮数 (默认: 10)')

    parser.add_argument('--project', type=str, default='runs/train',
                        help='实验结果保存目录 (默认: runs/train)')
    parser.add_argument('--name', type=str, default='baseline_yolo11',
                        help='实验名称，会在 project 下生成同名文件夹 (默认: baseline_yolo11)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    # 这里直接使用本地Ultralytics库中自带的配置文件
    # yolo11_baseline = 'ultralytics/cfg/models/11/yolo11.yaml'  # YOLO11 基线模型配置文件路径

    # 使用配置文件初始化模型（不加载预训练权重）
    custom_yaml= 'ultralytics/cfg/models/11/yolo11s_CBAM.yaml'
    model = YOLO(custom_yaml)
    with open(custom_yaml, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    print("YAML 文件正文如下：\n" + yaml_content)

    # (可选） 或者使用预训练权重文件（推荐，包含了模型架构和权重）
    # model = YOLO('./yolo11s.pt')
    # model = YOLO('./yolo11s.pt')

    # (可选) 如果需要修改模型配置文件，可以在这里加载修改后的配置
    # model=YOLO(args.model)  # 使用修改后的模型配置文件

    # 根据 --augment 参数决定增强配置
    if args.augment:
        # 启用增强：使用命令行参数值（或默认的推荐值）
        augment_config = {
            'augment': True,
            'auto_augment': args.auto_augment,
            'mosaic': args.mosaic,
            'mixup': args.mixup,
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'translate': args.translate,
            'scale': args.scale,
            'fliplr': args.fliplr,
            'erasing': args.erasing,
        }
        print("✅ 数据增强已启用，配置如下：")
        for key, value in augment_config.items():
            print(f"  {key}: {value}")
    else:
        # 关闭增强：所有参数设为 0 或 None
        augment_config = {
            'augment': False,
            'auto_augment': None,
            'mosaic': 0.0,
            'mixup': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'fliplr': 0.0,
            'erasing': 0.0,
        }
        print("❌ 数据增强已关闭")

    # 开始训练
    model.train(
        data=args.cfg,
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # 展开增强配置字典
        **augment_config,
        # 这两个参数不受总开关控制
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        # 早停与 checkpoint
        patience=50,
        save_period=10,
        amp=True,
    )
    print(f"Training complete. Results saved to: {save_dir}")
