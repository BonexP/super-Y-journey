# evaluate_convnext_models.py

import argparse
import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

# 从ultralytics导入必要的模块
from ultralytics import YOLO


# ------------------ 配置区 ------------------
# 请根据您的实际情况修改这些路径
# 实验结果的根目录
RUNS_DIR = './runs/train'
# 您要评估的实验名称模式，使用通配符 *
EXPERIMENT_PATTERN = 'baseline_yolo11_custom_ConvNeXt*'
# 数据集配置文件 .yaml 的路径
DATASET_YAML_PATH = '/home/user/PROJECT/pp/NEU-DET_YOLO_state_qmh/NEU-DET.yaml'
# 最终报告保存的文件名
OUTPUT_CSV_FILE = 'convnext_models_evaluation_summary.csv'


# -------------------------------------------

def get_model_profile(model_path, imgsz=640):
    """
    加载模型并计算其参数量和GFLOPs（优先使用 thop，失败则返回 N/A）。
    当可训练参数为0时，回退使用总参数量。
    """
    try:
        model = YOLO(model_path).model.eval()

        # 统计参数量：同时计算总参数与可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))

        # 若可训练参数为0，回退为总参数
        chosen_params = trainable_params if trainable_params > 0 else total_params
        if trainable_params == 0 and total_params > 0:
            print("  [!] Detected all params frozen (requires_grad=False). Using total params instead.")

        params_m = round(chosen_params / 1e6, 2)

        # 计算GFLOPs（thop优先）
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)

        gflops = 'N/A'
        try:
            from thop import profile as thop_profile
            flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
            gflops = round(flops / 1e9, 2)
        except Exception as e:
            print(f"  [!] GFLOPs profiling unavailable: {e}")

        return params_m, gflops

    except Exception as e:
        print(f"  [!] Error profiling model {model_path}: {e}")
        return 'N/A', 'N/A'


def evaluate_experiment(exp_path: Path, dataset_yaml: str):
    """
    对单个实验目录进行完整的评估。

    Args:
        exp_path (Path): 实验目录的路径对象。
        dataset_yaml (str): 数据集配置文件的路径。

    Returns:
        dict: 包含该模型所有信息的字典，如果关键文件不存在则返回None。
    """
    experiment_name = exp_path.name
    print(f"--- Processing experiment: {experiment_name} ---")

    # 1. 定位关键文件
    model_path = exp_path / 'weights' / 'best.pt'
    args_path = exp_path / 'args.yaml'

    if not model_path.exists():
        print(f"  [!] Warning: 'best.pt' not found in {exp_path}. Skipping.")
        return None
    if not args_path.exists():
        print(f"  [!] Warning: 'args.yaml' not found in {exp_path}. Hyperparameters will be missing.")
        args_data = {}
    else:
        with open(args_path, 'r') as f:
            args_data = yaml.safe_load(f)

    # 2. 提取超参数
    imgsz = args_data.get('imgsz', 640)
    epochs = args_data.get('epochs', 'N/A')
    lr0 = args_data.get('lr0', 'N/A')
    optimizer = args_data.get('optimizer', 'N/A')
    batch_size = args_data.get('batch', 'N/A')

    # 3. 检查模型架构 (参数量, GFLOPs)
    print("  [*] Profiling model architecture...")
    params_m, gflops = get_model_profile(str(model_path), imgsz)
    print(f"  [*] Profiled: Params={params_m}M, GFLOPs={gflops}")

    # 4. 在数据集上进行验证
    print("  [*] Running validation on NEU-DET dataset...")
    try:
        # 加载模型用于验证
        model = YOLO(str(model_path))

        metrics = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            batch=batch_size,
            split='val',
            verbose=False
        )

        # 提取核心指标（使用平均P/R，避免索引问题）
        map50_95 = round(metrics.box.map, 4)   # mAP50-95
        map50 = round(metrics.box.map50, 4)    # mAP50
        precision = round(metrics.box.mp, 4)   # mean Precision
        recall = round(metrics.box.mr, 4)      # mean Recall

        print(f"  [*] Validation complete: mAP50-95={map50_95}, mAP50={map50}")

    except Exception as e:
        print(f"  [!] Error during validation for {experiment_name}: {e}")
        map50_95, map50, precision, recall = 'Error', 'Error', 'Error', 'Error'

    # 5. 汇总结果
    result_dict = {
        'Experiment': experiment_name,
        'Params (M)': params_m,
        'GFLOPs': gflops,
        'mAP50-95': map50_95,
        'mAP50': map50,
        'Precision': precision,
        'Recall': recall,
        'Epochs': epochs,
        'ImgSize': imgsz,
        'BatchSize': batch_size,
        'Optimizer': optimizer,
        'Initial LR': lr0,
    }
    return result_dict


def main():
    """
    主函数，遍历所有实验并生成报告。
    """
    print("Starting evaluation of ConvNeXt-YOLO models...")

    # 检查数据集配置文件是否存在
    if not Path(DATASET_YAML_PATH).exists():
        print(f"[X] FATAL ERROR: Dataset config file not found at '{DATASET_YAML_PATH}'")
        print("Please update the DATASET_YAML_PATH variable in the script.")
        return

    # 查找所有匹配的实验目录
    base_path = Path(RUNS_DIR)
    experiment_paths = sorted(list(base_path.glob(EXPERIMENT_PATTERN)))

    if not experiment_paths:
        print(f"[X] No experiment directories found matching pattern '{EXPERIMENT_PATTERN}' in '{RUNS_DIR}'")
        return

    print(f"Found {len(experiment_paths)} experiments to evaluate.")

    all_results = []
    # 使用tqdm创建进度条
    for exp_path in tqdm(experiment_paths, desc="Evaluating Experiments"):
        result = evaluate_experiment(exp_path, DATASET_YAML_PATH)
        if result:
            all_results.append(result)

    if not all_results:
        print("Evaluation finished, but no valid results were collected.")
        return

    # 7. 创建并展示最终的表格
    print("\n\n" + "=" * 20 + " FINAL EVALUATION SUMMARY " + "=" * 20)
    df = pd.DataFrame(all_results)

    # 优化列的顺序
    column_order = [
        'Experiment', 'Params (M)', 'GFLOPs', 'mAP50-95', 'mAP50',
        'Precision', 'Recall', 'Epochs', 'ImgSize', 'BatchSize', 'Optimizer', 'Initial LR'
    ]
    df = df[column_order]

    # 按mAP50-95降序排序，方便查看最佳模型
    df_sorted = df.sort_values(by='mAP50-95', ascending=False).reset_index(drop=True)

    # 使用 to_string() 打印完整的DataFrame，防止列被截断
    print(df_sorted.to_string())

    # 8. 保存到CSV文件
    try:
        df_sorted.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n[✔] Summary successfully saved to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\n[!] Error saving summary to CSV: {e}")


if __name__ == '__main__':
    main()
