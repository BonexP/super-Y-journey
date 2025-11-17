#!/usr/bin/env bash
# run_yolo.sh 用法：./run_yolo.sh <name>

set -euo pipefail

# 显示帮助文档的函数
show_help() {
  cat << EOF
YOLO 模型训练脚本

用法：
    $0 <name>

参数：
    name    训练任务的名称，用于标识此次训练

功能：
    - 使用 nohup 在后台启动 YOLO 模型训练
    - 自动生成日志文件：<name>.log
    - 显示训练进程的 PID

示例：
    $0 yolo_experiment_001
    $0 my_training_task

注意：
    - 确保 train.py 文件存在于当前目录
    - 训练将在后台运行，可以安全关闭终端
    - 日志文件会保存在当前目录下

EOF
}

# 1. 检查参数
if [[ $# -ne 1 ]]; then
  show_help
  exit 1
fi

NAME=$1
LOG_FILE="./${NAME}.log"

# 2. 启动训练
nohup python train.py --name "${NAME}" > "${LOG_FILE}" 2>&1 &

# 3. 打印提示
echo "[$(date '+%F %T')] 已启动训练：name=${NAME}"
echo "日志文件：${LOG_FILE}"
echo "进程 PID：$!"
