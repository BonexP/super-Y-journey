#!/usr/bin/env bash
# run_yolo.sh 用法：./run_yolo.sh <name>

set -euo pipefail

# 1. 检查参数
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <name>"
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

