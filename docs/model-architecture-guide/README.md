# YOLO模型架构修改指南

本文档提供了完整的YOLO模型架构修改指南，帮助你理解如何修改卷积层、插入注意力层以及自定义模型组件。

## 📚 文档目录

1. [模型组织结构概览](./01-model-organization.md) - 理解YOLO模型的整体代码组织
2. [核心模块详解](./02-core-modules.md) - 深入了解各个核心模块的作用
3. [修改卷积层指南](./03-modifying-conv-layers.md) - 如何修改具体的卷积层实现
4. [添加注意力层指南](./04-adding-attention.md) - 如何插入和自定义注意力机制
5. [模型配置文件详解](./05-yaml-configuration.md) - 理解和修改YAML配置文件
6. [实战示例](./06-practical-examples.md) - 完整的修改示例

## 🎯 快速开始

### 你需要修改的主要文件位置：

1. **卷积层实现**: `ultralytics/nn/modules/conv.py`
2. **模块块实现**: `ultralytics/nn/modules/block.py`
3. **注意力机制**: `ultralytics/nn/modules/transformer.py`
4. **模型构建逻辑**: `ultralytics/nn/tasks.py`
5. **模型配置**: `ultralytics/cfg/models/v8/yolov8.yaml`

### 典型修改流程：

```
1. 在模块文件中定义新的层/块（如 conv.py 或 block.py）
2. 在 __init__.py 中导出新模块
3. 在 tasks.py 的 parse_model 函数中注册新模块
4. 在 YAML 配置文件中使用新模块
5. 训练和测试模型
```

## 📖 详细文档

请按顺序阅读以下文档以全面理解YOLO模型架构：

- **第一步**：阅读[模型组织结构概览](./01-model-organization.md)了解整体架构
- **第二步**：阅读[核心模块详解](./02-core-modules.md)了解各模块功能
- **第三步**：根据需求选择阅读修改指南（卷积层或注意力层）
- **第四步**：参考[实战示例](./06-practical-examples.md)进行实践

## ⚡ 重要提示

1. 修改模型前建议先理解整体架构
2. 任何新增的模块都需要在多个地方进行注册
3. 修改后务必测试模型能否正常构建和训练
4. 建议使用版本控制跟踪所有修改

## 🔗 相关资源

- 原始YOLO项目: https://github.com/ultralytics/ultralytics
- YOLOv8文档: https://docs.ultralytics.com/models/yolov8/
