import torch
# from models.yolo import Model # Adjust import path if necessary
from ultralytics import YOLO
# from carafe import grad_check
if __name__ == "__main__":
    # Load the new model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    custom_yaml= 'ultralytics/cfg/models/11/yolo11s_CARAFE.yaml'

    # 创建模型时就指定设备
    with   torch.cuda.device(0):
        model = YOLO(custom_yaml)
    model = model.to(device)

    # 强制模型在 CUDA 上进行初始化计算
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 设置默认 CUDA 设备

    model.eval()

    # Print model summary to check layers
    print(model)

    # Test forward pass
    try:
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        output = model(dummy_input)
        print("Model built and forward pass successful!")
    except Exception as e:
        print(f"An error occurred: {e}")
