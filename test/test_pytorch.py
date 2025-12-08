import torch

def check_gpu_support():
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # 获取当前GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {device_count}")

        # 获取当前设备信息
        current_device = torch.cuda.current_device()
        print(f"Current device index: {current_device}")

        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"GPU name: {gpu_name}")

        # 检查CUDA版本
        print(f"CUDA version: {torch.version.cuda}")

        # 简单张量计算测试
        print("\nRunning GPU test...")
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print("GPU test passed! Simple tensor calculation successful.")
    else:
        print("No GPU support available. PyTorch is using CPU only.")

    # 打印PyTorch版本
    print(f"\nPyTorch version: {torch.__version__}")

if __name__ == "__main__":
    check_gpu_support()