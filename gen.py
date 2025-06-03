import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import os
import argparse
from sit import SiT_models
from meanflow_sampler import meanflow_sampler

def generate_mnist_samples(model_name, ckpt_path, output_dir, num_samples=25, class_label=None, cfg_scale=1.0, resolution=32):
    """
    加载 MNIST 训练的 MeanFlow 模型，一步采样生成图片
    
    参数:
        ckpt_path: str - 模型检查点路径
        output_dir: str - 输出图片保存目录
        num_samples: int - 生成图片数量
        class_label: int or None - 指定生成类别 (0-9), None 表示随机生成
        cfg_scale: float - 分类器指导强度 (1.0 表示无指导)
        resolution: int - 图片分辨率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型 (MNIST 配置)
    block_kwargs = {
        "fused_attn": False,
        "qk_norm": False,
    }
    latent_size = resolution
    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=1,  # MNIST 是单通道
        num_classes=10,  # MNIST 有 10 类
        use_cfg=(cfg_scale > 1.0),  # 根据是否使用 CFG 启用
        **block_kwargs
    ).to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # 设置输出图像布局 (5x5 网格)
    grid_dim = int(num_samples ** 0.5)
    grid_images = []
    
    # 生成样本
    with torch.no_grad():
        # 生成随机噪声
        z = torch.randn(num_samples, 1, resolution, resolution, device=device)
        
        # 创建标签
        if class_label is not None:
            y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
            print(f"Generating class {class_label} samples")
        else:
            y = torch.randint(0, 10, (num_samples,), device=device)
            print("Generating random class samples")
        
        # 一步采样生成图片
        samples = meanflow_sampler(model, z, y=y, cfg_scale=cfg_scale, num_steps=1)
        
        # 从 [-1, 1] 转换为 [0, 1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        
        # 创建网格并保存
        grid = torchvision.utils.make_grid(samples, nrow=grid_dim, padding=2, normalize=False)
        grid_image = T.ToPILImage()(grid.cpu())
        grid_image_path = os.path.join(output_dir, "meanflow_samples.png")
        grid_image.save(grid_image_path)
        print(f"Saved sample grid to {grid_image_path}")
        
        # 单独保存每张图片
        for i in range(num_samples):
            img = samples[i].squeeze()  # 去掉通道维度 (1, 32, 32) -> (32, 32)
            img = T.ToPILImage()(img.cpu())
            img_path = os.path.join(output_dir, f"sample_{i}_class_{y[i].item()}.png")
            img.save(img_path)
    
    return samples

if __name__ == "__main__":
    generate_mnist_samples(
        model_name="SiT-T/2",
        ckpt_path="work_dir/minst/checkpoints/0004680.pt",
        output_dir=".results",
        num_samples=25,
        class_label=None,
        cfg_scale=1.0,
        resolution=32
    )
    