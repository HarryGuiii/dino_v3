import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import time
import os
import numpy as np

# --- 配置 ---
MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
TEST_IMAGE = "./source/test/recv08R6gTefu1.png" # 请确保这个文件存在
NUM_ITERATIONS = 50 # 模拟50次搜索请求

def benchmark():
    print(f"🚀 正在加载 DINOv3 模型 ({MODEL_NAME})...")
    start_load = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # 使用 bfloat16 以获得最佳性能和稳定性
    model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    
    print(f"✅ 模型加载完成，耗时: {time.time() - start_load:.2f} 秒")

    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 错误: 找不到测试图片 {TEST_IMAGE}")
        return

    img = Image.open(TEST_IMAGE).convert('RGB')

    # --- 1. GPU 预热 (Warm-up) ---
    # 第一次运行通常较慢，因为需要初始化 CUDA 核函数
    print("🔥 正在预热 GPU...")
    with torch.inference_mode():
        for _ in range(5):
            inputs = processor(images=img, return_tensors="pt").to(device).to(torch.bfloat16)
            _ = model(**inputs)
    print("✅ 预热完成")

    # --- 2. 性能测试 (Profiling) ---
    print(f"⏱️ 开始测试单张图片推理速度 (循环 {NUM_ITERATIONS} 次)...")
    latencies = []

    with torch.inference_mode():
        for i in range(NUM_ITERATIONS):
            start_time = time.time()
            
            # 步骤 1: 预处理
            inputs = processor(images=img, return_tensors="pt").to(device).to(torch.bfloat16)
            
            # 步骤 2: 推理
            outputs = model(**inputs)
            embedding = outputs.pooler_output
            
            # 步骤 3: 强制同步 CUDA（确保我们测量的是 GPU 真实执行时间）
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000 # 转换为毫秒
            latencies.append(latency)
            
            if (i+1) % 10 == 0:
                print(f"已完成 {i+1}/{NUM_ITERATIONS}...")

    # --- 3. 统计结果 ---
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    min_latency = np.min(latencies)
    
    print("\n" + "="*30)
    print("📊 DINOv3 单图搜索耗时统计")
    print("="*30)
    print(f"平均耗时 (Average): {avg_latency:.2f} 毫秒 (ms)")
    print(f"95% 分位数 (P95):   {p95_latency:.2f} 毫秒 (ms)")
    print(f"最快耗时 (Min):     {min_latency:.2f} 毫秒 (ms)")
    print(f"每秒处理 (FPS):     {1000/avg_latency:.2f}")
    print("="*30)
    print("注：此耗时包含 [图像预处理 + GPU推理 + CUDA同步]")

if __name__ == "__main__":
    benchmark()
