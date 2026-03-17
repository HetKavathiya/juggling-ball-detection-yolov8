"""
Optimized YOLOv8 training for juggling balls - CPU/GPU AUTO-DETECT
Fixed for any hardware configuration
"""
from ultralytics import YOLO
import torch
import os

def train_juggling_detector(
    data_yaml='data.yaml',
    num_balls=3,
    epochs=150,
    imgsz=640,
    batch_size=16,
    use_gpu=True
):
    """
    Train YOLOv8 optimized for small identical objects (juggling balls)
    Auto-detects GPU/CPU
    
    Args:
        data_yaml: Path to data.yaml
        num_balls: Number of balls in your juggling
        epochs: Training epochs
        imgsz: Image size
        batch_size: Batch size
        use_gpu: Try to use GPU if available
    """
    
    # Check CUDA availability
    if use_gpu and torch.cuda.is_available():
        device = 0  # Use first GPU
        device_type = "GPU"
    else:
        device = 'cpu'
        device_type = "CPU"
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"❌ Error: {data_yaml} not found!")
        print("Make sure your data.yaml is in the correct path.")
        return None, None
    
    # Choose model based on ball count
    if num_balls <= 4:
        model_choice = 'yolov8s.pt'
    else:
        model_choice = 'yolov8m.pt'
    
    print(f"\n{'='*70}")
    print(f"📦 YOLO MODEL TRAINING - JUGGLING BALLS")
    print(f"{'='*70}")
    print(f"📌 Model: {model_choice}")
    print(f"🎪 Balls to detect: {num_balls}")
    print(f"💻 Device: {device_type}")
    print(f"📊 Image size: {imgsz}x{imgsz}")
    print(f"📈 Batch size: {batch_size}")
    print(f"⏱️  Epochs: {epochs}")
    print(f"���� Data: {data_yaml}")
    print(f"{'='*70}\n")
    
    print(f"🔍 System Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"   Using device: {device} ({device_type})\n")
    
    model = YOLO(model_choice)
    
    # Train with VALID hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,              # Auto CPU/GPU
        patience=30,
        save=True,
        verbose=True,
        project='runs/detect',
        name='juggling_ball_detector',
        
        # Augmentation
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.3,
        shear=10,
        perspective=0.001,
        
        # Optimizer
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        close_mosaic=15,
        workers=0,              # Set to 0 for CPU, 4 for GPU
        seed=42,
    )
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"📁 Output: runs/detect/juggling_ball_detector/")
    print(f"📊 Best model: runs/detect/juggling_ball_detector/weights/best.pt")
    print(f"{'='*70}\n")
    
    # Validate
    print("🔍 Validating model...\n")
    metrics = model.val()
    
    print(f"\n{'='*70}")
    print(f"📈 VALIDATION RESULTS")
    print(f"{'='*70}")
    if hasattr(metrics, 'box'):
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"{'='*70}\n")
    
    return model, results


if __name__ == "__main__":
    # Configuration
    DATA_YAML = 'data.yaml'
    NUM_BALLS = 3
    
    # Training settings for CPU (adjust if needed)
    EPOCHS = 100              # Reduced for CPU (use 150 for GPU)
    IMG_SIZE = 512            # Reduced for CPU (use 640 for GPU)
    BATCH_SIZE = 8            # Reduced for CPU (use 16 for GPU)
    
    print("⚠️  Detected CPU-only PyTorch installation")
    print("Training will be SLOWER on CPU, but will work!")
    print("For faster training, consider installing GPU PyTorch:\n")
    print("  GPU (CUDA 12.1): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  GPU (CUDA 11.8): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  (Or visit: https://pytorch.org/get-started/locally/)\n")
    
    # Train
    model, results = train_juggling_detector(
        data_yaml=DATA_YAML,
        num_balls=NUM_BALLS,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        use_gpu=True  # Will auto-detect and use CPU if GPU not available
    )
    
    if model is not None:
        print("✨ Training completed!")
        print("🚀 Next: Run 'python process_juggling_video.py'")