#!/usr/bin/env python3
"""
GPU Verification Script for Qwen2-Audio Setup
Verifies Blackwell GPU compatibility and readiness for audio model inference
"""

import torch
import sys
import subprocess
import platform


def check_cuda_availability():
    """Check CUDA availability and GPU details"""
    print("🔍 Checking CUDA and GPU Setup...")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False

    print(f"✅ CUDA is available: {torch.version.cuda}")
    print(f"🔥 Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(
            i).total_memory / (1024**3)
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # Check if it's a Blackwell GPU
        if "B100" in gpu_name or "B200" in gpu_name or "Blackwell" in gpu_name:
            print(f"🚀 Detected Blackwell GPU: {gpu_name}")
        else:
            print(f"⚡ GPU: {gpu_name}")

    return True


def test_gpu_memory():
    """Test GPU memory allocation"""
    print("\n💾 Testing GPU Memory...")
    try:
        # Test allocating a large tensor
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        print(f"✅ Successfully allocated tensor on {device}")

        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"📊 Memory allocated: {allocated:.2f}GB")
        print(f"📊 Memory cached: {cached:.2f}GB")

        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False


def check_audio_dependencies():
    """Check if audio processing dependencies are available"""
    print("\n🎵 Checking Audio Dependencies...")

    try:
        import librosa
        print(f"✅ librosa: {librosa.__version__}")
    except ImportError:
        print("❌ librosa not installed")
        return False

    try:
        import soundfile
        print(f"✅ soundfile: {soundfile.__version__}")
    except ImportError:
        print("❌ soundfile not installed")
        return False

    try:
        import torchaudio
        print(f"✅ torchaudio: {torchaudio.__version__}")
    except ImportError:
        print("❌ torchaudio not installed")
        return False

    return True


def estimate_qwen_audio_requirements():
    """Estimate memory requirements for Qwen2-Audio"""
    print("\n🤖 Qwen2-Audio Model Requirements:")
    print("📋 Qwen2-Audio-7B-Instruct:")
    print("   - Model size: ~14GB (bfloat16)")
    print("   - Inference memory: ~16-20GB")
    print("   - Recommended VRAM: 24GB+")
    print("   - Optimal VRAM: 40GB+ (for batching)")

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        print(f"\n📊 Your GPU Memory: {total_memory:.1f}GB")

        if total_memory >= 40:
            print("🚀 Excellent! Can handle Qwen2-Audio-7B with batching")
        elif total_memory >= 24:
            print("✅ Good! Can handle Qwen2-Audio-7B single inference")
        else:
            print("⚠️  Limited memory. Consider model quantization")


def main():
    """Main verification function"""
    print("🔧 Qwen2-Audio GPU Verification")
    print("=" * 50)

    success = True

    # Check CUDA and GPU
    if not check_cuda_availability():
        success = False

    # Test GPU memory
    if not test_gpu_memory():
        success = False

    # Check audio dependencies
    if not check_audio_dependencies():
        success = False

    # Show requirements
    estimate_qwen_audio_requirements()

    print("\n" + "=" * 50)
    if success:
        print("🎉 System is ready for Qwen2-Audio implementation!")
    else:
        print("❌ Please resolve the issues above before proceeding")

    return success


if __name__ == "__main__":
    main()
