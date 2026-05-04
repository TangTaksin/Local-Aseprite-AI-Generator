#!/usr/bin/env python3
"""
Local AI Generator for Aseprite - Enhanced Startup Script
Professional startup script with improved error handling and user experience.
Version 2.0 - Ready for Publishing
"""
import sys
import os
import subprocess
import platform
import shutil
import time
from pathlib import Path

def print_banner():
    """Display startup banner with version info."""
    print("\n" + "="*60)
    print("🎮 LOCAL AI GENERATOR FOR ASEPRITE v2.0")
    print("🎨 Professional AI-Powered Pixel Art Generation")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python from https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def install_dependencies():
    """Install required dependencies with progress indication."""
    print_section("Installing Dependencies")
    
    # Check if dependencies are already installed
    print("🔍 Checking if dependencies are already installed...")
    try:
        import torch
        import diffusers
        import flask
        import peft
        import cv2
        print("✅ All dependencies already installed!")
        return True
    except ImportError as e:
        print(f"📦 Some dependencies missing: {e}")
        print("🔄 Installing required packages...")
    
    # Upgrade pip first
    print("📦 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ Pip upgraded")
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not upgrade pip, continuing...")
    
    # Install PyTorch first (largest package)
    print("\n🔥 Installing PyTorch (this is the largest download)...")
    print("📥 This may take 5-10 minutes depending on your internet connection...")
    print("💡 The process may appear frozen - this is normal, please wait...")
    
    torch_installed = False
    
    # Try CUDA version first if available
    if platform.system() == "Windows":
        print("🎮 Trying PyTorch with CUDA support...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu130"
            ])
            print("✅ PyTorch with CUDA installed successfully")
            torch_installed = True
        except subprocess.CalledProcessError:
            print("⚠️  CUDA version failed, trying CPU version...")
    
    # Fallback to CPU version
    if not torch_installed:
        print("💻 Installing PyTorch CPU version...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
            print("✅ PyTorch CPU version installed successfully")
            torch_installed = True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyTorch: {e}")
            return False
    
    # Install other packages
    print("\n📚 Installing AI and web server libraries...")
    print("📥 This may take another 5-10 minutes...")
    
    other_requirements = [
        "diffusers>=0.27.0", "transformers>=4.40.0", "accelerate>=0.29.0",
        "flask>=2.3.0", "flask-cors>=4.0.0", "pillow>=9.5.0",
        "numpy>=1.24.0", "safetensors>=0.4.0", "peft>=0.11.0",
        "opencv-python", "scipy", "timm", "einops", "kornia"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + other_requirements)
        print("✅ All other dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some packages failed, trying individual installation...")
        
        # Try installing packages individually
        failed_packages = []
        for package in other_requirements:
            try:
                print(f"📦 Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"⚠️  Some packages failed: {failed_packages}")
            print("🔄 Trying alternative installation method...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-cache-dir"
                ] + failed_packages)
                print("✅ Alternative installation successful")
            except subprocess.CalledProcessError:
                print("❌ Alternative installation also failed")
                print("\n🔧 Troubleshooting suggestions:")
                print("   • Check your internet connection")
                print("   • Free up disk space (need 5-10GB)")
                print("   • Try running as Administrator")
                print("   • Temporarily disable antivirus software")
                return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import flask
        print(f"✅ Flask {flask.__version__}")
        
        print("🎉 All dependencies installed and verified!")
        return True
        
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        print("Some packages may not have installed correctly.")
        return False

def select_startup_model():
    """Interactive model selection with improved UI."""
    print_section("Base Model Selection")
    
    models = [
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "SDXL Base (Recommended) - High quality, ~7GB",
            "size": "~7GB"
        },
        {
            "name": "runwayml/stable-diffusion-v1-5",
            "description": "SD 1.5 - Faster, smaller, ~4GB",
            "size": "~4GB"
        }
    ]
    
    print("Choose a base model to load on startup:")
    print()
    
    for i, model in enumerate(models):
        print(f"  [{i+1}] {model['description']}")
        print(f"      Size: {model['size']}")
        print()
    
    print(f"  [{len(models)+1}] None (load manually later)")
    print()
    
    while True:
        try:
            choice_str = input(f"Enter choice [1-{len(models)+1}] (default: 1): ").strip()
            choice = int(choice_str) if choice_str else 1
            
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                print(f"✅ Selected: {selected['name']}")
                return selected["name"]
            elif choice == len(models) + 1:
                print("✅ No model will be loaded on startup")
                return "none"
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(models)+1}")
                
        except ValueError:
            print("❌ Please enter a valid number")

def configure_offline_mode():
    """Configure offline mode with explanation."""
    print_section("Network Configuration")
    
    print("Offline Mode Options:")
    print("  • Online:  Download models from internet (recommended for first run)")
    print("  • Offline: Use only cached models (faster startup, requires previous download)")
    print()
    
    response = input("Run in Offline Mode? [y/N]: ").lower().strip()
    
    if response in ['y', 'yes']:
        print("✅ Offline mode enabled - using cached models only")
        return True
    else:
        print("✅ Online mode enabled - models will download as needed")
        return False

def setup_directories():
    """Create necessary directories with proper structure."""
    print_section("Setting Up Directories")
    
    directories = ["loras", "models", "cache"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified: {directory}/")
    
    print("✅ Directory structure ready")

def check_system_requirements():
    """Check system requirements and provide recommendations."""
    print_section("System Requirements Check")
    
    try:
        import psutil
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f}GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected")
            print("   Consider closing other applications for better performance")
        else:
            print("✅ RAM: Sufficient")
        
        # Disk space check
        disk_free = psutil.disk_usage('.').free / (1024**3)
        print(f"💽 Free disk space: {disk_free:.1f}GB")
        
        if disk_free < 10:
            print("⚠️  Warning: Less than 10GB free space")
            print("   AI models require significant storage space")
        else:
            print("✅ Disk space: Sufficient")
            
    except ImportError:
        print("ℹ️  Install psutil for detailed system info: pip install psutil")
    
    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"🔥 VRAM: {vram_gb:.1f}GB")
            
            if vram_gb < 6:
                print("⚠️  Warning: Less than 6GB VRAM")
                print("   Consider using smaller models or reducing batch size")
            else:
                print("✅ GPU: Excellent for AI generation")
        else:
            print("⚠️  No CUDA GPU detected - using CPU (slower)")
            print("   Consider using a NVIDIA GPU for better performance")
    except ImportError:
        print("📦 PyTorch not yet installed")

def main():
    """Main startup function with comprehensive setup."""
    print_banner()
    
    # Python version check
    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # System requirements check
    check_system_requirements()
    
    # Check and install dependencies
    print("\n🔧 Checking Python dependencies...")
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        print("🔧 You can try manual installation:")
        print("   pip install torch torchvision torchaudio diffusers transformers flask")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Model selection
    chosen_model = select_startup_model()
    
    # Offline mode configuration
    offline_mode = configure_offline_mode()
    
    # Final preparation
    print_section("Starting Server")
    
    # Clean up any Python cache
    try:
        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
        print("🧹 Cleaned Python cache")
    except:
        pass
    
    # Start the server
    try:
        print("🚀 Importing server modules...")
        from sd_server import main as run_server
        
        print("🌟 Server starting with configuration:")
        print(f"   • Model: {chosen_model}")
        print(f"   • Offline: {offline_mode}")
        print(f"   • URL: http://127.0.0.1:5000")
        
        print("\n" + "="*60)
        print("🎉 LOCAL AI GENERATOR SERVER STARTING...")
        print("🎨 Ready for Aseprite plugin connection!")
        print("="*60)
        
        # Start the server
        run_server(default_model_to_load=chosen_model, offline=offline_mode)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        print("Thank you for using Local AI Generator!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Server error occurred:")
        print(f"   {str(e)}")
        print("\n🐛 Full error details:")
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("   • Check if port 5000 is available")
        print("   • Verify all dependencies are installed")
        print("   • Try running as administrator")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()