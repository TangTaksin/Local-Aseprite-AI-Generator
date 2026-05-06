#!/usr/bin/env python3
"""
Local AI Generator สำหรับ Aseprite - สคริปต์เริ่มต้นที่ปรับปรุงใหม่
สคริปต์เริ่มต้นระดับมืออาชีพพร้อมระบบจัดการข้อผิดพลาดและประสบการณ์ผู้ใช้ที่ดียิ่งขึ้น
เวอร์ชัน 2.1 - รองรับ RTX 5070 Ti และโครงสร้างโฟลเดอร์ใหม่สำหรับโมเดล Local
"""
import sys
import os
import subprocess
import platform
import shutil
import time
from pathlib import Path

def print_banner():
    """แสดงแบนเนอร์เริ่มต้นพร้อมข้อมูลเวอร์ชัน"""
    print("\n" + "="*60)
    print("🎮 LOCAL AI GENERATOR FOR ASEPRITE v2.1")
    print("🎨 การสร้าง Pixel Art ด้วย AI ระดับมืออาชีพ (Optimized for RTX 50 Series)")
    print("="*60)

def print_section(title):
    """พิมพ์หัวข้อส่วนต่างๆ ในรูปแบบที่กำหนด"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def check_python_version():
    """ตรวจสอบว่าเวอร์ชันของ Python ตรงตามความต้องการหรือไม่"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ต้องการ Python 3.8 ขึ้นไป")
        print(f"   เวอร์ชันปัจจุบัน: {version.major}.{version.minor}.{version.micro}")
        print("   กรุณาอัปเกรด Python จาก https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - ปกติ")
    return True

def install_dependencies():
    """ติดตั้งไลบรารีที่จำเป็นพร้อมแสดงสถานะการทำงาน"""
    print_section("กำลังติดตั้งไลบรารีที่จำเป็น (Dependencies)")
    
    # ตรวจสอบว่ามีการติดตั้งไลบรารีไว้แล้วหรือไม่
    print("🔍 กำลังตรวจสอบการติดตั้งไลบรารีที่มีอยู่...")
    try:
        import torch
        import diffusers
        import flask
        import peft
        import cv2
        import accelerate
        import safetensors
        print("✅ ติดตั้งไลบรารีทั้งหมดเรียบร้อยแล้ว!")
        return True
    except ImportError as e:
        print(f"📦 พบไลบรารีบางส่วนขาดหายไป: {e}")
        print("🔄 กำลังติดตั้งแพ็คเกจที่จำเป็น...")
    
    # อัปเกรด pip ก่อนเป็นอันดับแรก
    print("📦 กำลังอัปเกรด pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ อัปเกรด pip สำเร็จ")
    except subprocess.CalledProcessError:
        print("⚠️  คำเตือน: ไม่สามารถอัปเกรด pip ได้ กำลังดำเนินการต่อ...")
    
    # ติดตั้ง PyTorch ก่อน (เนื่องจากเป็นแพ็คเกจที่ใหญ่ที่สุด)
    print("\n🔥 กำลังติดตั้ง PyTorch (นี่คือไฟล์ดาวน์โหลดที่ใหญ่ที่สุด)...")
    print("📥 อาจใช้เวลา 5-10 นาที ขึ้นอยู่กับความเร็วอินเทอร์เน็ตของคุณ...")
    
    torch_installed = False
    
    # สำหรับ RTX 5070 Ti แนะนำ CUDA 12.4 (cu124) เพื่อความเสถียรสูงสุด
    if platform.system() == "Windows":
        print("🎮 กำลังติดตั้ง PyTorch CUDA 12.4 สำหรับ RTX 50 Series...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124"
            ])
            print("✅ ติดตั้ง PyTorch แบบ CUDA สำเร็จ")
            torch_installed = True
        except subprocess.CalledProcessError:
            print("⚠️  การติดตั้งเวอร์ชัน CUDA 12.4 ล้มเหลว กำลังลองติดตั้งเวอร์ชันมาตรฐาน...")
    
    if not torch_installed:
        print("💻 กำลังติดตั้ง PyTorch เวอร์ชัน CPU/มาตรฐาน...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"
            ])
            print("✅ ติดตั้ง PyTorch สำเร็จ")
            torch_installed = True
        except subprocess.CalledProcessError as e:
            print(f"❌ ติดตั้ง PyTorch ล้มเหลว: {e}")
            return False
    
    # ติดตั้งแพ็คเกจอื่นๆ ที่จำเป็นสำหรับ Pony XL และโมเดล Single File
    print("\n📚 กำลังติดตั้งไลบรารี AI และเว็บเซิร์ฟเวอร์...")
    
    other_requirements = [
        "diffusers>=0.27.0", "transformers>=4.40.0", "accelerate>=0.29.0",
        "flask>=2.3.0", "flask-cors>=4.0.0", "pillow>=9.5.0",
        "numpy>=1.24.0", "safetensors>=0.4.0", "peft>=0.11.0",
        "opencv-python", "scipy", "timm", "einops", "kornia", "scikit-learn",
        "omegaconf", "psutil"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + other_requirements)
        print("✅ ติดตั้งไลบรารีอื่นๆ ทั้งหมดสำเร็จ")
    except subprocess.CalledProcessError as e:
        print(f"❌ ติดตั้งไลบรารีล้มเหลว: {e}")
        return False
    
    return True

def select_startup_model():
    """เลือกโมเดลเริ่มต้นผ่านหน้าจอโต้ตอบ"""
    print_section("การเลือกโมเดลหลัก (Base Model Selection)")
    
    # ดึงรายชื่อโมเดลจากโฟลเดอร์ models เพื่อแสดงในตัวเลือกด้วย
    local_models = []
    if os.path.exists("models"):
        local_models = [f for f in os.listdir("models") if f.endswith(('.safetensors', '.ckpt'))]

    models = [
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "SDXL Base (แนะนำ) - คุณภาพสูง",
            "size": "~7GB"
        },
        {
            "name": "runwayml/stable-diffusion-v1-5",
            "description": "SD 1.5 - เร็วกว่า, ขนาดเล็กกว่า",
            "size": "~4GB"
        }
    ]
    
    print("เลือกโมเดลหลักที่จะโหลดตอนเริ่มต้น:")
    print()
    
    idx = 1
    for model in models:
        print(f"   [{idx}] {model['description']} ({model['name']})")
        idx += 1
    
    # แสดงโมเดล Local ที่เจอ
    for l_model in local_models:
        print(f"   [{idx}] Local: {l_model}")
        idx += 1

    print(f"   [{idx}] ไม่โหลด (โหลดเองภายหลัง)")
    print()
    
    while True:
        try:
            choice_str = input(f"ระบุตัวเลือก [1-{idx}] (ค่าเริ่มต้นคือ 1): ").strip()
            choice = int(choice_str) if choice_str else 1
            
            if 1 <= choice <= len(models):
                selected = models[choice - 1]["name"]
                print(f"✅ เลือก: {selected}")
                return selected
            elif len(models) < choice < idx:
                selected = local_models[choice - len(models) - 1]
                print(f"✅ เลือกโมเดล Local: {selected}")
                return selected
            elif choice == idx:
                print("✅ จะไม่มีการโหลดโมเดลตอนเริ่มต้น")
                return "none"
            else:
                print(f"❌ ตัวเลือกไม่ถูกต้อง กรุณาระบุ 1-{idx}")
        except ValueError:
            print("❌ กรุณากรอกเป็นตัวเลขเท่านั้น")

def configure_offline_mode():
    """ตั้งค่าโหมดออฟไลน์พร้อมคำอธิบาย"""
    print_section("การตั้งค่าเครือข่าย")
    
    print("ตัวเลือกโหมดออฟไลน์:")
    print("  • ออนไลน์: ดาวน์โหลดโมเดลจากอินเทอร์เน็ต (แนะนำสำหรับการรันครั้งแรก)")
    print("  • ออฟไลน์: ใช้เฉพาะโมเดลที่ดาวน์โหลดไว้แล้ว (เริ่มเครื่องได้เร็วขึ้น)")
    print()
    
    response = input("รันในโหมดออฟไลน์หรือไม่? [y/N]: ").lower().strip()
    return response in ['y', 'yes']

def setup_directories():
    """สร้างโครงสร้างโฟลเดอร์ที่จำเป็นสำหรับการทำงาน"""
    print_section("กำลังตั้งค่าโฟลเดอร์")
    
    # เพิ่มโฟลเดอร์ outputs เพื่อเก็บประวัติการเจน
    directories = ["loras", "models", "cache", "outputs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 ตรวจสอบ/สร้าง: {directory}/")
    
    print("✅ โครงสร้างโฟลเดอร์พร้อมใช้งาน")

def check_system_requirements():
    """ตรวจสอบสเปคเครื่องและให้คำแนะนำ"""
    print_section("ตรวจสอบความต้องการของระบบ")
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f}GB")
        disk_free = psutil.disk_usage('.').free / (1024**3)
        print(f"💽 พื้นที่ว่างในดิสก์: {disk_free:.1f}GB")
    except ImportError:
        print("ℹ️  ระบบแนะนำให้ติดตั้ง psutil: pip install psutil")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"🔥 VRAM: {vram_gb:.1f}GB")
            if vram_gb >= 12:
                print("🚀 GPU ของคุณยอดเยี่ยมมาก (RTX 50 Series Detect!)")
        else:
            print("⚠️ ไม่พบ NVIDIA GPU (CUDA) - จะใช้ CPU แทน")
    except:
        print("📦 ยังไม่ได้ติดตั้ง PyTorch")

def main():
    """ฟังก์ชันหลักสำหรับการเริ่มระบบ"""
    print_banner()
    
    if not check_python_version():
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)
    
    # 1. ตั้งค่าโฟลเดอร์ก่อนเพื่อให้โมเดลสแกนเจอ
    setup_directories()
    
    # 2. ตรวจสอบและติดตั้งไลบรารี
    if not install_dependencies():
        print("\n❌ การติดตั้งไลบรารีล้มเหลว!")
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)
    
    # 3. ตรวจสอบสเปคเครื่อง
    check_system_requirements()
    
    # 4. เลือกโมเดลและตั้งค่าโหมด
    chosen_model = select_startup_model()
    offline_mode = configure_offline_mode()
    
    print_section("กำลังเริ่มเซิร์ฟเวอร์")
    
    try:
        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
    except:
        pass
    
    try:
        print("🚀 กำลังนำเข้าโมดูลเซิร์ฟเวอร์...")
        from sd_server import main as run_server
        
        print("\n" + "="*60)
        print("🎉 เซิร์ฟเวอร์พร้อมทำงาน!")
        print(f"   • โมเดลเริ่มต้น: {chosen_model}")
        print(f"   • โหมดออฟไลน์: {offline_mode}")
        print("="*60)
        
        # รันเซิร์ฟเวอร์ด้วยการตั้งค่าที่เลือก
        run_server(default_model_to_load=chosen_model, offline=offline_mode)
        
    except Exception as e:
        import traceback
        print(f"\n❌ เกิดข้อผิดพลาด:")
        traceback.print_exc()
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)

if __name__ == "__main__":
    main()