#!/usr/bin/env python3
"""
Local AI Generator สำหรับ Aseprite - สคริปต์เริ่มต้นที่ปรับปรุงใหม่
สคริปต์เริ่มต้นระดับมืออาชีพพร้อมระบบจัดการข้อผิดพลาดและประสบการณ์ผู้ใช้ที่ดียิ่งขึ้น
เวอร์ชัน 2.0 - พร้อมสำหรับการเผยแพร่
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
    print("🎮 LOCAL AI GENERATOR FOR ASEPRITE v2.0")
    print("🎨 การสร้าง Pixel Art ด้วย AI ระดับมืออาชีพ")
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
    print("💡 กระบวนการอาจดูเหมือนค้าง - นี่เป็นเรื่องปกติ โปรดรอสักครู่...")
    
    torch_installed = False
    
    # พยายามติดตั้งเวอร์ชัน CUDA หากเป็น Windows
    if platform.system() == "Windows":
        print("🎮 กำลังพยายามติดตั้ง PyTorch พร้อมรองรับ CUDA...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu130"
            ])
            print("✅ ติดตั้ง PyTorch แบบ CUDA สำเร็จ")
            torch_installed = True
        except subprocess.CalledProcessError:
            print("⚠️  การติดตั้งเวอร์ชัน CUDA ล้มเหลว กำลังลองติดตั้งเวอร์ชัน CPU...")
    
    # ติดตั้งเวอร์ชัน CPU หากแบบ CUDA ล้มเหลวหรือไม่ได้อยู่ใน Windows
    if not torch_installed:
        print("💻 กำลังติดตั้ง PyTorch เวอร์ชัน CPU...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
            print("✅ ติดตั้ง PyTorch เวอร์ชัน CPU สำเร็จ")
            torch_installed = True
        except subprocess.CalledProcessError as e:
            print(f"❌ ติดตั้ง PyTorch ล้มเหลว: {e}")
            return False
    
    # ติดตั้งแพ็คเกจอื่นๆ
    print("\n📚 กำลังติดตั้งไลบรารี AI และเว็บเซิร์ฟเวอร์...")
    print("📥 อาจใช้เวลาอีกประมาณ 5-10 นาที...")
    
    other_requirements = [
        "diffusers>=0.27.0", "transformers>=4.40.0", "accelerate>=0.29.0",
        "flask>=2.3.0", "flask-cors>=4.0.0", "pillow>=9.5.0",
        "numpy>=1.24.0", "safetensors>=0.4.0", "peft>=0.11.0",
        "opencv-python", "scipy", "timm", "einops", "kornia","scikit-learn"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + other_requirements)
        print("✅ ติดตั้งไลบรารีอื่นๆ ทั้งหมดสำเร็จ")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  บางแพ็คเกจติดตั้งล้มเหลว กำลังลองติดตั้งทีละตัว...")
        
        # พยายามติดตั้งแพ็คเกจทีละตัว
        failed_packages = []
        for package in other_requirements:
            try:
                print(f"📦 กำลังติดตั้ง {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                print(f"✅ ติดตั้ง {package} สำเร็จ")
            except subprocess.CalledProcessError:
                print(f"❌ ติดตั้ง {package} ล้มเหลว")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"⚠️  แพ็คเกจที่ล้มเหลว: {failed_packages}")
            print("🔄 กำลังลองใช้วิธีการติดตั้งทางเลือก...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-cache-dir"
                ] + failed_packages)
                print("✅ การติดตั้งทางเลือกสำเร็จ")
            except subprocess.CalledProcessError:
                print("❌ การติดตั้งทางเลือกก็ล้มเหลวเช่นกัน")
                print("\n🔧 ข้อแนะนำในการแก้ไขปัญหา:")
                print("   • ตรวจสอบการเชื่อมต่ออินเทอร์เน็ตของคุณ")
                print("   • เคลียร์พื้นที่ในฮาร์ดดิสก์ (ต้องการประมาณ 5-10GB)")
                print("   • ลองรันด้วยสิทธิ์ Administrator")
                print("   • ปิดโปรแกรมสแกนไวรัสชั่วคราว")
                return False
    
    # ตรวจสอบความถูกต้องหลังการติดตั้ง
    print("\n🔍 กำลังตรวจสอบความถูกต้องของการติดตั้ง...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - อุปกรณ์: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import flask
        print(f"✅ Flask {flask.__version__}")
        
        print("🎉 ติดตั้งและตรวจสอบไลบรารีทั้งหมดเรียบร้อยแล้ว!")
        return True
        
    except ImportError as e:
        print(f"❌ การตรวจสอบล้มเหลว: {e}")
        print("บางแพ็คเกจอาจติดตั้งไม่ถูกต้อง")
        return False

def select_startup_model():
    """เลือกโมเดลเริ่มต้นผ่านหน้าจอโต้ตอบ"""
    print_section("การเลือกโมเดลหลัก (Base Model Selection)")
    
    models = [
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "SDXL Base (แนะนำ) - คุณภาพสูง, ขนาดประมาณ 7GB",
            "size": "~7GB"
        },
        {
            "name": "runwayml/stable-diffusion-v1-5",
            "description": "SD 1.5 - เร็วกว่า, ขนาดเล็กกว่า, ขนาดประมาณ 4GB",
            "size": "~4GB"
        }
    ]
    
    print("เลือกโมเดลหลักที่จะโหลดตอนเริ่มต้น:")
    print()
    
    for i, model in enumerate(models):
        print(f"   [{i+1}] {model['description']}")
        print(f"      ขนาด: {model['size']}")
        print()
    
    print(f"   [{len(models)+1}] ไม่โหลด (โหลดเองภายหลัง)")
    print()
    
    while True:
        try:
            choice_str = input(f"ระบุตัวเลือก [1-{len(models)+1}] (ค่าเริ่มต้นคือ 1): ").strip()
            choice = int(choice_str) if choice_str else 1
            
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                print(f"✅ เลือก: {selected['name']}")
                return selected["name"]
            elif choice == len(models) + 1:
                print("✅ จะไม่มีการโหลดโมเดลตอนเริ่มต้น")
                return "none"
            else:
                print(f"❌ ตัวเลือกไม่ถูกต้อง กรุณาระบุ 1-{len(models)+1}")
                
        except ValueError:
            print("❌ กรุณากรอกเป็นตัวเลขเท่านั้น")

def configure_offline_mode():
    """ตั้งค่าโหมดออฟไลน์พร้อมคำอธิบาย"""
    print_section("การตั้งค่าเครือข่าย")
    
    print("ตัวเลือกโหมดออฟไลน์:")
    print("  • ออนไลน์: ดาวน์โหลดโมเดลจากอินเทอร์เน็ต (แนะนำสำหรับการรันครั้งแรก)")
    print("  • ออฟไลน์: ใช้เฉพาะโมเดลที่ดาวน์โหลดไว้แล้ว (เริ่มเครื่องได้เร็วขึ้น แต่ต้องเคยดาวน์โหลดมาก่อน)")
    print()
    
    response = input("รันในโหมดออฟไลน์หรือไม่? [y/N]: ").lower().strip()
    
    if response in ['y', 'yes']:
        print("✅ เปิดใช้งานโหมดออฟไลน์ - จะใช้เฉพาะโมเดลใน Cache เท่านั้น")
        return True
    else:
        print("✅ เปิดใช้งานโหมดออนไลน์ - จะดาวน์โหลดโมเดลเพิ่มหากจำเป็น")
        return False

def setup_directories():
    """สร้างโครงสร้างโฟลเดอร์ที่จำเป็นสำหรับการทำงาน"""
    print_section("กำลังตั้งค่าโฟลเดอร์")
    
    directories = ["loras", "models", "cache"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 ตรวจสอบ/สร้าง: {directory}/")
    
    print("✅ โครงสร้างโฟลเดอร์พร้อมใช้งาน")

def check_system_requirements():
    """ตรวจสอบสเปคเครื่องและให้คำแนะนำ"""
    print_section("ตรวจสอบความต้องการของระบบ")
    
    try:
        import psutil
        
        # ตรวจสอบ RAM
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f}GB")
        
        if memory_gb < 8:
            print("⚠️  คำเตือน: ตรวจพบ RAM น้อยกว่า 8GB")
            print("   แนะนำให้ปิดแอปพลิเคชันอื่นเพื่อให้ทำงานได้ดีขึ้น")
        else:
            print("✅ RAM: เพียงพอ")
        
        # ตรวจสอบพื้นที่ว่างในดิสก์
        disk_free = psutil.disk_usage('.').free / (1024**3)
        print(f"💽 พื้นที่ว่างในดิสก์: {disk_free:.1f}GB")
        
        if disk_free < 10:
            print("⚠️  คำเตือน: ตรวจพบพื้นที่ว่างน้อยกว่า 10GB")
            print("   โมเดล AI ต้องใช้พื้นที่จัดเก็บค่อนข้างมาก")
        else:
            print("✅ พื้นที่ในดิสก์: เพียงพอ")
            
    except ImportError:
        print("ℹ️  ติดตั้ง psutil เพื่อดูข้อมูลระบบที่ละเอียดขึ้น: pip install psutil")
    
    # ตรวจสอบ GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"🔥 VRAM: {vram_gb:.1f}GB")
            
            if vram_gb < 6:
                print("⚠️  คำเตือน: VRAM น้อยกว่า 6GB")
                print("   แนะนำให้ใช้โมเดลขนาดเล็กหรือลดขนาด Batch")
            else:
                print("✅ GPU: ยอดเยี่ยมสำหรับการสร้างภาพด้วย AI")
        else:
            print("⚠️  ไม่พบ NVIDIA GPU (CUDA) - จะใช้ CPU แทน (ทำงานช้ากว่า)")
            print("   แนะนำให้ใช้ NVIDIA GPU เพื่อประสิทธิภาพที่ดีที่สุด")
    except ImportError:
        print("📦 ยังไม่ได้ติดตั้ง PyTorch")

def main():
    """ฟังก์ชันหลักสำหรับการเริ่มระบบ"""
    print_banner()
    
    # ตรวจสอบเวอร์ชัน Python
    if not check_python_version():
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)
    
    # ตรวจสอบสเปคเครื่อง
    check_system_requirements()
    
    # ตรวจสอบและติดตั้งไลบรารี
    print("\n🔧 กำลังตรวจสอบไลบรารี Python...")
    if not install_dependencies():
        print("\n❌ การติดตั้งไลบรารีล้มเหลว!")
        print("🔧 คุณสามารถลองติดตั้งด้วยตัวเองได้โดยใช้คำสั่ง:")
        print("   pip install torch torchvision torchaudio diffusers transformers flask")
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)
    
    # ตั้งค่าโฟลเดอร์
    setup_directories()
    
    # เลือกโมเดล
    chosen_model = select_startup_model()
    
    # ตั้งค่าโหมดออฟไลน์
    offline_mode = configure_offline_mode()
    
    # เตรียมการขั้นตอนสุดท้าย
    print_section("กำลังเริ่มเซิร์ฟเวอร์")
    
    # ล้าง Python cache เก่า
    try:
        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
        print("🧹 ล้าง Python cache เรียบร้อย")
    except:
        pass
    
    # เริ่มต้นเซิร์ฟเวอร์
    try:
        print("🚀 กำลังนำเข้าโมดูลเซิร์ฟเวอร์...")
        from sd_server import main as run_server
        
        print("🌟 กำลังเริ่มเซิร์ฟเวอร์ด้วยการตั้งค่าดังนี้:")
        print(f"   • โมเดล: {chosen_model}")
        print(f"   • ออฟไลน์: {offline_mode}")
        print(f"   • URL: http://127.0.0.1:5000")
        
        print("\n" + "="*60)
        print("🎉 กำลังเริ่ม LOCAL AI GENERATOR SERVER...")
        print("🎨 พร้อมสำหรับการเชื่อมต่อจากปลั๊กอิน Aseprite แล้ว!")
        print("="*60)
        
        # รันเซิร์ฟเวอร์
        run_server(default_model_to_load=chosen_model, offline=offline_mode)
        
    except KeyboardInterrupt:
        print("\n\n👋 เซิร์ฟเวอร์ถูกหยุดโดยผู้ใช้")
        print("ขอบคุณที่ใช้ Local AI Generator!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ เกิดข้อผิดพลาดที่เซิร์ฟเวอร์:")
        print(f"   {str(e)}")
        print("\n🐛 รายละเอียดข้อผิดพลาดทั้งหมด:")
        traceback.print_exc()
        print("\n🔧 แนวทางการแก้ไข:")
        print("   • ตรวจสอบว่าพอร์ต 5000 ถูกใช้งานอยู่หรือไม่")
        print("   • ตรวจสอบการติดตั้งไลบรารีทั้งหมด")
        print("   • ลองรันด้วยสิทธิ์ Administrator")
        input("\nกด Enter เพื่อออก...")
        sys.exit(1)

if __name__ == "__main__":
    main()