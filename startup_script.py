#!/usr/bin/env python3
"""
Local AI Generator สำหรับ Aseprite
เวอร์ชัน 2.2
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path


def print_banner():
    print("\n" + "=" * 60)
    print("🎮 LOCAL AI GENERATOR FOR ASEPRITE v2.2")
    print("=" * 60)


def print_section(title):
    """พิมพ์หัวข้อส่วนต่างๆ ในรูปแบบที่กำหนด"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))


def check_python_version():
    """ตรวจสอบว่าเวอร์ชันของ Python ตรงตามความต้องการหรือไม่ (เป้าหมายคือ 3.11.x)"""
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ เวอร์ชัน Python เก่าเกินไปสำหรับระบบนี้")
        print(f"   เวอร์ชันปัจจุบัน: {version.major}.{version.minor}.{version.micro}")
        print("   กรุณาใช้ Python 3.10.x (แนะนำ 3.10 ขึ้นไป)")
        return False

    if version.major == 3 and version.minor > 14:
        print(
            f"⚠️ คำเตือน: คุณกำลังใช้ Python {version.major}.{version.minor}.{version.micro}"
        )
        print("   หากรันแล้วพบข้อผิดพลาด แนะนำให้ดาวน์เกรดกลับมาที่ 3.14")
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - ใช้งานได้")

    return True


def install_dependencies():
    """ติดตั้งไลบรารีตามเวอร์ชันที่ทดสอบแล้วว่าใช้งานกับ Pony XL ได้ชัวร์"""
    print("\n📋 กำลังจัดการไลบรารี (Internal Dependency Management)")
    print("-" * 30)

    # อัปเกรด pip
    print("📦 กำลังอัปเกรด pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # ติดตั้ง PyTorch
    print("\n🔥 กำลังติดตั้ง PyTorch (CUDA 13.0)")
    torch_packages = [
        "torch",
        "torchvision",
        "torchaudio",
    ]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *torch_packages,
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
    )

    # ติดตั้งไลบรารี AI และระบบเสริม (ระบุเวอร์ชันเพื่อป้องกัน Conflict)
    print("\n📚 กำลังติดตั้งไลบรารี AI และระบบเสริม...")
    other_requirements = [
        "diffusers==0.38.0",
        "transformers==4.57.6",
        "safetensors==0.8.0rc0",
        "accelerate==1.13.0",
        "huggingface-hub==0.36.2",
        "flask",
        "flask-cors",
        "sentencepiece",
        "kornia",
        "einops",
        "pillow",
        "peft",
        "opencv-python",
        "timm",
        "numpy",
    ]

    # ใช้การติดตั้งแบบปกติ
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + other_requirements)

    print("\n✅ ติดตั้งไลบรารีทั้งหมดเรียบร้อยแล้ว!")
    return True


def select_startup_model():
    """เลือกโมเดลเริ่มต้นผ่านหน้าจอโต้ตอบ"""
    print_section("การเลือกโมเดลหลัก (Base Model Selection)")

    # ดึงรายชื่อโมเดลจากโฟลเดอร์ models เพื่อแสดงในตัวเลือกด้วย
    local_models = []
    if os.path.exists("models"):
        local_models = [
            f for f in os.listdir("models") if f.endswith((".safetensors", ".ckpt"))
        ]

    models = [
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "SDXL Base (แนะนำ) - คุณภาพสูง",
            "size": "~7GB",
        },
        {
            "name": "runwayml/stable-diffusion-v1-5",
            "description": "SD 1.5 - เร็วกว่า, ขนาดเล็กกว่า",
            "size": "~4GB",
        },
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
    return response in ["y", "yes"]


def setup_directories():
    """สร้างโครงสร้างโฟลเดอร์ที่จำเป็นสำหรับการทำงาน"""
    print_section("กำลังตั้งค่าโฟลเดอร์")

    # เพิ่มโฟลเดอร์ outputs เพื่อเก็บประวัติการเจน
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

        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f}GB")
        disk_free = psutil.disk_usage(".").free / (1024**3)
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
                print("🚀 GPU ของคุณยอดเยี่ยมมาก")
        else:
            print("⚠️ ไม่พบ NVIDIA GPU (CUDA) - จะใช้ CPU แทน")
    except ImportError:
        print("📦 ยังไม่ได้ติดตั้ง PyTorch")
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")


def main():
    """ฟังก์ชันหลักสำหรับการเริ่มระบบ"""
    print_banner()

    if not check_python_version():
        print("   หากรันแล้วพบข้อผิดพลาด แนะนำให้ดาวน์เกรดกลับมาที่ 3.14")
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

        print("\n" + "=" * 60)
        print("🎉 เซิร์ฟเวอร์พร้อมทำงาน!")
        print(f"   • โมเดลเริ่มต้น: {chosen_model}")
        print(f"   • โหมดออฟไลน์: {offline_mode}")
        print("=" * 60)

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
