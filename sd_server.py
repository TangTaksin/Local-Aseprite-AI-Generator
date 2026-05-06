#!/usr/bin/env python3
"""
เซิร์ฟเวอร์ Local AI Generator สำหรับ Aseprite
การสร้าง Pixel Art ระดับมืออาชีพโดยใช้ Stable Diffusion พร้อมรองรับ LoRA และการลบพื้นหลังด้วย BiRefNet
เวอร์ชัน 2.0 - ปรับปรุงและเพิ่มประสิทธิภาพสำหรับการใช้งานจริง
"""
import os
import sys
import json
import base64
import io
import warnings
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np

# ปิดการแจ้งเตือนคำเตือนต่างๆ เพื่อให้ Console สะอาด
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")

# ปิดคำเตือนของ Flask development server
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

class PixelArtSDServer:
    def __init__(self):
        self.pipeline = None
        self.segmentation_model = None
        self.segmentation_processor = None
        self.model_loaded = False
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}
        self.offline_mode = False
        
        # ตั้งค่าพื้นฐานสำหรับการสร้างภาพ Pixel Art
        self.default_settings = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels"
        }
        
        print(f"🚀 Local AI Generator Server v2.0")
        print(f"📱 อุปกรณ์ที่ใช้: {self.device}")
        print(f"🔥 ใช้งาน CUDA ได้: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def load_segmentation_model(self):
        """โหลดโมเดล BiRefNet สำหรับการลบพื้นหลังระดับมืออาชีพ"""
        if self.segmentation_model and self.segmentation_processor:
            return True
            
        print("📦 กำลังโหลดโมเดล BiRefNet สำหรับการลบพื้นหลัง...")
        
        try:
            model_name = 'zhengpeng7/BiRefNet'
            
            # ตั้งค่าตัวประมวลผลภาพ (Preprocessing)
            self.segmentation_processor = transforms.Compose([
                transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # โหลดโมเดลสำหรับ Segmentation
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            
            # ย้ายไป GPU และบังคับเป็น Float32 เพื่อป้องกันปัญหา Dtype Mismatch
            self.segmentation_model.to(self.device)
            self.segmentation_model.float() 
            self.segmentation_model.eval()
            
            print("✅ โหลดโมเดล BiRefNet สำเร็จ (โหมด Float32)")
            return True
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล BiRefNet: {e}")
            return False

    def remove_background(self, pil_image):
        """ใช้ BiRefNet เพื่อสร้าง Mask ความโปร่งใสคุณภาพสูง"""
        if not self.load_segmentation_model():
            raise Exception("ไม่สามารถโหลดโมเดลลบพื้นหลังได้")
            
        print("🎭 กำลังลบพื้นหลังด้วย BiRefNet...")
        try:
            with torch.no_grad():
                # แปลงเป็น RGB เพื่อเข้ากระบวนการ
                rgb_image = pil_image.convert("RGB")
                
                # เตรียม Tensor สำหรับ Input
                input_tensor = self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)

                # บังคับชนิดข้อมูลให้ตรงกับโมเดล (ป้องกัน Error float/Half)
                input_tensor = input_tensor.to(dtype=next(self.segmentation_model.parameters()).dtype)
                
                # สร้าง Mask
                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]
                
                # ปรับขนาด Mask ให้เท่ากับรูปภาพต้นฉบับ
                mask = F.interpolate(logits, size=pil_image.size[::-1], mode='nearest')
                mask = torch.sigmoid(mask).squeeze()
                
                # สร้าง Binary Mask ที่ขอบคมชัด (เหมาะสำหรับ Pixel Art)
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
                
            # นำ Mask ไปใช้เพื่อสร้างพื้นหลังโปร่งใส (RGBA)
            mask_image = Image.fromarray(binary_mask * 255, mode='L')
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)
            
            print("✅ ลบพื้นหลังเสร็จสิ้น")
            return rgba_image
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดระหว่างการลบพื้นหลัง: {e}")
            return pil_image.convert("RGBA")

    def image_to_base64(self, image):
        """แปลงภาพ PIL เป็น base64 encoded string"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return base64.b64encode(image.tobytes()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """โหลดและจัดเก็บโมเดล AI ใน Cache เพื่อประสิทธิภาพในการเรียกใช้งาน"""
        try:
            local_only = self.offline_mode
            if local_only:
                print("🔒 เปิดใช้งานโหมดออฟไลน์: กำลังโหลดจากหน่วยความจำภายในเท่านั้น")
                
            # ตรวจสอบว่าโมเดลอยู่ใน Cache หรือไม่
            if model_name in self.model_cache:
                print(f"⚡ กำลังดึง {model_name} จาก Cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True
            
            print(f"📥 กำลังโหลด Base Model: {model_name}")
            precision = torch.float16 if self.device == "cuda" else torch.float32
            
            # โหลด Pipeline ตามประเภทของโมเดล
            if "xl" in model_name.lower():
                print("🔧 กำลังโหลด SDXL pipeline พร้อม VAE ที่ปรับแต่งแล้ว...")
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=precision,
                    local_files_only=local_only
                )
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    vae=vae,
                    torch_dtype=precision,
                    use_safetensors=True,
                    local_files_only=local_only
                )
            else:
                print("🔧 กำลังโหลด SD 1.5 pipeline...")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=precision,
                    use_safetensors=True,
                    local_files_only=local_only
                )
            
            # ย้าย Pipeline ไปยังอุปกรณ์ (GPU/CPU) และปรับปรุงประสิทธิภาพ
            self.pipeline = self.pipeline.to(self.device)
            
            # เก็บโมเดลลง Cache
            self.model_cache[model_name] = self.pipeline
            self.current_model = model_name
            self.model_loaded = True
            
            print(f"✅ โหลดโมเดลสำเร็จ: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถโหลดโมเดล {model_name} ได้: {str(e)}")
            return False

    def generate_image(self, prompt, lora_model=None, lora_strength=1.0, **kwargs):
        """สร้างภาพด้วยการควบคุมสไตล์ผ่าน LoRA และการตั้งค่าที่เหมาะสม"""
        if not self.model_loaded:
            raise Exception("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")
            
        try:
            print(f"🎨 กำลังสร้างภาพด้วย Prompt: '{prompt[:50]}...'")
            
            pipeline_kwargs = {}
            
            # จัดการการโหลด LoRA และกำหนดระดับความเข้มข้น (Strength)
            if lora_model and lora_model.lower() not in ['none', '']:
                print(f"🎭 กำลังโหลด LoRA: {lora_model} (ความแรง: {lora_strength})")
                
                if os.path.exists(lora_model):
                    # โหลดจากไฟล์ LoRA ในเครื่อง
                    lora_path, weight_name = os.path.split(lora_model)
                    self.pipeline.load_lora_weights(lora_path, weight_name=weight_name)
                else:
                    # โหลดจาก Hugging Face Hub
                    self.pipeline.load_lora_weights(lora_model)
                
                # กำหนดค่าความแรงของ LoRA
                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
            
            # รวมค่าเริ่มต้นกับค่าที่ส่งมาจาก Request
            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)
            
            # กำหนดขนาดภาพพื้นฐานตามรุ่นของโมเดล
            if "width" not in kwargs or "height" not in kwargs:
                if "xl" in self.current_model.lower():
                    gen_params.setdefault('width', 1024)
                    gen_params.setdefault('height', 1024)
                else:
                    gen_params.setdefault('width', 512)
                    gen_params.setdefault('height', 512)
            
            # ต่อท้าย Prompt เพื่อบังคับความเป็น Pixel Art
            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]
            
            # จัดการเรื่อง Seed
            seed = gen_params.get("seed", -1)
            generator = torch.Generator(device=self.device)
            
            if seed is not None and int(seed) != -1:
                generator.manual_seed(int(seed))
                print(f"🎲 ใช้ Seed: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(random_seed)
                print(f"🎲 สุ่ม Seed ใหม่: {random_seed}")
                seed = random_seed
            
            # เตรียมพารามิเตอร์สุดท้ายสำหรับ Pipeline
            pipeline_kwargs.update({
                "prompt": prompt,
                "negative_prompt": gen_params["negative_prompt"],
                "width": gen_params["width"],
                "height": gen_params["height"],
                "num_inference_steps": int(gen_params["num_inference_steps"]),
                "guidance_scale": float(gen_params["guidance_scale"]),
                "generator": generator
            })
            
            print(f"⚙️ การตั้งค่า: {gen_params['width']}x{gen_params['height']}, {gen_params['num_inference_steps']} steps")
            
            # เริ่มกระบวนการสร้างภาพ
            result = self.pipeline(**pipeline_kwargs)
            
            print("✅ สร้างภาพสำเร็จ")
            return result.images[0], generator.initial_seed()
        
        finally:
            # ล้างค่า LoRA ออกจาก Pipeline เพื่อไม่ให้ปนเปื้อนกับการสร้างภาพครั้งต่อไป
            if lora_model and lora_model.lower() not in ['none', ''] and hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()

    def process_for_pixel_art(self, image, target_size=(64, 64), colors=16):
        """การประมวลผลภาพขั้นสูงเพื่อให้เป็น Pixel Art และจำกัดจำนวนสี"""
        print(f"🖼️ กำลังแปลงเป็น Pixel Art: ขนาด {target_size}, จำนวนสี {colors}")
        
        # ปรับขนาดภาพแบบ Nearest Neighbor เพื่อให้พิกเซลคมชัด ไม่เบลอ
        image = image.resize(target_size, Image.NEAREST)
        
        # จำกัดจำนวนสี (Color Quantization)
        if colors > 0:
            if image.mode == 'RGBA':
                # แยก Alpha Channel ไว้เพื่อรักษาความโปร่งใสหลังจำกัดสี
                alpha = image.getchannel('A')
                rgb_image = image.convert('RGB').quantize(
                    colors=int(colors) - 1,  # กันไว้ 1 สีสำหรับ Transparency
                    method=Image.MEDIANCUT
                )
                image = rgb_image.convert('RGBA')
                image.putalpha(alpha)
            else:
                image = image.quantize(
                    colors=int(colors),
                    method=Image.MEDIANCUT
                ).convert('RGB')
        
        print("✅ ประมวลผล Pixel Art สำเร็จ")
        return image

# สร้าง Instance ของเซิร์ฟเวอร์
sd_server = PixelArtSDServer()

@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint หลักสำหรับรับคำสั่งสร้างภาพ"""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400
        
        print(f"\n🎯 คำขอสร้างภาพใหม่: {prompt[:30]}...")
        
        defaults = sd_server.default_settings
        kwargs = {
            "lora_model": data.get('lora_model'),
            "lora_strength": data.get('lora_strength', 1.0),
            "num_inference_steps": data.get('steps', defaults.get('num_inference_steps')),
            "guidance_scale": data.get('guidance_scale', defaults.get('guidance_scale')),
            "seed": data.get('seed', -1),
            "negative_prompt": data.get('negative_prompt', defaults.get('negative_prompt')),
            "width": data.get('width', 1024),   # ความละเอียดฐานในการสร้าง
            "height": data.get('height', 1024)  # ความละเอียดฐานในการสร้าง
        }
        
        # ขั้นตอนสร้างภาพหลัก
        start_time = datetime.now()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)
        
        # ลบพื้นหลังถ้ามีการร้องขอ
        if data.get('remove_background', False):
            image = sd_server.remove_background(image)
        
        # แปลงเป็น Pixel Art ตามขนาดที่กำหนดจาก Aseprite
        pixel_width = int(data.get('pixel_width', 64))
        pixel_height = int(data.get('pixel_height', 64))
        colors = int(data.get('colors', 16))
        
        pixel_image = sd_server.process_for_pixel_art(
            image,
            target_size=(pixel_width, pixel_height),
            colors=colors
        )
        
        # ส่งภาพกลับในรูปแบบ Base64
        img_base64 = sd_server.image_to_base64(pixel_image)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ เวลาที่ใช้ทั้งหมด: {generation_time:.2f} วินาที")
        
        return jsonify({
            "success": True,
            "image": {
                "base64": img_base64,
                "width": pixel_width,
                "height": pixel_height,
                "mode": "rgba"
            },
            "seed": used_seed,
            "prompt": prompt,
            "generation_time": generation_time
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ เกิดข้อผิดพลาดในการสร้างภาพ: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะของเซิร์ฟเวอร์"""
    return jsonify({
        "status": "healthy",
        "model_loaded": sd_server.model_loaded,
        "current_model": sd_server.current_model,
        "device": sd_server.device,
        "version": "2.0.0"
    })

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """โหลดโมเดล AI ที่ระบุ"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ model_name"}), 400
        
        print(f"📦 กำลังโหลดโมเดล: {model_name}")
        
        if sd_server.load_model(model_name):
            return jsonify({
                "success": True,
                "model": model_name,
                "device": sd_server.device
            })
        else:
            return jsonify({
                "success": False,
                "error": f"ไม่สามารถโหลด {model_name} ได้"
            }), 500
            
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """แสดงรายการโมเดลที่มีให้เลือก"""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5"
    ]
    return jsonify({"models": models})

@app.route('/loras', methods=['GET'])
def list_loras():
    """แสดงรายการ LoRA ทั้งจาก Hub และในเครื่อง"""
    lora_models = [
        "None",
        "nerijs/pixel-art-xl",
        "ntc-ai/SDXL-LoRA-slider.pixel-art"
    ]
    
    # สแกนหาไฟล์ LoRA เพิ่มเติมในโฟลเดอร์ 'loras'
    lora_directory = "loras"
    if not os.path.isdir(lora_directory):
        os.makedirs(lora_directory)
    
    for filename in os.listdir(lora_directory):
        if filename.endswith(".safetensors"):
            lora_path = os.path.join(lora_directory, filename).replace("\\", "/")
            if lora_path not in lora_models:
                lora_models.append(lora_path)
    
    return jsonify({"loras": lora_models})

def main(default_model_to_load=None, offline=False):
    """ฟังก์ชันหลักสำหรับเริ่มการทำงานของเซิร์ฟเวอร์"""
    print("\n" + "="*50)
    print("🎮 LOCAL AI GENERATOR SERVER v2.0")
    print("="*50)
    
    sd_server.offline_mode = offline
    
    if default_model_to_load and default_model_to_load.lower() != "none":
        print(f"📦 กำลังเตรียมโหลดโมเดลเริ่มต้น: {default_model_to_load}")
        sd_server.load_model(default_model_to_load)
    else:
        print("⏸️ จะยังไม่โหลดโมเดลตอนเริ่มเครื่อง (จะโหลดเมื่อได้รับคำสั่งครั้งแรก)")
    
    print("\n🌐 ข้อมูลการตั้งค่าเซิร์ฟเวอร์:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • อุปกรณ์: {sd_server.device}")
    print(f"   • โหมดออฟไลน์: {offline}")
    
    print("\n✅ เซิร์ฟเวอร์พร้อมใช้งาน! เข้าถึงได้ที่ http://127.0.0.1:5000")
    print("🎨 ปลั๊กอิน Aseprite สามารถเชื่อมต่อและเริ่มสร้างภาพได้ทันที!")
    print("\n" + "="*50 + "\n")
    
    try:
        # ปิดการแสดง Banner ของ Flask
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 กำลังปิดเซิร์ฟเวอร์...")
    except Exception as e:
        print(f"❌ เซิร์ฟเวอร์ขัดข้อง: {e}")

if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")