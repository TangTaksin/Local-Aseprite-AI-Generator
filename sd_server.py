#!/usr/bin/env python3
"""
เซิร์ฟเวอร์ Local AI Generator สำหรับ Aseprite - Optimized for RTX 5070 Ti
การสร้าง Pixel Art ระดับมืออาชีพโดยใช้ Stable Diffusion พร้อมรองรับ LoRA และการลบพื้นหลังด้วย BiRefNet
เวอร์ชัน 2.2 - Optimized for Blackwell Architecture (RTX 5070 Ti)
"""

from diffusers.utils import logging as diffusers_logging
import logging
import os
import sys
import json
import platform
import base64
import io
import warnings
import glob
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
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
diffusers_logging.set_verbosity_error()
log = logging.getLogger("werkzeug")
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

        is_windows = platform.system() == "Windows"
        self.use_compile = False if is_windows else True

        # 🎯 RTX 5070 Ti Optimized Settings
        self.optimized_settings = {
            "use_bf16": torch.cuda.is_bf16_supported(),
            "use_compile": self.use_compile,
            "enable_xformers": not is_windows,
            "enable_attention_slicing": True,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",  # or "medium" for more speed
        }

        # ตั้งค่าพื้นฐานสำหรับการสร้างภาพ Pixel Art
        self.default_settings = {
            "num_inference_steps": 25,  # Reduced from 30 (DPM++ converges faster)
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels",
        }

        print(f"🚀 Local AI Generator Server v2.2 [RTX 5070 Ti Optimized]")
        print(f"📱 อุปกรณ์ที่ใช้: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"💾 VRAM: {props.total_memory / 1024**3:.1f}GB")
            print(f"⚡ Compute Capability: {props.major}.{props.minor}")
            print(f"🔷 BF16 Support: {torch.cuda.is_bf16_supported()}")

            # 🎯 Apply global CUDA optimizations for Blackwell
            self._apply_global_cuda_optimizations()

    def _apply_global_cuda_optimizations(self):
        """Apply global CUDA optimizations for RTX 5070 Ti"""
        if self.device != "cuda":
            return

        print("\n🔧 Applying RTX 5070 Ti Global Optimizations...")

        # 1. Enable cuDNN benchmark for consistent input sizes (pixel art = fixed resolutions)
        if self.optimized_settings["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
            print("   ✅ cuDNN Benchmark: Enabled (faster for fixed resolutions)")

        # 2. Set float32 matmul precision for Tensor Cores
        precision = self.optimized_settings["float32_matmul_precision"]
        try:
            torch.set_float32_matmul_precision(precision)
            print(f"   ✅ TF32 Precision: {precision} (optimized for Tensor Cores)")
        except:
            print(f"   ⚠️ Could not set matmul precision")

        # 3. Enable TF32 for CUDA matmul operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   ✅ TF32 Acceleration: Enabled")

        # 4. Pre-allocate some VRAM to reduce fragmentation (optional)
        # torch.cuda.set_per_process_memory_fraction(0.95, 0)  # Use 95% of VRAM

        print("   ✅ Global CUDA optimizations applied!\n")

    def _optimize_pipeline_for_blackwell(self, pipeline, model_type="sdxl"):
        if self.device != "cuda":
            return pipeline

        print(f"🎯 Optimizing {model_type.upper()} pipeline for RTX 5070 Ti...")

        dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
        pipeline = pipeline.to(self.device, dtype=dtype)
        print(f"   ✅ Precision: {dtype}")

        # 📝 channels_last disabled: Blackwell SDXL performs faster with default contiguous layout
        print("   ✅ Memory Layout: contiguous (default)")

        try:
            if hasattr(pipeline, "disable_attention_slicing"):
                pipeline.disable_attention_slicing()  # SDPA runs fastest without slicing
            print("   ✅ Attention: Native SDPA (Blackwell optimized)")
        except Exception as e:
            print(f"   ⚠️ Attention setup note: {e}")

        try:
            pipeline.enable_vae_slicing()
            print("   ✅ VAE Slicing: Enabled")
        except:
            pass

        if model_type == "sdxl" and hasattr(pipeline, "scheduler"):
            try:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
                print("   ✅ Scheduler: DPM++ 2M Karras")
            except:
                pass

        print("   ✅ Pipeline optimization complete!\n")
        return pipeline

    def load_segmentation_model(self):
        """โหลดโมเดล BiRefNet สำหรับการลบพื้นหลังระดับมืออาชีพ - Optimized"""
        if self.segmentation_model and self.segmentation_processor:
            return True

        print("📦 Loading BiRefNet for background removal (optimized)...")

        try:
            model_name = "zhengpeng7/BiRefNet"

            # Setup preprocessing
            self.segmentation_processor = transforms.Compose(
                [
                    transforms.Resize(
                        (352, 352), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            # Load model with optimizations
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()

            # Optimize segmentation model too
            if self.device == "cuda":
                try:
                    self.segmentation_model.to(memory_format=torch.channels_last)
                except:
                    pass

            print("✅ BiRefNet loaded successfully (optimized)")
            return True

        except Exception as e:
            print(f"❌ Error loading BiRefNet: {e}")
            return False

    def remove_background(self, pil_image):
        """ใช้ BiRefNet เพื่อสร้าง Mask ความโปร่งใสคุณภาพสูง - Optimized"""
        if not self.load_segmentation_model():
            raise Exception("ไม่สามารถโหลดโมเดลลบพื้นหลังได้")

        print("🎭 Removing background with BiRefNet...")
        try:
            # Use inference_mode for slightly better performance than no_grad
            with torch.inference_mode():
                rgb_image = pil_image.convert("RGB")
                input_tensor = (
                    self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)
                )

                # Match dtype to model
                model_dtype = next(self.segmentation_model.parameters()).dtype
                input_tensor = input_tensor.to(dtype=model_dtype)

                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]

                # Resize mask to original size
                mask = F.interpolate(
                    logits,
                    size=pil_image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.sigmoid(mask).squeeze()

                # Create sharp binary mask for pixel art
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)

            # Apply mask
            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)

            print("✅ Background removal complete")
            return rgba_image

        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            return pil_image.convert("RGBA")

    def image_to_base64(self, image):
        """แปลงภาพ PIL เป็น base64 encoded string"""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return base64.b64encode(image.tobytes()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """โหลดและจัดเก็บโมเดล AI ใน Cache - RTX 5070 Ti Optimized"""
        try:
            local_only = self.offline_mode

            # Check cache first
            if model_name in self.model_cache:
                print(f"⚡ Loading {model_name} from cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True

            print(f"📥 Loading model: {model_name}")

            # Determine model type and precision
            is_sdxl = "xl" in model_name.lower()
            precision = (
                torch.bfloat16
                if (self.device == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            print(f"   ✅ Using precision: {precision}")

            # Check for local model file
            local_model_path = os.path.join("models", model_name)
            is_local_file = os.path.isfile(local_model_path)

            if is_local_file:
                print(f"📂 Found local model: {local_model_path}")

                if is_sdxl:
                    print("⚙️ Loading SDXL with Blackwell optimizations...")
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=precision,
                        local_files_only=local_only,
                    )
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        local_model_path,
                        vae=vae,
                        torch_dtype=precision,
                        use_safetensors=True,
                        config="stabilityai/stable-diffusion-xl-base-1.0",
                    )
                else:
                    print("🔧 Loading SD 1.5 from local file...")
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        local_model_path,
                        torch_dtype=precision,
                        load_safety_checker=False,
                    )
            else:
                # Load from HuggingFace
                if is_sdxl:
                    print("🌐 Loading SDXL from HuggingFace...")
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=precision,
                        local_files_only=local_only,
                    )
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_name,
                        vae=vae,
                        torch_dtype=precision,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )
                else:
                    print("🌐 Loading SD 1.5 from HuggingFace...")
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=precision,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )

            # 🎯 Apply RTX 5070 Ti specific optimizations
            model_type = "sdxl" if is_sdxl else "sd15"
            self.pipeline = self._optimize_pipeline_for_blackwell(
                self.pipeline, model_type
            )

            # 🚀 Warmup: Run a dummy inference to compile CUDA kernels
            if self.device == "cuda":
                print("🔥 Running warmup inference (caching kernels & VRAM)...")
                try:
                    with torch.inference_mode():
                        _ = self.pipeline(
                            prompt="pixel",
                            width=512 if not is_sdxl else 1024,
                            height=512 if not is_sdxl else 1024,
                            num_inference_steps=1,
                            guidance_scale=1.0,
                        )
                    print("✅ Warmup complete - Ready for fast inference!")
                except Exception as e:
                    print(f"⚠️ Warmup skipped: {e}")

            # Cache the optimized pipeline
            self.model_cache[model_name] = self.pipeline
            self.current_model = model_name
            self.model_loaded = True

            print(f"✅ Model loaded and optimized: {model_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def generate_image(self, prompt, lora_model=None, lora_strength=1.0, **kwargs):
        """สร้างภาพด้วยการควบคุมสไตล์ผ่าน LoRA - Optimized inference"""
        if not self.model_loaded:
            raise Exception("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")

        try:
            print(f"🎨 Generating: '{prompt[:50]}...'")

            pipeline_kwargs = {}

            # Handle LoRA loading
            if lora_model and lora_model.lower() not in ["none", ""]:
                print(f"🎭 Loading LoRA: {lora_model} (strength: {lora_strength})")

                full_lora_path = lora_model
                if not os.path.exists(full_lora_path):
                    full_lora_path = os.path.join("loras", lora_model)

                if os.path.exists(full_lora_path):
                    lora_dir, weight_name = os.path.split(full_lora_path)
                    self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
                else:
                    self.pipeline.load_lora_weights(lora_model)

                pipeline_kwargs["cross_attention_kwargs"] = {
                    "scale": float(lora_strength)
                }

            # Merge settings
            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)

            # Set default resolution based on model
            if "width" not in kwargs or "height" not in kwargs:
                if "xl" in self.current_model.lower():
                    gen_params.setdefault("width", 1024)
                    gen_params.setdefault("height", 1024)
                else:
                    gen_params.setdefault("width", 512)
                    gen_params.setdefault("height", 512)

            # Add pixel art suffix
            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]

            # Handle seed
            seed = gen_params.get("seed", -1)
            generator = torch.Generator(device=self.device)

            if seed is not None and int(seed) != -1:
                generator.manual_seed(int(seed))
                print(f"🎲 Using seed: {seed}")
            else:
                import random

                random_seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(random_seed)
                print(f"🎲 Random seed: {random_seed}")
                seed = random_seed

            # Prepare final pipeline kwargs
            pipeline_kwargs.update(
                {
                    "prompt": prompt,
                    "negative_prompt": gen_params["negative_prompt"],
                    "width": gen_params["width"],
                    "height": gen_params["height"],
                    "num_inference_steps": int(gen_params["num_inference_steps"]),
                    "guidance_scale": float(gen_params["guidance_scale"]),
                    "generator": generator,
                }
            )

            print(
                f"⚙️ Settings: {gen_params['width']}x{gen_params['height']}, {gen_params['num_inference_steps']} steps"
            )

            # 🎯 Use inference_mode for better performance
            with torch.inference_mode():
                result = self.pipeline(**pipeline_kwargs)

            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()

        finally:
            # Unload LoRA to prevent contamination
            if (
                lora_model
                and lora_model.lower() not in ["none", ""]
                and hasattr(self.pipeline, "unload_lora_weights")
            ):
                self.pipeline.unload_lora_weights()

            # # 🧹 Clean up VRAM after generation
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()

    def process_for_pixel_art(self, image, target_size=(64, 64), colors=16):
        """การประมวลผลภาพขั้นสูงเพื่อให้เป็น Pixel Art - Optimized"""
        # Resize with NEAREST for crisp pixels
        image = image.resize(target_size, Image.NEAREST)

        # Color quantization
        if colors > 0:
            if image.mode == "RGBA":
                alpha = image.getchannel("A")
                rgb_image = image.convert("RGB").quantize(
                    colors=int(colors) - 1,
                    method=Image.MEDIANCUT,
                    dither=Image.NONE,  # No dithering for clean pixel art
                )
                image = rgb_image.convert("RGBA")
                image.putalpha(alpha)
            else:
                image = image.quantize(
                    colors=int(colors), method=Image.MEDIANCUT, dither=Image.NONE
                ).convert("RGB")

        return image


# Create server instance
sd_server = PixelArtSDServer()


@app.route("/generate", methods=["POST"])
def generate():
    """Endpoint หลักสำหรับรับคำสั่งสร้างภาพ"""
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400

        print(f"\n🎯 New generation request: {prompt[:30]}...")

        defaults = sd_server.default_settings
        kwargs = {
            "lora_model": data.get("lora_model"),
            "lora_strength": data.get("lora_strength", 1.0),
            "num_inference_steps": data.get(
                "steps", defaults.get("num_inference_steps")
            ),
            "guidance_scale": data.get(
                "guidance_scale", defaults.get("guidance_scale")
            ),
            "seed": data.get("seed", -1),
            "negative_prompt": data.get(
                "negative_prompt", defaults.get("negative_prompt")
            ),
            "width": data.get("width", 1024),
            "height": data.get("height", 1024),
        }

        # Generate image
        start_time = time.time()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)

        # Remove background if requested
        if data.get("remove_background", False):
            image = sd_server.remove_background(image)

        # Convert to pixel art
        pixel_width = int(data.get("pixel_width", 64))
        pixel_height = int(data.get("pixel_height", 64))
        colors = int(data.get("colors", 16))

        pixel_image = sd_server.process_for_pixel_art(
            image, target_size=(pixel_width, pixel_height), colors=colors
        )

        # Encode to base64
        img_base64 = sd_server.image_to_base64(pixel_image)

        generation_time = time.time() - start_time
        print(f"⏱️ Total time: {generation_time:.2f}s")

        return jsonify(
            {
                "success": True,
                "image": {
                    "base64": img_base64,
                    "width": pixel_width,
                    "height": pixel_height,
                    "mode": "rgba",
                },
                "seed": used_seed,
                "prompt": prompt,
                "generation_time": generation_time,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """ตรวจสอบสถานะของเซิร์ฟเวอร์"""
    vram_used = 0
    vram_total = 0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": sd_server.model_loaded,
            "current_model": sd_server.current_model,
            "device": sd_server.device,
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 2),
            "version": "2.2.0-rtx5070ti",
        }
    )


@app.route("/load_model", methods=["POST"])
def load_model_route():
    """โหลดโมเดล AI ที่ระบุ"""
    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ model_name"}), 400

        print(f"📦 Loading model: {model_name}")

        if sd_server.load_model(model_name):
            return jsonify(
                {"success": True, "model": model_name, "device": sd_server.device}
            )
        else:
            return (
                jsonify({"success": False, "error": f"ไม่สามารถโหลด {model_name} ได้"}),
                500,
            )

    except Exception as e:
        print(f"❌ Model load error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models():
    """แสดงรายการโมเดลที่มีให้เลือก"""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]

    model_directory = "models"
    os.makedirs(model_directory, exist_ok=True)

    for filename in os.listdir(model_directory):
        if filename.endswith(".safetensors") or filename.endswith(".ckpt"):
            if filename not in models:
                models.append(filename)

    return jsonify({"models": models})


@app.route("/loras", methods=["GET"])
def list_loras():
    """แสดงรายการ LoRA"""
    lora_models = ["None", "nerijs/pixel-art-xl", "ntc-ai/SDXL-LoRA-slider.pixel-art"]

    lora_directory = "loras"
    os.makedirs(lora_directory, exist_ok=True)

    for filename in os.listdir(lora_directory):
        if filename.endswith(".safetensors"):
            if filename not in lora_models:
                lora_models.append(filename)

    return jsonify({"loras": lora_models})


def main(default_model_to_load=None, offline=False):
    """ฟังก์ชันหลักสำหรับเริ่มการทำงานของเซิร์ฟเวอร์"""
    print("\n" + "=" * 60)
    print("🎮 LOCAL AI GENERATOR SERVER v2.2 [RTX 5070 Ti OPTIMIZED]")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)

    sd_server.offline_mode = offline

    if default_model_to_load and default_model_to_load.lower() != "none":
        print(f"📦 Pre-loading model: {default_model_to_load}")
        sd_server.load_model(default_model_to_load)
    else:
        print("⏸️ Model will load on first request")

    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {sd_server.device}")
    print(f"   • Offline Mode: {offline}")
    if torch.cuda.is_available():
        print(f"   • BF16 Enabled: {torch.cuda.is_bf16_supported()}")
        print(f"   • Optimizations: xFormers, Channels-Last, TF32, DPM++ Scheduler")

    print("\n✅ Server ready! Connect at http://127.0.0.1:5000")
    print("🎨 Aseprite plugin can now connect and generate!")
    print("\n" + "=" * 60 + "\n")

    try:
        cli = sys.modules.get("flask.cli")
        if cli:
            cli.show_server_banner = lambda *x: None

        app.run(
            host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")
