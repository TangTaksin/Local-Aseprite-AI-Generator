#!/usr/bin/env python3
"""
เซิร์ฟเวอร์ Local AI Generator สำหรับ Aseprite - Optimized for RTX 5070 Ti
เวอร์ชัน 2.4 - LoRA Caching, PNG Payload, Thread Lock, CUDA Cache Clear
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
import threading
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
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ปิดการแจ้งเตือนคำเตือนต่างๆ เพื่อให้ Console สะอาด
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")

diffusers_logging.set_verbosity_error()
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)


class PixelArtSDServer:
    def __init__(self, default_model=None):
        self.pipeline = None
        self.segmentation_model = None
        self.segmentation_processor = None
        self.model_loaded = False
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = {}
        self.offline_mode = False
        self.default_model = default_model
        self._current_lora = None
        self._current_lora_strength = None
        self._generation_lock = threading.Lock()
        is_windows = platform.system() == "Windows"
        self.use_compile = False if is_windows else True

        self.optimized_settings = {
            "use_bf16": torch.cuda.is_bf16_supported(),
            "use_compile": self.use_compile,
            "enable_xformers": not is_windows,
            "enable_attention_slicing": True,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",
        }

        self.default_settings = {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels",
        }

        print(f"🚀 Local AI Generator Server v2.4 [Optimized Mode]")
        print(f"📱 อุปกรณ์ที่ใช้: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"💾 VRAM: {props.total_memory / 1024**3:.1f}GB")
            print(f"⚡ Compute Capability: {props.major}.{props.minor}")
            print(f"🔷 BF16 Support: {torch.cuda.is_bf16_supported()}")
            self._apply_global_cuda_optimizations()

    def _apply_global_cuda_optimizations(self):
        if self.device != "cuda":
            return
        print("\n🔧 Applying RTX 5070 Ti Global Optimizations...")
        if self.optimized_settings["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision(
                self.optimized_settings["float32_matmul_precision"]
            )
        except:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   ✅ Global CUDA optimizations applied!\n")

    def _optimize_pipeline_for_blackwell(self, pipeline, model_type="sdxl"):
        if self.device != "cuda":
            return pipeline
        print(f"🎯 Optimizing {model_type.upper()} pipeline for RTX 5070 Ti...")
        dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
        pipeline = pipeline.to(self.device, dtype=dtype)
        try:
            if hasattr(pipeline, "disable_attention_slicing"):
                pipeline.disable_attention_slicing()
        except:
            pass
        try:
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
        except:
            pass
        if self.optimized_settings["use_compile"]:
            print("   🚀 Compiling U-Net with torch.compile...")
            pipeline.unet = torch.compile(
                pipeline.unet, mode="max-autotune", fullgraph=False, dynamic=False
            )
        if model_type == "sdxl" and hasattr(pipeline, "scheduler"):
            try:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
            except:
                pass
        print("   ✅ Pipeline optimization complete!\n")
        return pipeline

    def load_segmentation_model(self):
        if self.segmentation_model and self.segmentation_processor:
            return True
        print("📦 Loading BiRefNet for background removal...")
        try:
            model_name = "zhengpeng7/BiRefNet"
            self.segmentation_processor = transforms.Compose(
                [
                    transforms.Resize(
                        (1024, 1024),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode,
                torch_dtype=dtype,
            )

            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()

            if self.device == "cuda":
                try:
                    self.segmentation_model.to(memory_format=torch.channels_last)
                except:
                    pass

            if self.optimized_settings["use_compile"]:
                print("🚀 Compiling BiRefNet with torch.compile...")
                self.segmentation_model = torch.compile(
                    self.segmentation_model, mode="reduce-overhead"
                )
            print("✅ BiRefNet loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading BiRefNet: {e}")
            return False

    def remove_background(self, pil_image):
        if not self.load_segmentation_model():
            raise Exception("ไม่สามารถโหลดโมเดลลบพื้นหลังได้")
        print("🎭 Removing background with BiRefNet...")
        try:
            with torch.inference_mode():
                rgb_image = pil_image.convert("RGB")
                input_tensor = (
                    self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)
                )
                model_dtype = next(self.segmentation_model.parameters()).dtype
                input_tensor = input_tensor.to(dtype=model_dtype)
                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]
                mask = F.interpolate(
                    logits,
                    size=pil_image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.sigmoid(mask).squeeze()
                binary_mask = (mask > 0.4).cpu().numpy().astype(np.uint8)
            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            # mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=1))
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)
            print("✅ Background removal complete")
            return rgba_image
        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            return pil_image.convert("RGBA")

    def image_to_base64(self, image, format="PNG"):
        """
        PNG ดีกว่า raw bytes สำหรับ pixel art เพราะ:
        - Lossless compression
        - 64x64 pixel art PNG มักอยู่ที่ 1-3KB แทน 16KB raw
        """
        if image.mode not in ("RGBA", "RGB"):
            image = image.convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format=format, optimize=True, compress_level=6)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        try:
            local_only = self.offline_mode
            if model_name in self.model_cache:
                print(f"⚡ Loading {model_name} from cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True

            print(f"📥 Loading model: {model_name}")
            is_sdxl = "xl" in model_name.lower()
            precision = (
                torch.bfloat16
                if (self.device == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            local_model_path = os.path.join("models", model_name)
            is_local_file = os.path.isfile(local_model_path)

            if is_local_file:
                print(f"📂 Found local model: {local_model_path}")
                if is_sdxl:
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
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        local_model_path,
                        torch_dtype=precision,
                        load_safety_checker=False,
                    )
            else:
                if is_sdxl:
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
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=precision,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )

            model_type = "sdxl" if is_sdxl else "sd15"
            self.pipeline = self._optimize_pipeline_for_blackwell(
                self.pipeline, model_type
            )

            if self.device == "cuda":
                print("🔥 Running warmup inference...")
                try:
                    with torch.inference_mode():
                        _ = self.pipeline(
                            prompt="pixel",
                            width=512 if not is_sdxl else 1024,
                            height=512 if not is_sdxl else 1024,
                            num_inference_steps=3,
                            guidance_scale=1.0,
                        )
                    print("✅ Warmup complete - Ready for fast inference!")
                except Exception as e:
                    print(f"⚠️ Warmup skipped: {e}")

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

    def _switch_lora(self, lora_model, lora_strength):
        needs_load = (
            lora_model != self._current_lora
            or lora_strength != self._current_lora_strength
        )
        if not needs_load:
            return

        if self._current_lora and hasattr(self.pipeline, "unload_lora_weights"):
            self.pipeline.unload_lora_weights()

        if lora_model and str(lora_model).lower() not in ["none", "", "null"]:
            full_path = (
                lora_model
                if os.path.exists(lora_model)
                else os.path.join("loras", lora_model)
            )
            if os.path.exists(full_path):
                lora_dir, weight_name = os.path.split(full_path)
                self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
            else:
                self.pipeline.load_lora_weights(lora_model)
            print(f"✅ LoRA switched: {lora_model} @ {lora_strength}")
        else:
            print("⚡ LoRA cleared")

        self._current_lora = lora_model
        self._current_lora_strength = lora_strength

    def generate_image(self, prompt, lora_model=None, lora_strength=1.0, **kwargs):
        if not self.model_loaded:
            raise Exception("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")

        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("Server busy")

        try:
            print(f"🎨 Generating: '{prompt[:50]}...'")
            self._switch_lora(lora_model, lora_strength)

            pipeline_kwargs = {}
            if lora_model and str(lora_model).lower() not in ["none", "", "null"]:
                pipeline_kwargs["lora_scale"] = float(lora_strength)

            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)

            if "width" not in kwargs or "height" not in kwargs:
                if "xl" in self.current_model.lower():
                    gen_params.setdefault("width", 1024)
                    gen_params.setdefault("height", 1024)
                else:
                    gen_params.setdefault("width", 512)
                    gen_params.setdefault("height", 512)

            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]

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

            with torch.inference_mode():
                result = self.pipeline(**pipeline_kwargs)

            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()
        finally:
            self._generation_lock.release()

    def process_for_pixel_art(
        self,
        image: Image.Image,
        target_size: tuple[int, int] = (64, 64),
        colors: int = 16,
        use_dithering: bool = False,
        alpha_threshold: int = 128,
        enhance_contrast: float = 1.0,
        sharpen_amount: float = 2.0,
    ) -> Image.Image:
        print(f"🖼️ กำลังแปลงเป็น Pixel Art: ขนาด {target_size}, จำนวนสี {colors}")
        if colors < 2:
            raise ValueError("❌ จำนวนสีต้องมากกว่าหรือเท่ากับ 2")
        alpha = None
        if image.mode in ("RGBA", "LA", "PA", "P"):
            try:
                temp_image = image.convert("RGBA")
                alpha = temp_image.getchannel("A")
                alpha = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
                image = temp_image.convert("RGB")
            except Exception as e:
                print(f"⚠️ Warning processing alpha: {e}")
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        if enhance_contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(enhance_contrast)
        if sharpen_amount > 0:
            image = image.filter(
                ImageFilter.UnsharpMask(
                    radius=1, percent=int(100 * sharpen_amount), threshold=2
                )
            )
        image = image.resize(target_size, Image.NEAREST)
        if colors > 0:
            dither_mode = Image.FLOYDSTEINBERG if use_dithering else Image.NONE
            image = image.quantize(
                colors=colors, method=Image.MEDIANCUT, dither=dither_mode
            ).convert("RGB")
        if alpha is not None:
            alpha = alpha.resize(target_size, Image.NEAREST)
            alpha = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
            image = image.convert("RGBA")
            image.putalpha(alpha)
        print("✅ ประมวลผล Pixel Art สำเร็จ (คมชัด + สีสันสดใส)")
        return image


sd_server = PixelArtSDServer()


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400

        print(f"\n🎯 New generation request: {prompt[:30]}...")

        if not sd_server.model_loaded:
            print("⏳ Model not loaded yet. Loading now...")
            model_to_load = (
                data.get("model_name")
                or sd_server.default_model
                or "stabilityai/stable-diffusion-xl-base-1.0"
            )
            if not sd_server.load_model(model_to_load):
                return (
                    jsonify(
                        {"success": False, "error": "Failed to load model on demand"}
                    ),
                    500,
                )

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

        start_time = time.time()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)

        if data.get("remove_background", False):
            image = sd_server.remove_background(image)

        pixel_width = int(data.get("pixel_width", 64))
        pixel_height = int(data.get("pixel_height", 64))
        colors = int(data.get("colors", 16))
        pixel_image = sd_server.process_for_pixel_art(
            image, target_size=(pixel_width, pixel_height), colors=colors
        )

        img_base64 = sd_server.image_to_base64(pixel_image)
        generation_time = time.time() - start_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"⏱️ Total time: {generation_time:.2f}s")
        return jsonify(
            {
                "success": True,
                "image": {
                    "base64": img_base64,
                    "width": pixel_width,
                    "height": pixel_height,
                    "mode": "png",
                },
                "seed": used_seed,
                "prompt": prompt,
                "generation_time": generation_time,
            }
        )

    except RuntimeError as e:
        if "Server busy" in str(e):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Server is generating another image. Please retry.",
                    }
                ),
                503,
            )


@app.route("/health", methods=["GET"])
def health_check():
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
            "version": "2.4.0-rtx5070ti-optimized",
        }
    )


@app.route("/load_model", methods=["POST"])
def load_model_route():
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
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]
    model_directory = "models"
    os.makedirs(model_directory, exist_ok=True)
    if os.path.exists(model_directory):
        for filename in os.listdir(model_directory):
            if filename.endswith(".safetensors") or filename.endswith(".ckpt"):
                if filename not in models:
                    models.append(filename)
    return jsonify({"models": models})


@app.route("/loras", methods=["GET"])
def list_loras():
    lora_models = ["None", "nerijs/pixel-art-xl", "ntc-ai/SDXL-LoRA-slider.pixel-art"]
    lora_directory = "loras"
    os.makedirs(lora_directory, exist_ok=True)
    if os.path.exists(lora_directory):
        for filename in os.listdir(lora_directory):
            if filename.endswith(".safetensors"):
                if filename not in lora_models:
                    lora_models.append(filename)
    return jsonify({"loras": lora_models})


def main(default_model_to_load=None, offline=False):
    print("\n" + "=" * 60)
    print("🎮 LOCAL AI GENERATOR SERVER v2.4 [OPTIMIZED]")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)
    sd_server.offline_mode = offline
    sd_server.default_model = default_model_to_load
    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {sd_server.device}")
    print(f"   • Offline Mode: {offline}")
    print(f"   • Default Model: {default_model_to_load} (Lazy Load)")
    if torch.cuda.is_available():
        print(f"   • BF16 Enabled: {torch.cuda.is_bf16_supported()}")
        print(f"   • Optimizations: LoRA Cache, Thread Lock, PNG Payload")
    print("\n✅ Server ready! Connect at http://127.0.0.1:5000")
    print("🎨 Aseprite plugin can now connect and generate!")
    print("\n" + "=" * 60 + "\n")
    try:
        cli = sys.modules.get("flask.cli")
        if cli:
            cli.show_server_banner = lambda *x: None
        app.run(
            host="127.0.0.1", port=5000, debug=False, threaded=False, use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")
