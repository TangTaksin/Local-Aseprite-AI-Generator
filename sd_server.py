#!/usr/bin/env python3
"""
Local AI Generator Server for Aseprite
Professional pixel art generation using Stable Diffusion with LoRA support and BiRefNet background removal.
Version 2.0 - Enhanced and optimized for publishing
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")

# Suppress Flask development server warning
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
        
        self.default_settings = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels"
        }
        
        print(f"🚀 Local AI Generator Server v2.0")
        print(f"📱 Device: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def load_segmentation_model(self):
        """Loads the BiRefNet model for professional background removal."""
        if self.segmentation_model and self.segmentation_processor:
            return True
            
        print("📦 Loading BiRefNet model for background removal...")
        
        try:
            model_name = 'zhengpeng7/BiRefNet'
            
            # การตั้งค่าความละเอียดภาพ
            self.segmentation_processor = transforms.Compose([
                transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # โหลดโมเดล
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            
            # ย้ายไป GPU และบังคับเป็น Float32 เพื่อแก้ปัญหา Dtype Mismatch
            self.segmentation_model.to(self.device)
            self.segmentation_model.float() 
            self.segmentation_model.eval()
            
            print("✅ BiRefNet model loaded successfully (Forced Float32)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading BiRefNet model: {e}")
            return False
        
        try:
            model_name = 'zhengpeng7/BiRefNet'
            
            # Create processor for image preprocessing
            self.segmentation_processor = transforms.Compose([
                transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # Load the segmentation model
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            self.segmentation_model.to(self.device)
            self.segmentation_model.float()
            self.segmentation_model.eval()
            
            print("✅ BiRefNet model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading BiRefNet model: {e}")
            return False

    def remove_background(self, pil_image):
        """Uses BiRefNet to create a high-quality transparency mask."""
        if not self.load_segmentation_model():
            raise Exception("Background removal model could not be loaded.")
            
        print("🎭 Removing background with BiRefNet...")
        try:
            with torch.no_grad():
                # Convert to RGB for processing
                rgb_image = pil_image.convert("RGB")
                
                # Preprocess image
                input_tensor = self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)

                # บังคับชนิดข้อมูลภาพให้ตรงกับ Model (ป้องกัน Error float/Half)
                input_tensor = input_tensor.to(dtype=next(self.segmentation_model.parameters()).dtype)
                
                # Generate mask
                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]
                
                # Resize mask to original image size
                mask = F.interpolate(logits, size=pil_image.size[::-1], mode='nearest')
                mask = torch.sigmoid(mask).squeeze()
                
                # Create binary mask with sharp edges for pixel art
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
                
            # Apply mask to create transparent background
            mask_image = Image.fromarray(binary_mask * 255, mode='L')
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)
            
            print("✅ Background removal complete")
            return rgba_image
            
        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            return pil_image.convert("RGBA")

    def image_to_base64(self, image):
        """Convert PIL image to base64 encoded bytes."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return base64.b64encode(image.tobytes()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Load and cache AI models for efficient generation."""
        try:
            local_only = self.offline_mode
            if local_only:
                print("🔒 Offline mode enabled: loading from cache only")
                
            # Check cache first
            if model_name in self.model_cache:
                print(f"⚡ Loading {model_name} from cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True
            
            print(f"📥 Loading base model: {model_name}")
            precision = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load appropriate pipeline based on model type
            if "xl" in model_name.lower():
                print("🔧 Loading SDXL pipeline with optimized VAE...")
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
                print("🔧 Loading SD 1.5 pipeline...")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=precision,
                    use_safetensors=True,
                    local_files_only=local_only
                )
            
            # Move to device and optimize
            self.pipeline = self.pipeline.to(self.device)
            
            # Cache the model for future use
            self.model_cache[model_name] = self.pipeline
            self.current_model = model_name
            self.model_loaded = True
            
            print(f"✅ Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {str(e)}")
            return False

    def generate_image(self, prompt, lora_model=None, lora_strength=1.0, **kwargs):
        """Generate images with LoRA style control and optimized settings."""
        if not self.model_loaded:
            raise Exception("No base model loaded. Please load a model first.")
            
        try:
            print(f"🎨 Generating image with prompt: '{prompt[:50]}...'")
            
            # Prepare pipeline arguments
            pipeline_kwargs = {}
            
            # Handle LoRA loading with strength control
            if lora_model and lora_model.lower() not in ['none', '']:
                print(f"🎭 Loading LoRA: {lora_model} (strength: {lora_strength})")
                
                if os.path.exists(lora_model):
                    # Local LoRA file
                    lora_path, weight_name = os.path.split(lora_model)
                    self.pipeline.load_lora_weights(lora_path, weight_name=weight_name)
                else:
                    # Hub LoRA
                    self.pipeline.load_lora_weights(lora_model)
                
                # Apply LoRA strength
                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
            
            # Merge default settings with provided kwargs
            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)
            
            # Use provided dimensions or set defaults based on model
            if "width" not in kwargs or "height" not in kwargs:
                if "xl" in self.current_model.lower():
                    gen_params.setdefault('width', 1024)
                    gen_params.setdefault('height', 1024)
                else:
                    gen_params.setdefault('width', 512)
                    gen_params.setdefault('height', 512)
            
            # Enhance prompt for pixel art if needed
            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]
            
            # Handle seed generation
            seed = gen_params.get("seed", -1)
            generator = torch.Generator(device=self.device)
            
            if seed is not None and int(seed) != -1:
                generator.manual_seed(int(seed))
                print(f"🎲 Using seed: {seed}")
            else:
                # Generate a truly random seed
                import random
                random_seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(random_seed)
                print(f"🎲 Using random seed: {random_seed}")
                seed = random_seed
            
            # Prepare final generation parameters
            pipeline_kwargs.update({
                "prompt": prompt,
                "negative_prompt": gen_params["negative_prompt"],
                "width": gen_params["width"],
                "height": gen_params["height"],
                "num_inference_steps": int(gen_params["num_inference_steps"]),
                "guidance_scale": float(gen_params["guidance_scale"]),
                "generator": generator
            })
            
            print(f"⚙️ Generation settings: {gen_params['width']}x{gen_params['height']}, {gen_params['num_inference_steps']} steps")
            
            # Generate the image
            result = self.pipeline(**pipeline_kwargs)
            
            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()
        
        finally:
            # Clean up LoRA to prevent conflicts
            if lora_model and lora_model.lower() not in ['none', ''] and hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()

    def process_for_pixel_art(self, image, target_size=(64, 64), colors=16):
        """Advanced pixel art post-processing with color quantization."""
        print(f"🖼️ Processing for pixel art: {target_size}, {colors} colors")
        
        # Resize with nearest neighbor for sharp pixels
        image = image.resize(target_size, Image.NEAREST)
        
        # Apply color quantization if specified
        if colors > 0:
            if image.mode == 'RGBA':
                # Preserve alpha channel during quantization
                alpha = image.getchannel('A')
                rgb_image = image.convert('RGB').quantize(
                    colors=int(colors) - 1,  # Reserve one color for transparency
                    method=Image.MEDIANCUT
                )
                image = rgb_image.convert('RGBA')
                image.putalpha(alpha)
            else:
                image = image.quantize(
                    colors=int(colors),
                    method=Image.MEDIANCUT
                ).convert('RGB')
        
        print("✅ Pixel art processing complete")
        return image

# Initialize the server instance
sd_server = PixelArtSDServer()

@app.route('/generate', methods=['POST'])
def generate():
    """Main generation endpoint with comprehensive error handling."""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"success": False, "error": "No prompt provided"}), 400
        
        print(f"\n🎯 New generation request: {prompt[:30]}...")
        
        # Extract parameters with defaults
        defaults = sd_server.default_settings
        kwargs = {
            "lora_model": data.get('lora_model'),
            "lora_strength": data.get('lora_strength', 1.0),
            "num_inference_steps": data.get('steps', defaults.get('num_inference_steps')),
            "guidance_scale": data.get('guidance_scale', defaults.get('guidance_scale')),
            "seed": data.get('seed', -1),
            "negative_prompt": data.get('negative_prompt', defaults.get('negative_prompt')),
            "width": data.get('width', 1024),   # Base generation resolution
            "height": data.get('height', 1024)  # Base generation resolution
        }
        
        # Generate the base image
        start_time = datetime.now()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)
        
        # Apply background removal if requested
        if data.get('remove_background', False):
            print("🎭 Applying background removal...")
            image = sd_server.remove_background(image)
        
        # Process for pixel art
        pixel_width = int(data.get('pixel_width', 64))
        pixel_height = int(data.get('pixel_height', 64))
        colors = int(data.get('colors', 16))
        
        pixel_image = sd_server.process_for_pixel_art(
            image,
            target_size=(pixel_width, pixel_height),
            colors=colors
        )
        
        # Convert to base64
        img_base64 = sd_server.image_to_base64(pixel_image)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ Total generation time: {generation_time:.2f}s")
        
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
        print(f"❌ Generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Server health and status endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": sd_server.model_loaded,
        "current_model": sd_server.current_model,
        "device": sd_server.device,
        "version": "2.0.0"
    })

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Load a specific AI model."""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({"success": False, "error": "No model_name provided"}), 400
        
        print(f"📦 Loading model: {model_name}")
        
        if sd_server.load_model(model_name):
            return jsonify({
                "success": True,
                "model": model_name,
                "device": sd_server.device
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to load {model_name}"
            }), 500
            
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available base models."""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5"
    ]
    return jsonify({"models": models})

@app.route('/loras', methods=['GET'])
def list_loras():
    """List available LoRA models (both hub and local)."""
    lora_models = [
        "None",
        "nerijs/pixel-art-xl",
        "ntc-ai/SDXL-LoRA-slider.pixel-art"
    ]
    
    # Add local LoRA files
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
    """Main server startup function."""
    print("\n" + "="*50)
    print("🎮 LOCAL AI GENERATOR SERVER v2.0")
    print("="*50)
    
    sd_server.offline_mode = offline
    
    if default_model_to_load and default_model_to_load.lower() != "none":
        print(f"📦 Loading default model: {default_model_to_load}")
        sd_server.load_model(default_model_to_load)
    else:
        print("⏸️ No model loaded on startup (will load on first request)")
    
    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {sd_server.device}")
    print(f"   • Offline Mode: {offline}")
    
    print("\n✅ Server ready! Access at http://127.0.0.1:5000")
    print("🎨 Aseprite plugin can now connect and generate images!")
    print("\n" + "="*50 + "\n")
    
    try:
        # Disable Flask development server warning
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")