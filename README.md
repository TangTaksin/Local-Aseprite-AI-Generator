## 🎨 Local Aseprite AI Generator (LAAG)
### **Stop painting every pixel. Start commanding them.**

**Local Aseprite AI Generator** คือสะพานเชื่อมระหว่างพลังของ Generative AI (Stable Diffusion) เข้ากับเครื่องมือวาด Pixel Art ที่ดีที่สุดในโลกอย่าง Aseprite โปรแกรมนี้จะเปลี่ยนข้อความของคุณให้กลายเป็นสไปรท์ที่พร้อมใช้งานในเกมภายในไม่กี่วินาที โดยประมวลผลผ่าน GPU ในเครื่องของคุณ 100%

---

### 🌟 Key Highlights
* **Local & Private:** ไม่มีการส่งข้อมูลไป Cloud ความลับของโปรเจกต์คุณจะอยู่แต่ในเครื่องของคุณเท่านั้น
* **Professional Pixel Quantization:** อัลกอริทึม Median Cut และ Nearest Neighbor ที่คัดกรองสีและพิกเซลให้คมกริบ ไม่เบลอเหมือน AI ทั่วไป
* **State-of-the-art Background Removal:** มาพร้อมกับ **BiRefNet** ที่สามารถลบพื้นหลังที่ซับซ้อนออกได้อัตโนมัติ ให้คุณได้สไปรท์โปร่งใสทันที
* **LoRA Support:** ปรับแต่งสไตล์ให้เข้ากับเกมของคุณ ไม่ว่าจะเป็น 8-bit, 16-bit, หรือ Isometric
* **SDXL & SD 1.5 Ready:** รองรับโมเดลมาตรฐานอุตสาหกรรมทั้งหมด

---

### 🛠 How it Works
1.  **Run the Server:** เริ่มต้นการทำงานด้วย Python Flask Server ที่รองรับการเร่งความเร็วผ่าน CUDA
2.  **Prompt & Generate:** ส่งคำสั่งผ่าน Aseprite Plugin (Lua Script)
3.  **Automatic Processing:** Server จะทำการสร้างภาพ -> ลบพื้นหลัง -> ปรับขนาด (Downscale) -> จัดการจำนวนสี (Quantize)
4.  **Instant Import:** ภาพจะปรากฏใน Canvas ของ Aseprite พร้อมให้คุณวาดต่อได้ทันที!

---

### 📊 Technical Specifications
| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.10+ / Flask |
| **AI Engine** | Diffusers (SDXL / SD 1.5) |
| **Segmentation** | BiRefNet (zhengpeng7) |
| **Processing** | PyTorch / Torchvision / PIL |
| **Interface** | REST API (JSON / Base64) |

---

### 🚀 Getting Started

1.  **Clone this repo:**
    ```bash
    git clone https://github.com/your-username/local-aseprite-ai-generator.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Engine:**
    ```bash
    python server.py
    ```

---

### 🛡 License
MIT License

Copyright (c) 2026 [Daggoot]
Based on PixelAI by Red335 (https://red335.itch.io/pixelai-local-ai-directly-in-aseprite)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------------------------------
CREDITS & ACKNOWLEDGMENTS:
This project is a modified and enhanced version of "PixelAI - Local AI directly in Aseprite" 
originally created by Red335. 

Original Tool: https://red335.itch.io/pixelai-local-ai-directly-in-aseprite
All rights to the original concept and initial implementation belong to the original author.

---