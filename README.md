# AI-Based Identity Verification (Liveness, Face Similarity, and OCR System)

An advanced multi-stage identity verification system integrating **liveness detection**, **face similarity analysis**, and **optical character recognition (OCR)**.  
The system combines deep learning and image processing techniques to verify user identity based on live facial input and ID document analysis.

---

## Overview

The system performs three sequential processes:

1. **Liveness Detection** – Ensures that the user is a real, live person using **6DRepNet** head-pose estimation (commands: turn right, turn left).  
2. **Face Similarity Analysis** – Compares the face from the webcam with the photo on the ID using **InsightFace (ArcFace embeddings)**.  
3. **OCR and Text Detection** – Extracts and analyzes text fields (Name, Surname) from the ID using **CRAFT** and **Tesseract OCR**.

Each step is fully automated and outputs JSON results for integration with higher-level authentication systems.

---

## System Architecture

```
📂 identity-verification
├── main.py → Main controller that runs all modules
│
├── main_liveness_similarity.py → Liveness detection + facial similarity using 6DRepNet and InsightFace
├── main_ocr.py → OCR pipeline triggered if liveness & similarity succeed
│ ├── text_detection.py → Text region detection using CRAFT
│ └── ocr.py → OCR text extraction using Tesseract
│
├── models/
│ ├── 6DRepNet_300W_LP_AFLW2000.pth → Pretrained 6DRepNet weights
│ └── craft_mlt_25k.pth → CRAFT model weights
│
├── test_images/
│ └── kimlik.jpg → Example ID image
│
├── images_similarity/
│ ├── face.png → Captured live face
│ └── kimlik.png → Captured ID face
│
├── result/
│ ├── liveness_result.json → Liveness & similarity output
│ ├── ocr_data.json → OCR parsed text
│ ├── crops/ → Cropped text regions from ID
│ └── res_kimlik.txt → Text coordinates from CRAFT
│
└── face_recognition/ → InsightFace model files
```

---

## Technologies

- **Computer Vision:** OpenCV, InsightFace (ArcFace), 6DRepNet  
- **Text Detection:** CRAFT (Character Region Awareness for Text Detection)  
- **OCR Engine:** Tesseract (multi-language: English, Turkish, Russian)  
- **Deep Learning Framework:** PyTorch  
- **Programming Language:** Python 3.8+

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/orkhanseyfullayev/identity-verification.git
   cd identity-verification
2. **Create virtual environment and install dependencies**
   ```
   python -m venv venv
   source venv/bin/activate        # On macOS/Linux
   venv\Scripts\activate           # On Windows
   pip install -r requirements.txt
   ```
3. **Install additional dependencies (if missing)**
```
   pip install opencv-python numpy torch torchvision insightface sixdrepnet pytesseract scikit-image
```


---

## Running the System
1. **Run the complete verification pipeline:**
   ```
   python main.py
   ```
2. **The program will:**  
   • Capture a live face and an ID photo.  
   • Perform liveness detection and face similarity check.  
   • If verified, run text detection and OCR to extract name and surname.  
   • Save all outputs in the result/ directory.  


---

## Output Files

   • ```result/liveness_result.json``` → Liveness & face similarity results  
   • ```result/ocr_data.json``` → Extracted text data from OCR  
   • ```result/res_kimlik.txt``` → Raw coordinates of detected text boxes.  
   • ```result/crops/``` → Cropped text images.  











