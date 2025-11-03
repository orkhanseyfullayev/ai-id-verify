AI-Based Identity Verification (Liveness, Face Similarity, and OCR System)

An advanced multi-stage identity verification system integrating liveness detection, face similarity analysis, and optical character recognition (OCR).
The system combines deep learning and image processing methods to verify user identity based on live facial input and ID document analysis.

🧠 Overview

The system performs three sequential processes:

Liveness Detection – Ensures that the user is a real, live person using 6DRepNet head-pose estimation (commands: turn right, turn left).

Face Similarity Analysis – Compares the face from the webcam with the photo on the ID using InsightFace (ArcFace embeddings).

OCR and Text Detection – Extracts and analyzes text fields (Name, Surname) from the ID using CRAFT and Tesseract OCR.

Each step is fully automated and outputs JSON results for integration with higher-level authentication systems.

⚙️ System Architecture
main.py
 ├─▶ main_liveness_similarity.py      → Liveness and facial similarity detection
 ├─▶ main_ocr.py                      → Runs OCR pipeline if verification succeeds
 │    ├─ text_detection.py            → Text region detection using CRAFT
 │    └─ ocr.py                       → Tesseract OCR + text extraction
 └─▶ result/
      ├─ liveness_result.json         → Liveness & similarity output
      ├─ ocr_data.json                → OCR parsed text
      ├─ crops/                       → Text crops from CRAFT
      └─ res_kimlik.txt               → Text coordinates

🧩 Technologies

Computer Vision: OpenCV, InsightFace (ArcFace), 6DRepNet

Text Detection: CRAFT (Character Region Awareness for Text Detection)

OCR Engine: Tesseract (multi-language: English, Turkish, Russian)

Deep Learning Framework: PyTorch

Programming Language: Python 3.8+

📦 Project Structure
identity-verification/
├─ main.py
├─ main_liveness_similarity.py
├─ main_ocr.py
├─ text_detection.py
├─ ocr.py
│
├─ models/
│   ├─ 6DRepNet_300W_LP_AFLW2000.pth
│   └─ craft_mlt_25k.pth
│
├─ test_images/
│   └─ kimlik.jpg
│
├─ images_similarity/
│   ├─ face.png
│   └─ kimlik.png
│
└─ result/
    ├─ crops/
    ├─ ocr_data.json
    ├─ res_kimlik.txt
    └─ liveness_result.json

🧰 Installation
1️⃣ Clone the repository
git clone https://github.com/<username>/identity-verification.git
cd identity-verification

2️⃣ Create environment and install dependencies
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Install additional dependencies
opencv-python
numpy
torch
torchvision
insightface
sixdrepnet
pytesseract
scikit-image

4️⃣ Configure models

Place 6DRepNet_300W_LP_AFLW2000.pth in models/

Place craft_mlt_25k.pth in models/ or CRAFT-pytorch/

5️⃣ Install Tesseract OCR

For Windows: Tesseract Download (UB Mannheim)

Default path in ocr.py:

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


Mac/Linux users can simply use:

brew install tesseract

🚀 How to Run
Run the complete pipeline
python main.py


Steps performed:

Capture real-time face and ID images

Perform liveness detection (turn head right/left)

If passed, perform face similarity analysis

If matched, perform text detection + OCR on ID

📁 Output Files
File	Description
result/liveness_result.json	Stores liveness & similarity status
result/ocr_data.json	OCR-extracted text data
result/crops/	Individual cropped text regions
result/res_kimlik.txt	Bounding box coordinates for text
images_similarity/	Captured webcam and ID photos
🧪 Example JSON Output
{
  "liveness_result": "passed",
  "similarity_result": "matched",
  "score": 82
}

🔍 Thresholds and Configuration
Parameter	Default	Description
Liveness time limit	10s	Duration to complete head-pose commands
Similarity threshold	0.50	Above = matched, below = not matched
Suspicious range	0.40–0.50	Manual review required
Image resize	1400×900	Standardized for OCR accuracy
🧩 Troubleshooting
Issue	Possible Cause	Solution
Camera not opening	Device busy or wrong index	Try cv2.VideoCapture(1) or cv2.VideoCapture(2)
Face not detected	Poor lighting or low-resolution ID image	Improve lighting or distance
Tesseract not found	Incorrect installation path	Update tesseract_cmd path in ocr.py
CRAFT output empty	Model weights missing	Check craft_mlt_25k.pth path
Low OCR accuracy	Noisy input or shadows	Apply preprocessing (grayscale, thresholding)
🧾 License

Licensed under the MIT License
.

👤 Author

Developed by Orkhan Seyfullayev
For questions or collaboration: GitHub Profile
