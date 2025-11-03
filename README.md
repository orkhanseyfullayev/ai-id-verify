# AI-Based Identity Verification (Liveness, Face Similarity, and OCR System)

An advanced multi-stage identity verification system integrating **liveness detection**, **face similarity analysis**, and **optical character recognition (OCR)**.  
The system combines deep learning and image processing techniques to verify user identity based on live facial input and ID document analysis.

---

## 🧠 Overview

The system performs three sequential processes:

1. **Liveness Detection** – Ensures that the user is a real, live person using **6DRepNet** head-pose estimation (commands: turn right, turn left).  
2. **Face Similarity Analysis** – Compares the face from the webcam with the photo on the ID using **InsightFace (ArcFace embeddings)**.  
3. **OCR and Text Detection** – Extracts and analyzes text fields (Name, Surname) from the ID using **CRAFT** and **Tesseract OCR**.

Each step is fully automated and outputs JSON results for integration with higher-level authentication systems.

---

## ⚙️ System Architecture

main.py
├─▶ main_liveness_similarity.py → Liveness and facial similarity detection
├─▶ main_ocr.py → Runs OCR pipeline if verification succeeds
│ ├─ text_detection.py → Text region detection using CRAFT
│ └─ ocr.py → Tesseract OCR + text extraction
└─▶ result/
├─ liveness_result.json → Liveness & similarity output
├─ ocr_data.json → OCR parsed text
├─ crops/ → Text crops from CRAFT
└─ res_kimlik.txt → Text coordinates
