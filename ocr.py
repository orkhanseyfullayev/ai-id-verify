import cv2
import pytesseract
import os
import re
import json

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory and file paths
crop_folder = './result/crops'
coord_file = './result/res_kimlik.txt'

# Read coordinates sequentially (filter out empty lines)
with open(coord_file, 'r') as f:
    coord_lines = [line.strip() for line in f if line.strip()]

# Get files in order
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

file_list = sorted(
    [f for f in os.listdir(crop_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=extract_number
)

# Optional: print number of coordinates and images
print(f"Number of coordinates: {len(coord_lines)}")
print(f"Number of images: {len(file_list)}")

# OCR data will be collected here
ocr_data = []

# Perform OCR for each cropped image + coordinate + filename
for i, filename in enumerate(file_list):
    path = os.path.join(crop_folder, filename)
    image = cv2.imread(path)

    # Process the image
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR result (supports English + Turkish + Russian, PSM 6)
    text = pytesseract.image_to_string(thresh, lang='tur+eng+rus', config='--psm 6').strip()

    # Coordinate (safe by line index)
    coord = coord_lines[i] if i < len(coord_lines) else "?"

    # Add to list
    ocr_data.append({
        "filename": filename,
        "text": text,
        "coordinate": coord
    })

# Print results
print("\nOCR Data List:")
for item in ocr_data:
    print(f"{item['filename']} -> \"{item['text']}\" @ {item['coordinate']}")

# Save results as JSON
with open('./result/ocr_data.json', 'w', encoding='utf-8') as f:
    json.dump(ocr_data, f, ensure_ascii=False, indent=2)
