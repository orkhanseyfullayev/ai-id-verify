import subprocess
import json
import os

print("\nStarting text detection...")
subprocess.run(["python", "text_detection.py"])

print("\nStarting OCR analysis...")
subprocess.run(["python", "ocr.py"])

ocr_json_path = "./result/ocr_data.json"
with open(ocr_json_path, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

# Find the coordinate of 'Name' (exact match)
name_coordinate = None
for item in ocr_data:
    if item["text"].strip() == "Name":
        name_coordinate = item["coordinate"]
        break

# Find the coordinate of 'Surname' (case-insensitive)
surname_coordinate = None
for item in ocr_data:
    if item["text"].strip().lower() in ["surname", "lastname"]:
        surname_coordinate = item["coordinate"]
        break

# Find the value under 'Name'
name_value = None
if name_coordinate:
    name_coords = list(map(int, name_coordinate.split(",")))
    name_bottom_y = max(name_coords[1::2])
    for item in ocr_data:
        if item["coordinate"] == "?":
            continue
        coords = list(map(int, item["coordinate"].split(",")))
        top_y = min(coords[1::2])
        vertical_distance = top_y - name_bottom_y
        if 50 <= vertical_distance <= 100:
            name_value = item["text"]
            break

# Find the value under 'Surname'
surname_value = None
if surname_coordinate:
    surname_coords = list(map(int, surname_coordinate.split(",")))
    surname_bottom_y = max(surname_coords[1::2])
    for item in ocr_data:
        if item["coordinate"] == "?":
            continue
        coords = list(map(int, item["coordinate"].split(",")))
        top_y = min(coords[1::2])
        vertical_distance = top_y - surname_bottom_y
        if 40 <= vertical_distance <= 100:
            surname_value = item["text"]
            break

# Print results
if name_value:
    print()
    print(f"\nName: {name_value}")
if surname_value:
    print(f"Surname: {surname_value}")
