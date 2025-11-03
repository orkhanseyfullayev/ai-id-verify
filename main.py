import subprocess
import json
import os

print("\nStarting Liveness and Similarity checks...")
subprocess.run(["python", "main_liveness_similarity.py"])

# Read JSON result file
result_file = "liveness_result.json"
if not os.path.exists(result_file):
    print("Result file not found.")
    exit()

with open(result_file, "r", encoding="utf-8") as f:
    result = json.load(f)

liveness = result.get("liveness_result")
similarity = result.get("similarity_result")
score = result.get("score")  # Added field

# Check conditions
if liveness == "passed" and similarity == "matched":
    print("\nStarting OCR process...")
    subprocess.run(["python", "main_ocr.py"])
    print("\nLiveness Analysis Result: Successful")
    print(f"Face Similarity Result: Matched (score: %{score})")
    print()
    print()

elif liveness != "passed":
    print()
    print("\nLiveness Analysis Result: Failed (The process was not completed in time)")
    print("Similarity analysis was not performed because liveness verification failed.")
    print("OCR process was not executed.")
    print()
    print()

elif similarity == "not_detected":
    print()
    print("\nSimilarity Analysis Result: Face not detected.")
    print("OCR process was not executed.")
    print()
    print()

elif similarity == "suspicious":
    print()
    print("\nSimilarity Analysis Result: Suspicious (Manual review required. Try again with correct face angle.)")
    print("OCR process was not executed.")
    print()
    print()

else:
    print()
    print("\nSimilarity Analysis Result: Not matched.")
    print("OCR process was not executed.")
    print()
    print()
