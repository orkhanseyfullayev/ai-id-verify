import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('1.jpg')
text = pytesseract.image_to_string(image)
print("Detected text:", text)