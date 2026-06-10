import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext('steel_plate_tag.png')
print("--- Extracted Text Nodes ---")
for idx, res in enumerate(results):
    print(f"{idx}: {res[1]} (conf: {res[2]})")
print("\n--- Joint Raw Text ---")
print(" ".join([res[1] for res in results]))
