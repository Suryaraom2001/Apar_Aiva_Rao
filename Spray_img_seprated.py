import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil

# Load YOLO model
model_path = r"D:\Apr inns\Surya_T2\Surya_T2\human_detection.pt"
model = YOLO(model_path)

# Folder paths
human_image_path = r"D:\Apr inns\Surya_T2\Surya_T2\output\human"
spray_image_path = r"D:\Apr inns\Surya_T2\Surya_T2\output\spray"
not_human_image_path = r"D:\Apr inns\Surya_T2\Surya_T2\output\not_human"    

os.makedirs(human_image_path, exist_ok=True)
os.makedirs(spray_image_path, exist_ok=True)
os.makedirs(not_human_image_path, exist_ok=True)

# Detect human using YOLO
def detect_human(image_path, model, confidence_threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f" Error reading: {image_path}")
        return False

    results = model.predict(
        source=img,
        classes=[0],  # Class 0 = person
        conf=confidence_threshold,
        verbose=False
    )

    for result in results:
        if len(result.boxes) > 0:
            return True
    return False

# Main folder
folder_path = r"D:\Apr inns\Surya_T2\Surya_T2"

# Process images
for folder_name in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder_name)

    if os.path.isdir(folder_full_path):
        print(f"\nScanning: {folder_name}")

        for image_file in os.listdir(folder_full_path):
            image_path = os.path.join(folder_full_path, image_file)

            if not os.path.isfile(image_path):
                continue

            print(f" Processing: {image_file}")

            try:
                # Check if "spray" is in the file name
                if "spray" in image_file.lower():
                    print(" → Spray image (by name)")
                    shutil.copy(image_path, os.path.join(spray_image_path, image_file))
                    continue

                # Check for human
                is_human = detect_human(image_path, model)

                if is_human:
                    print(" → Human detected")
                    shutil.copy(image_path, os.path.join(human_image_path, image_file))
                else:
                    print(" → No human detected")
                    shutil.copy(image_path, os.path.join(not_human_image_path, image_file))

            except Exception as e:
                print(f" Error processing {image_file}: {e}")
