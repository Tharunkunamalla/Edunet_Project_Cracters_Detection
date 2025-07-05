# 🌟Edunet_Project_Cracters_Detection
# Problem Statement:
Accurate detection and classification of lunar and planetary craters play a crucial role in surface age estimation, geological mapping, and landing site selection for space missions. Traditional manual crater detection is labor-intensive, time-consuming, and prone to human error. Automated methods are necessary for processing the vast amounts of high-resolution imagery captured by modern satellites and rovers.

This project aims to develop a deep learning-based crater detection system using the YOLOv5 (You Only Look Once) object detection framework. The model is trained to detect and classify craters in planetary surface images, including those from Mars and the Moon, with high confidence and spatial precision. The goal is to automate the identification of craters and generate annotated outputs for further scientific analysis, improving the efficiency and accuracy of crater detection workflows.
<hr>
<hr>

# 🛰️ YOLOv5 Crater Detection Pipeline (Google Colab)

## 📌 Objective
This guide helps you run YOLOv5 object detection (e.g. crater detection) in **Google Colab**, step-by-step, including model loading, image inference, and result visualization.
> Note : In this ive ignored `best.py` bcz it has more size().... so instead of best.pt, I have used Yolo's Pretranined model.... YOLOv5.pt

---

## ✅ Steps to Follow

### 🔹 Step 1: Clone the YOLOv5 GitHub Repository
```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
```
Clones the YOLOv5 repo and moves into it.

---

### 🔹 Step 2: Install Required Python Libraries
```bash
!pip install -r requirements.txt --quiet
```
Installs libraries required by YOLOv5.

---

### 🔹 Step 3: Upload Your Model & Test Images
```python
from google.colab import files
print("📤 Upload your best.pt")
model_upload = files.upload()

print("📤 Upload your test zip")
zip_upload = files.upload()
```
Upload:
- `best.pt` → your custom model (optional).
- `test.zip` → images to test.

> ⚠️ If you skip `best.pt`, YOLOv5 uses default `yolov5s.pt`.

---

### 🔹 Step 4: Extract Test Images
```python
import zipfile, os

zip_path = list(zip_upload.keys())[0]
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/test')
```
Unzips uploaded test images.

---

### 🔹 Step 5: Run YOLOv5 Inference
```bash
!python detect.py --weights /content/yolov5/{list(model_upload.keys())[0]}                   --source /content/test/test/images                   --conf 0.4                   --save-txt                   --save-conf
```
Flags:
- `--weights`: Your model path.
- `--source`: Input images.
- `--conf`: Confidence threshold.
- `--save-txt`, `--save-conf`: Save results.

> 🟡 Skip `--weights` to use default model.

---

### 🔹 Step 6: View a Single Detected Image
```python
import cv2
import matplotlib.pyplot as plt
import os

output_dir = sorted(os.listdir("runs/detect"))[-1]
output_img_path = f"runs/detect/{output_dir}/" + os.listdir(f"runs/detect/{output_dir}")[0]

img = cv2.imread(output_img_path)
if img is None:
    print("❌ Image not found or not saved correctly!")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Detected Craters")
    plt.show()
```
Displays the first result image.

---

### 🔹 Optional: View All Detected Images
```python
import os
import cv2
import matplotlib.pyplot as plt

output_base = "/content/yolov5/runs/detect"
exp_folder = sorted(os.listdir(output_base))[-1]
output_folder = os.path.join(output_base, exp_folder)

image_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.lower().endswith(('.jpg', '.png'))]

for img_path in image_paths:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Detected: {os.path.basename(img_path)}")
    plt.show()
```
Loops through and displays all result images.

---

## 🔚 Summary

| Step | Purpose |
|------|---------|
| Clone | Get YOLOv5 repo |
| Install | Set up dependencies |
| Upload | Add model and data |
| Extract | Unzip test images |
| Detect | Run object detection |
| Display | View outputs visually |

---

> Let me know if you want this as a Colab template too! ✊ shinzou wo sasageyo

<hr> 

# Conclusion: 
The implementation of the YOLOv5 object detection model successfully enabled the automated detection of craters in satellite imagery with high accuracy and confidence scores. By training on a curated dataset and running inference on unseen test images, the model was able to detect multiple craters per image, annotate them with bounding boxes, and output prediction labels and confidence values.

The visual and textual results confirm the model's effectiveness in identifying craters of various sizes and shapes. This solution demonstrates the capability of deep learning to enhance planetary surface analysis and reduce manual annotation efforts. Future improvements may include integrating PSR (Permanently Shadowed Regions), TSR (Temporarily Shadowed Regions), and Sunlight region classification, expanding the model's applicability to landing site planning and lunar habitat studies.
<hr>
<img src="edunet.png"><img src="microsoft.png">
