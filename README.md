# 🍾 Bottle Detection using YOLOv8

**Real‑time bottle detection in images/video using a custom‑trained YOLOv8 model.**  
This project demonstrates a complete computer vision pipeline: dataset collection → annotation → training → inference.

---

## ✨ Features

- 🧠 **Custom YOLOv8 training** on a dataset of bottles (different shapes, colors, backgrounds)
- 📸 **Image & video inference** – detect bottles with bounding boxes and confidence scores
- 📊 **Evaluation metrics** – mAP@0.5, precision, recall
- 🖥️ **Simple inference script** – run on any image or video file
- 🧪 **Tested on real‑world scenes** (tabletop, shelves, fridge, etc.)

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|----------------|------------------------------------|
| Object Detection | YOLOv8 (Ultralytics)              |
| Image Processing | OpenCV                            |
| Annotation      | Roboflow / LabelImg                |
| Training        | Google Colab (GPU)                 |
| Language        | Python 3.11                        |

---

## 📁 Dataset

- **Total images:** 150+ (bottles of various sizes, colours, orientations)
- **Classes:** `bottle` (single class)
- **Annotation format:** YOLO (txt files with normalized coordinates)
- **Data split:** Train / Val / Test = 80% / 10% / 10%

> *Note:* The dataset is not included in this repository, but the annotation files and training script are provided.

---

## 🚀 Training

The model was trained on Google Colab with the following parameters:

```yaml
# training config
model: yolov8n.pt
epochs: 50
imgsz: 640
batch: 16
patience: 10
data: dataset.yaml
```

Training logs and the final weights (best.pt) are available in the runs/ folder.

---

🧪 Inference

On an Image

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('test_bottle.jpg')
results[0].show()
```

On a Video

```python
model = YOLO('best.pt')
results = model('test_video.mp4', save=True)
```

---

📊 Results

· mAP@0.5: 92%
· Precision: 0.89
· Recall: 0.91

Sample Input Detection Output
samples/bottle_original.jpg samples/bottle_detected.jpg

Example: Bottle detected with confidence 0.94

---

📁 Repository Structure

```
bottle-detection/
├── dataset.yaml           # dataset configuration
├── train.py               # training script (Colab compatible)
├── detect.py              # inference on image/video
├── best.pt                # trained model weights
├── samples/               # test images
├── runs/                  # training logs and results
├── requirements.txt       # dependencies
└── README.md              # this file
```

---

🔧 How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/your-username/bottle-detection.git
cd bottle-detection
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Run detection on an image

```bash
python detect.py --source sample.jpg
```

---

📦 Requirements

```
ultralytics
opencv-python
numpy
matplotlib
```

---

📈 Future Improvements

· Deploy as a Streamlit web app (similar to ZoneSentinel)
· Add real‑time webcam detection
· Extend to multiple bottle classes (e.g., plastic vs glass)

---

🙏 Acknowledgements

· Ultralytics YOLOv8
· Roboflow for dataset management

