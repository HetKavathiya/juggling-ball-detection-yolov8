# 🎯 Juggling Ball Detection using YOLOv8

This project focuses on detecting juggling balls in real-time using YOLOv8. The dataset is created from a custom video, frames are extracted using OpenCV, annotated with Roboflow, and used to train a high-accuracy object detection model.

---

## 🚀 Features

- Custom dataset created from video
- Frame extraction using OpenCV
- Annotation using Roboflow
- YOLOv8-based object detection
- Real-time ball detection
- Ball trajectory analysis (optional)

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Roboflow

---

## 📂 Dataset

- Extracted from a 1-minute juggling video
- ~600 annotated images
- Single class: `ball`

```bash
python run_analysis.py
