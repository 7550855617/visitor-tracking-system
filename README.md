#  Real-Time Unique Visitor Detection System

##  Overview

This project is a real-time **visitor detection system** that:
- Detects people using **YOLOv8**
- Detects faces using **MTCNN**
- Generates embeddings using **FaceNet**
- Compares visitors using **cosine similarity**
- Logs new/unique visitors by storing:
  - Their face image
  - A timestamp
  - A unique embedding (persisted)
  - An entry in a **SQLite** database

It ensures the same person is not logged twice unless their appearance is significantly different.

---

## Project Pipeline

1. **YOLOv8** detects people from webcam frames.
2. Each person crop is passed to **MTCNN** to detect faces.
3. Detected faces are encoded using **FaceNet** into 512-dimensional embeddings.
4. Embeddings are compared with saved embeddings using **cosine similarity**.
5. If a face is new (similarity < 0.7), it's:
   - Saved to the `logs/` folder
   - Logged into the `visitor_log.db` SQLite database
   - Embedded vector is stored in `known_embeddings.pkl` for future comparison

---

## 📁 Folder Structure

```
.
├── visitor_log.db           # SQLite database to log visitors
├── known_embeddings.pkl     # Stored embeddings for previously seen visitors
├── logs/                    # Folder to store cropped face images
├── visitor.py               # Main script
├── requirements.txt         # Required Python packages
└── README.md                # This file
```

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
opencv-python
numpy
torch
facenet-pytorch
scikit-learn
Pillow
ultralytics
```

> If you have a CUDA-enabled GPU, install the appropriate PyTorch version from [https://pytorch.org](https://pytorch.org) for better performance.

---

## ▶️ How to Run

```bash
python visitor.py
```

**Controls:**
- Press `Q` to exit the webcam window.

---

## 🗃️ What Gets Stored

- **Face images** are saved in `logs/visitor_<timestamp>.jpg`
- **Database entries** are stored in `visitor_log.db` with:
  - `id` (auto-increment)
  - `timestamp` (when seen)
  - `image_path` (path to the saved image)
  - `exit_time`
- **(Face embeddings,id)** are saved in `known_embeddings.pkl`

---

## 📊 How to View the Visitor Database

You can use any of the following:

### ✅ Python script:

```python
import sqlite3
conn = sqlite3.connect("visitor_log.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM visitors")
for row in cursor.fetchall():
    print(row)
conn.close()
```

### ✅ SQLite CLI:

```bash
sqlite3 visitor_log.db
sqlite> SELECT * FROM visitors;
```

### ✅ GUI Tool:
Use **[DB Browser for SQLite](https://sqlitebrowser.org/)** to view, filter, and export the database.

---

## 🧪 Customization Ideas

- Add face **names or labels**
- Track **visit frequency** by storing a unique visitor ID
- Use **FAISS** for fast large-scale face search
- Add **real-time visitor count** overlay
- Enable **CSV export** of visitor logs

---

## 🙋 Author

**Developed by:** Suman Acharya  
If you found this helpful or want to collaborate, feel free to connect!


This project is a part of a hackathon run by https://katomaran.com 