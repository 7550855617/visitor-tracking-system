
# Real-Time Unique Visitor Detection System

##  Overview

This project is a real-time **visitor detection and logging system** using:

- **YOLOv8** – for person detection
- **MTCNN** – for face detection
- **FaceNet** – for face embedding generation
- **Cosine Similarity** – to compare faces
- **SQLite** – to log entries and exit times
- **Pickle** – to persist known face embeddings

It ensures the same person is not logged again unless their appearance is significantly different or they revisit later.

> **Hackathon Project**: This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)

---

## 🔁 Project Pipeline

1. YOLOv8 detects people from webcam frames.
2. Each detected person is cropped and passed to MTCNN.
3. MTCNN detects the face(s) from the crop.
4. FaceNet generates a 512-dimension embedding for the face.
5. Embeddings are compared with known ones using cosine similarity.
6. If similarity < threshold (e.g. 0.7), the face is considered new and logged.
7. If already known:
   - Either exit time is updated (if continuing visit)
   - Or a new entry is logged (if revisiting, based on design choice)

---

## Folder Structure

```
.
├── visitor_log.db           # SQLite database of visitor logs
├── known_embeddings.pkl     # Stored embeddings with IDs
├── logs/                    # Saved face images
├── config.json              # Runtime configuration
├── visitor.py               # Main application script
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
opencv-python
numpy
torch
facenet-pytorch
scikit-learn
Pillow
ultralytics
```

> ⚡ Tip: If using GPU, install the correct PyTorch version from [https://pytorch.org](https://pytorch.org).

---

## ▶️ How to Run

```bash
python visitor.py
```

**Controls**:
- Press `Q` to quit the webcam window.

---

## 🧾 What Gets Logged

- Cropped face image → `logs/visitor_<timestamp>.jpg`
- SQLite entry:
  - `id`
  - `timestamp` (entry time)
  - `image_path`
  - `exit_time`
- Face embeddings with ID → `known_embeddings.pkl`

---

## Sample `config.json` Structure

```json
{
  "skip_detection": false,
  "similarity_threshold": 0.7
}
```

---

## Assumptions Made

- One face per person per frame is enough to identify the visitor.
- Faces are reasonably clear for embedding to be consistent.
- Exit time is either updated or new visit is logged on reappearance.
- Visitors are logged only when a face is detected inside a person box.

---

## Architecture Diagram

```
               ┌──────────────────────────┐
               │     OpenCV Webcam        │
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │       YOLOv8             │ ← Person detection
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │         MTCNN            │ ← Face detection from person box
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │       FaceNet            │ ← 512-dim face embedding
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │  Cosine Similarity Check │ ← Compare to known faces
               └─────┬────────────┬───────┘
                     ↓            ↓
        ┌──────────────┐   ┌────────────────┐
        │ New Visitor  │   │ Known Visitor  │
        └─────┬────────┘   └────┬────────────┘
              ↓                ↓
    ┌────────────────┐   ┌────────────────────┐
    │ Save image to   │   │ Update exit_time or│
    │ logs/, DB, pkl  │   │ re-log new entry   │
    └────────────────┘   └────────────────────┘
```

---

## How to View Visitor Logs

### Python:
```python
import sqlite3
conn = sqlite3.connect("visitor_log.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM visitors")
for row in cursor.fetchall():
    print(row)
conn.close()
```

### SQLite CLI:
```bash
sqlite3 visitor_log.db
sqlite> SELECT * FROM visitors;
```

### 🧰 GUI:
Use **[DB Browser for SQLite](https://sqlitebrowser.org/)**.

---

## 🔧 Customization Ideas

- Add face **labels (names)** using manual tagging
- Calculate **visit duration** (`exit_time - timestamp`)
- Track **visit frequency per person**
- Integrate **CSV export** or email summary
- Add **sound/alert** for unknown person
- Deploy using **Streamlit** or **Flask** dashboard

---

## 👨‍💻 Author

**Developed by:** Suman Acharya  
**Hackathon:** [Katomaran Hackathon](https://katomaran.com)

## explanation video link : https://drive.google.com/file/d/1lJDqRuU6url9FgUnVDQDhNCZBDTYyCmY/view?usp=drive_link

Feel free to connect or contribute!
