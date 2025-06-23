import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import datetime
import os
import pickle
from PIL import Image
from ultralytics import YOLO

#here we store the embeddings of faces and match with new person if came
EMBEDDING_FILE = "known_embeddings.pkl"
#here we store the faces if it already not in EMBEDDING_FILE
LOG_DIR = "logs"
#here we create a new database to store the timestamp and a unique id 
DB_FILE = "visitor_log.db"
#This is the simsilarity score we going to use if a person came into video whether that person came before or not 
SIMILARITY_THRESHOLD = 0.7


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#we load our yolov8 model for object dection 
yolo_model = YOLO("yolov8n.pt")
#we load our MTCNN model for face detection from yolov8 output
face_detector = MTCNN(keep_all=True, device=device)
#and from MTCNN output we create embedding using this pre_trained facenet model 
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


#sio we create the folder so save crop images from mtcnn if not already exist
os.makedirs(LOG_DIR, exist_ok=True)
#here create the database 
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
#so in database we create a table with two three columns one is id(unique),timestamp,image_path
cursor.execute('''
    CREATE TABLE IF NOT EXISTS visitors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT
    )
''')
conn.commit()


#Now here we import the embedding file  as known_embedding
if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "rb") as f:
        known_embeddings = pickle.load(f)
    print(f"[INFO] Loaded {len(known_embeddings)} known embeddings.")
else:
    known_embeddings = []

#SO THIS FUNCTION BASICALLY generate the embeddings of a person face 
def get_face_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))  # HWC → CHW
    face_tensor = torch.tensor(face).unsqueeze(0).to(torch.float32).to(device)
    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()[0]
    return embedding

#this function store the embedding if already not exist in known_emebdding file and also store in database 
def log_visitor(face_img, embedding):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/visitor_{timestamp}.jpg"
    cv2.imwrite(filename, face_img)

    # Save to DB
    cursor.execute("INSERT INTO visitors (timestamp, image_path) VALUES (?, ?)", (timestamp, filename))
    conn.commit()

    # Save to memory + file
    known_embeddings.append(embedding)
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(known_embeddings, f)

    print(f"[NEW VISITOR] Logged at {timestamp}")


#here we start the camera and getting frame by frame 
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #from here we get result from our yolov8 model 
    results = yolo_model(frame)
    for det in results[0].boxes.xyxy:
        #Now that result is basically the coordinates of the object detect by yolov8 such as center,height,length
        x1, y1, x2, y2 = map(int, det[:4])
        #so after getting the coordinates we crop the object(person) from the frame and give as input to MTCNN
        person_crop = frame[y1:y2, x1:x2]

        # Convert to PIL for MTCNN
        person_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        #so here we get two things from MTCNN one is boxes where face can be  and another is probabbilty score or confidance score 
        boxes, probs = face_detector.detect(person_pil)
        
        if boxes is not None:
            for box in boxes:
                #Now from that boxes we get the coordinates of faces and then we crop the face from the object 
                fx1, fy1, fx2, fy2 = map(int, box)
                face_crop = person_crop[fy1:fy2, fx1:fx2]

                if face_crop.size == 0:
                    continue
                #Now after getting the face we call our "get_face_embedding" function to get the embedding of that face 
                embedding = get_face_embedding(face_crop)
                #Now here we check if the embedding already exist in "known_embeddings" or not 
                if not known_embeddings:
                    log_visitor(face_crop, embedding)
                #if  similarity score less than 0.7 then append the embedding and store information in database and save crop image in log folder else not 
                else:
                    sims = cosine_similarity([embedding], known_embeddings)[0]
                    if max(sims) < SIMILARITY_THRESHOLD:
                        log_visitor(face_crop, embedding)
                    else:
                        print("[INFO] Known visitor — not logged again.")

                # Draw face box 
                cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (0, 255, 0), 2)

    cv2.imshow("Visitor Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
