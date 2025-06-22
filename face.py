import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# 1. Initialize InsightFace
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' if you have GPU
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Build Known Embeddings Database
def build_database(player_img_dir):
    db = {}
    for player in os.listdir(player_img_dir):
        player_dir = os.path.join(player_img_dir, player)
        if not os.path.isdir(player_dir):
            continue
        embeddings = []
        for img_file in os.listdir(player_dir):
            img_path = os.path.join(player_dir, img_file)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if len(faces) == 1:  # Use only images with one face
                embeddings.append(faces[0].embedding)
        if embeddings:
            db[player] = embeddings
    return db

# 3. Recognize Faces in a New Image
def recognize_faces(img_path, db, threshold=0.5):
    img = cv2.imread(img_path)
    faces = app.get(img)
    results = []
    for face in faces:
        best_match = None
        best_score = 1.0  # cosine distance, lower is better
        for player, embeddings in db.items():
            for emb in embeddings:
                score = cosine(face.embedding, emb)
                if score < best_score:
                    best_score = score
                    best_match = player
        if best_score < threshold:
            results.append((face.bbox, best_match, 1 - best_score))
        else:
            results.append((face.bbox, 'Unknown', 1 - best_score))
    return results

# 4. Example Usage
if __name__ == '__main__':
    # Directory structure: player_img_dir/LeBron_James/img1.jpg, img2.jpg, ...
    player_img_dir = 'nba_player_images'
    db = build_database(player_img_dir)
    print(f"Database built for {len(db)} players.")

    # Test on a new image
    test_img = 'test_nba_photo.jpg'
    results = recognize_faces(test_img, db, threshold=0.5)
    img = cv2.imread(test_img)
    for bbox, name, conf in results:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f'{name} ({conf:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite('output_face_recognition.jpg', img)
    print("Results saved to output_face_recognition.jpg")