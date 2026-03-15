import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Dataset path
dataset_path = "dataset"

faces = []
labels = []
names = {}
label_id = 0

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Load dataset and prepare data
# -------------------------------
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    names[label_id] = person

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))

            faces.append(face.flatten())
            labels.append(label_id)

    label_id += 1

# Convert to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Check dataset
if len(faces) == 0:
    print("No faces found in dataset!")
    exit()

# -------------------------------
# Train Model
# -------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(faces, labels)

print("Model trained successfully!")

# -------------------------------
# Start Camera
# -------------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)

        prediction = model.predict(face)

        name = names[prediction[0]]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show name
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
