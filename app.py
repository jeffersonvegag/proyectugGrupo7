import cv2
import face_recognition
import os
import numpy as np

known_faces_dir = "imgprueba.png"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.startswith("img") and (filename.endswith(".jpg") or filename.endswith(".png")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

test_image_path = "imgprueba.png""
test_image = face_recognition.load_image_file(test_image_path)
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)
test_image_cv = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "Desconocido"
    cv2.rectangle(test_image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(test_image_cv, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(test_image_cv, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

cv2.imshow('Test Image', test_image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
