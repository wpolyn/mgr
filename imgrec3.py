import cv2
import os
import pickle
import time

from inits import initialize_camera, image_processing, text_properties, load_registry, invert_registry
from models import initialize_detector, initialize_recognizer

cap, img_src = initialize_camera()
detector = initialize_detector()
recognizer = initialize_recognizer()
org, fontFace, fontScale, color, thickness = text_properties()
label_to_id = load_registry()
id_to_label = invert_registry(label_to_id)

recognizer.read(img_src+'/model.yml')

while (cap.isOpened()):
    ret, image = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if ret:
        image, gray, faces = image_processing(image, detector)
        #detect facial features and predict
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 80:  #if confidence less than 80, face detected and recognized
                cv2.putText(image, id_to_label[img_id], org, fontFace, fontScale, color, thickness)
            else:
                img_id = "unknown" 
                cv2.putText(image, str(img_id), org, fontFace, fontScale, color, thickness)
        cv2.imshow('frame', image)
        
        # Press button b to exit
        if key & 0xFF == ord('b'):  
            break

cap.release()
cv2.destroyAllWindows()