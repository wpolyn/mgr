import cv2

from inits3 import initialize_camera, text_properties, horizontal_image_processing, vertical_image_processing
from inits_registry import load_registry, invert_registry
from MLmodels import initialize_detector, initialize_recognizer

cap, img_src = initialize_camera()
detector, detector_name = initialize_detector()
recognizer, recognizer_name = initialize_recognizer()
org, fontFace, fontScale, color, thickness = text_properties()
label_to_id = load_registry()
id_to_label = invert_registry(label_to_id)

recognizer.read(img_src+'/model.yml')

while (cap.isOpened()):
    ret, image = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if ret:
        image, gray, faces = horizontal_image_processing(image, detector)
        #detect facial features and predict
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 80:  #if confidence less than 80, face detected and recognized
                cv2.putText(image, id_to_label[img_id], org, fontFace, fontScale, color, thickness)
            else:
                img_id = "unknown" 
                cv2.putText(image, str(img_id), org, fontFace, fontScale, color, thickness)
        #disable for vertical
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)        
        cv2.imshow('frame', image)
        
        # Press button b to exit
        if key & 0xFF == ord('b'):  
            break

cap.release()
cv2.destroyAllWindows()