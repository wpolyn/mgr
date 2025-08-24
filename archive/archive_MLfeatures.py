import cv2

from inits2 import initialize_camera, text_properties
from performance2 import performance
#from CNNmodels import initialize_recognizer
from MLmodels import initialize_detector
from landmarkcascades import get_landmarks
detector, detector_name = initialize_detector()
#recognizer, recognizer_name = initialize_recognizer()
monitoring = performance()
face_detector, eye_detector, nose_detector, smile_detector = get_landmarks()
display_size = (320, 240)
crop_size = 320
faces = None
reference_faces = []
face_counter = 0
threshold = 0.363
if __name__ == '__main__':
    cap, img_src = initialize_camera()
    org, fontFace, fontScale, color, thickness = text_properties()
    ready = False
    previous_faces = []

    #frame skipping settings
    frame_buffer = 2
    frame_counter = 0


    while (cap.isOpened()):
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        key = cv2.waitKey(1) & 0xFF
        h_image, w_image = image.shape[:2] #h=480 w=640
        #x_input = (w_image - crop_size) // 2 #160
        #y_input = (h_image - crop_size) // 2 #80
        
        if ret:
            #detection and recognition inside frame skipping if ready
            if ready and frame_counter % frame_buffer == 0:
                #model input processing and detection
                #feed = image[y_input:y_input + crop_size, x_input:x_input + crop_size] #y: 80<->400=320 x: 160<->480=320
                #feed = cv2.resize(feed, input_size)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)


                #recognition
                #previous_faces = []
                #if faces is not None:
                    #for face in faces:
                        #x, y, w, h = map(int, face[:4])
                        #cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                        #aligned_face = recognizer.alignCrop(feed, face)
                        #features_face = recognizer.feature(aligned_face)
                        
                        #best_match = 0
                        #compare current faces to reference faces
                        #for reference_face, reference_label in reference_faces:
                        #    match = recognizer.match(features_face, reference_face, cv2.FaceRecognizerSF_FR_COSINE)
                            #find the best match
                        #    if match > best_match:
                        #        best_match = match
                        #        best_label = reference_label
                        #check if the best match is over the threshold
                        #if best_match > threshold:
                        #    label = best_label
                        #    recognition = best_match
                        #else:
                            #under the threshold - add a new face to reference_faces
                        #    face_counter += 1
                        #    label = f"Coleg {face_counter}"
                        #    reference_faces.append((features_face, label))
                        #    recognition = best_match
                        #provide results for display
                        #previous_faces.append((face, recognition, label))
                frame_counter += 1
            else:
                frame_counter += 1

            #display processing
            display = cv2.resize(image, display_size)

            #displaying results if ready
            if ready and faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    features = gray[y:y + h, x:x + w]
                    width_alignment = (display_size[0] / w_image) #0.5
                    height_alignment = (display_size[1] / h_image) #0.5
                    eyes = eye_detector.detectMultiScale(features, 1.3, 5)
                    noses = nose_detector.detectMultiScale(features, 1.3, 5)
                    smiles = smile_detector.detectMultiScale(features, 1.3, 7)
                    
                    for eye in eyes[:2]:
                        x_eye, y_eye, w_eye, h_eye = map(int, eye[:4])
                        x_eyes = int((x_eye + x) * width_alignment)
                        y_eyes = int((y_eye + y) * height_alignment)
                        w_eyes = int((w_eye) * width_alignment)
                        h_eyes = int((h_eye) * height_alignment)
                        cv2.rectangle(display, (x_eyes, y_eyes), (x_eyes + w_eyes, y_eyes + h_eyes), (0, 255, 0), 2)
                    if len(noses) > 0:
                        x_nose, y_nose, w_nose, h_nose = map(int, noses[0])
                        x_noses = int((x_nose + x) * width_alignment)
                        y_noses = int((y_nose + y) * height_alignment)
                        w_noses = int((w_nose) * width_alignment)
                        h_noses = int((h_nose) * height_alignment)
                        cv2.rectangle(display, (x_noses, y_noses), (x_noses + w_noses, y_noses + h_noses), (0, 0, 255), 2)
                    if len(smiles) > 0:
                        x_smile, y_smile, w_smile, h_smile = map(int, smiles[0])
                        x_smiles = int((x_smile + x) * width_alignment)
                        y_smiles = int((y_smile + y) * height_alignment)
                        w_smiles = int((w_smile) * width_alignment)
                        h_smiles = int((h_smile) * height_alignment)
                        cv2.rectangle(display, (x_smiles, y_smiles), (x_smiles + w_smiles, y_smiles + h_smiles), (0, 255, 255), 2)
                    x = int(x * width_alignment)
                    y = int(y * height_alignment)
                    w = int(w * width_alignment)
                    h = int(h * height_alignment)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #horizontal display
            
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('frame', display)
            
            if key == ord('a') and not ready:
                ready = True
                monitoring.measure_start()
            #click B to exit
            elif key == ord('b'):
                mem_avg, cpu_avg = monitoring.measure_stop()
                break

    cap.release()
    cv2.destroyAllWindows()