import cv2

from inits2 import initialize_camera, text_properties
input_size = (320, 320)
display_size = (320, 240)
crop_size = 320
detector = cv2.FaceDetectorYN.create("models/yunet/yunet_n_320_320.onnx", "", input_size)
recognizer = cv2.FaceRecognizerSF.create("models/sface/face_recognition_sface_2021dec.onnx", "")
known_faces = []
reference_face = None
threshold = 0.363
if __name__ == '__main__':
    cap, img_src = initialize_camera()
    org, fontFace, fontScale, color, thickness = text_properties()
    ready = False

    #frame skipping
    frame_buffer = 2
    frame_counter = 0
    previous_face = []

    while (cap.isOpened()):
        ret, image = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if ret:
            if frame_counter % frame_buffer == 0:
                #model input processing
                h, w = image.shape[:2]
                x_input = (w - crop_size) // 2
                y_input = (h - crop_size) // 2
                image = image[y_input:y_input+crop_size, x_input:x_input+crop_size]
                image = cv2.resize(image, input_size)
                retval, faces = detector.detect(image)

                #
                previous_face = []
                if faces is not None:
                    for face in faces:
                        x, y, w, h = map(int, face[:4])
                        aligned_face = recognizer.alignCrop(image, face)
                        features_face = recognizer.feature(aligned_face)
                        #
                        if ready and reference_face is None:
                            reference_face = features_face
                        if reference_face is not None:
                            recognition = recognizer.match(features_face, reference_face, cv2.FaceRecognizerSF_FR_COSINE)
                            previous_face.append((face, recognition))
                frame_counter += 1
            else:
                frame_counter += 1
            #display processing
           
            #naprawic display rozdzielczosc tutaj teraz
        
            x_display = 0        
            y_display = (display_size[0] - display_size[1]) // 2
            display = image[y_display:y_display+display_size[1],x_display:display_size[0]]
            display = cv2.flip(display,1)
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for face, recognition in previous_face:
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if recognition is not None:
                    if recognition > threshold:
                        label = f"Wojtas: {recognition:.2f}"
                    else:
                        label = f"Nie-Wojtas: {recognition:.2f}"
                    cv2.putText(display, label, (x, y -10), fontFace, fontScale, color, thickness)
            cv2.imshow('frame', display)
            
            if key == ord('a') and not ready:
                ready = True
            #click B to exit
            elif key == ord('b'):
                break

    cap.release()
    cv2.destroyAllWindows()