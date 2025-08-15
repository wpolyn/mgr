import cv2

from inits2 import initialize_camera, text_properties
from performance2 import performance
from CNNmodels import initialize_detector, initialize_recognizer
detector, detector_name, input_size = initialize_detector()
recognizer, recognizer_name = initialize_recognizer()
monitoring = performance()

display_size = (320, 240)
crop_size = 320

#known_faces = []
reference_face = None
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
        image = cv2.flip(image,1)
        key = cv2.waitKey(1) & 0xFF
        if ret:
            #detection and recognition inside frame skipping if ready
            if ready and frame_counter % frame_buffer == 0:
                #model input processing and detection
                h_image, w_image = image.shape[:2] #h=480 w=640
                x_input = (w_image - crop_size) // 2 #160
                y_input = (h_image - crop_size) // 2 #80
                feed = image[y_input:y_input+crop_size, x_input:x_input+crop_size] #y: 80<->400=320 x: 160<->480=320
                feed = cv2.resize(feed, input_size)
                retval, faces = detector.detect(feed)
                #recognition
                previous_faces = []
                if faces is not None:
                    for face in faces:
                        x, y, w, h = map(int, face[:4])
                        aligned_face = recognizer.alignCrop(feed, face)
                        features_face = recognizer.feature(aligned_face)
                        #
                        if ready and reference_face is None:
                            reference_face = features_face
                        if reference_face is not None:
                            recognition = recognizer.match(features_face, reference_face, cv2.FaceRecognizerSF_FR_COSINE)
                            previous_faces.append((face, recognition))
                frame_counter += 1
            else:
                frame_counter += 1

            #display processing
            display = cv2.resize(image, display_size)

            #displaying results if ready
            if ready and previous_faces is not None:
                for face, recognition in previous_faces:
                    x, y, w, h = map(int, face[:4])
                    #align coordinates
                    #crop alignment
                    x = x + x_input
                    y = y + y_input
                    #resize alignment
                    width_alignment = (display_size[0]/w_image) #0.5          
                    height_alignment = (display_size[1]/h_image) #0.5
                    x = int(x * width_alignment)
                    y = int(y * height_alignment)
                    w = int(w * width_alignment)
                    h = int(h * height_alignment)
                
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    if recognition is not None:
                        if recognition > threshold:
                            label = f"Wojtas: {recognition:.2f}"
                        else:
                            label = f"Nie-Wojtas: {recognition:.2f}"
                        cv2.putText(display, label, (x, y -10), fontFace, fontScale, color, thickness)
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