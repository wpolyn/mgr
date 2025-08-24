import cv2

from imports.inits3 import initialize_camera, text_properties
from imports.CNNinits_registry import load_registry, invert_registry
from imports.MLmodels import initialize_recognizer
from imports.CNNmodels import initialize_detector

print("This implementation is adjusted for a 320x320 detector input resolution, please make the appropriate selection.")
display_size = (320, 240)
crop_size = 320

if __name__ == "__main__":
    cap, img_src = initialize_camera()
    org, fontFace, fontScale, color, thickness = text_properties()
    detector, detector_name, input_size = initialize_detector()
    label_to_id = load_registry()
    
    recognizer, recognizer_name = initialize_recognizer()
    id_to_label = invert_registry(label_to_id)

    recognizer.read(img_src+'/CNNmodel.yml')

    while (cap.isOpened()):
        ret, image = cap.read()
        key = cv2.waitKey(1) & 0xFF
        image = cv2.flip(image, 1)
        display = cv2.resize(image, display_size)
        h_image, w_image = image.shape[:2] #h=480 w=640
        x_input = (w_image - crop_size) // 2 #160
        y_input = (h_image - crop_size) // 2 #80
        if ret:
            feed = image[y_input:y_input + crop_size, x_input:x_input + crop_size]
            feed = cv2.resize(feed, input_size)
            gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
            retval, faces = detector.detect(feed)
            if faces is None:
                faces = []
            #detect facial features
            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    img_id, confidence = recognizer.predict(gray)

            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    #align coordinates
                    #crop alignment
                    x = x + x_input
                    y = y + y_input
                    #resize alignment
                    width_alignment = (display_size[0] / w_image) #0.5
                    height_alignment = (display_size[1] / h_image) #0.5
                    x = int(x * width_alignment)
                    y = int(y * height_alignment)
                    w = int(w * width_alignment)
                    h = int(h * height_alignment)                
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
                    if confidence < 70:
                        cv2.putText(display, f"{id_to_label[img_id]} : {confidence:.2f}", (x, y -10), fontFace, fontScale, color, thickness)
                    else:
                        img_id = "unknown" 
                        cv2.putText(display, str(img_id), (x, y -10), fontFace, fontScale, color, thickness)

            #disable for vertical
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)    
            h_display, w_display = display.shape[:2]
            print (h_display, w_display)
            cv2.imshow('frame', display)
            
            # Press button b to exit
            if key & 0xFF == ord('b'):  
                break

    cap.release()
    cv2.destroyAllWindows()