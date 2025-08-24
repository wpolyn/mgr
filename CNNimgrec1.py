import cv2

from imports.CNNmodels import initialize_detector
from imports.inits3 import initialize_camera, text_properties
from imports.CNNinits_registry import load_registry
from imports.performance2 import performance
from imports.measure_history2 import append_mem_history, append_cpu_history, append_duration_history

print("This implementation is adjusted for a 320x320 detector input resolution, please make the appropriate selection.")
display_size = (320, 240)
crop_size = 320

if __name__ == "__main__":
    cap, img_src = initialize_camera()
    org, fontFace, fontScale, color, thickness = text_properties()
    detector, detector_name, input_size = initialize_detector()
    label_to_id = load_registry()
    monitoring = performance()

    ready = False
    end = False
    sampleNum = 0

    #get input from user while preventing duplicate entries
    while True:
        label = input('Please enter your name:\n')
        if label not in label_to_id:
            break
        else:
            print(f'{label} is already registered. Please modify your input.')
    
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
            #prepare for sampling
            if ready == False: 
                cv2.putText(display, 'Click A to sample.', org, fontFace, fontScale, color, thickness)
            #sampling loop
            else:
                sampleNum = sampleNum + 1
                fileName=f"{img_src}/imagesCNN/{label}_{sampleNum:02d}.jpg"
                if sampleNum <= 50:
                    cv2.putText(display, 'Sampling in progress.', org, fontFace, fontScale, color, thickness)
                    cv2.imwrite(fileName, gray[y:y + h, x:x + w])
                else:
                    cv2.putText(display, 'Done, click B to quit.', org, fontFace, fontScale, color, thickness)
                    if end == False:
                        mem_avg, cpu_avg = monitoring.measure_stop()
                        duration = monitoring.duration_stop()
                        end = True
            
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

            #disable for vertical
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('frame', display)

            #click A to sample
            if key == ord('a') and not ready:
                monitoring.measure_start()
                monitoring.duration_start()
                ready = True
            #click B to exit
            elif key == ord('b'):
                append_mem_history(detector_name, mem_avg)
                append_cpu_history(detector_name, cpu_avg)
                append_duration_history(detector_name, duration)
                break

    cap.release()
    cv2.destroyAllWindows()