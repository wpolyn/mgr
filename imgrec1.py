import cv2

from imports.MLmodels import initialize_detector
from imports.inits3 import initialize_camera, text_properties, horizontal_image_processing, vertical_image_processing
from imports.inits_registry import load_registry
from imports.performance2 import performance
from imports.measure_history2 import append_mem_history, append_cpu_history, append_duration_history

if __name__ == '__main__':
    cap, img_src = initialize_camera()
    org, fontFace, fontScale, color, thickness = text_properties()
    detector, detector_model = initialize_detector()
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
        if ret:
            #image, gray, faces = vertical_image_processing(image, detector)
            image, gray, faces = horizontal_image_processing(image, detector)
            #detect facial features
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            #prepare for sampling
            if ready == False: 
                cv2.putText(image, 'Click A to sample.', org, fontFace, fontScale, color, thickness)
            #sampling loop
            else:
                sampleNum = sampleNum + 1
                fileName=f"{img_src}/images/{label}_{sampleNum:02d}.jpg"
                if sampleNum <= 50:
                    cv2.putText(image, 'Sampling in progress.', org, fontFace, fontScale, color, thickness)
                    cv2.imwrite(fileName, gray[y:y + h, x:x + w])
                else:
                    cv2.putText(image, 'Done, click B to quit.', org, fontFace, fontScale, color, thickness)
                    if end == False:
                        mem_avg, cpu_avg = monitoring.measure_stop()
                        duration = monitoring.duration_stop()
                        end = True
            #disable for vertical
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('frame', image)

            #click A to sample
            if key == ord('a') and not ready:
                monitoring.measure_start()
                monitoring.duration_start()
                ready = True
            #click B to exit
            elif key == ord('b'):
                append_mem_history(detector_model, mem_avg)
                append_cpu_history(detector_model, cpu_avg)
                append_duration_history(detector_model, duration)
                break
    cap.release()
    cv2.destroyAllWindows()