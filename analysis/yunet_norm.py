import os
import cv2
import time
import pickle

from imports.CNNmodels import initialize_detector

detector, detector_model, input_size = initialize_detector()
confidence_threshold = 0.02
detector.setScoreThreshold(confidence_threshold)
print(detector.getScoreThreshold())
main_dir = "WIDER_val"
output_dir = "WIDER_val_results"

if __name__ == '__main__':
    overall_output = []
    overall_faces = 0
    overall_duration = 0
    overall_files = 0

    for sub_dir, dirs, filenames in os.walk(main_dir, topdown=True):
        sub_path = os.path.relpath(sub_dir, main_dir)
        for filename in filenames:
            print(filename)
            txtname = (os.path.splitext(filename)[0]+".txt") #3_Riot_Riot_3_604.txt
            txtcontent = (os.path.splitext(filename)[0]) #3_Riot_Riot_3_604
            #dirfile = (os.path.join(sub_path, filename)) #3--Riot/3_Riot_Riot_3_958.jpg
            fullfile = (os.path.join(sub_dir, filename)) #TEST_val/1--Handshaking/1_Handshaking_Handshaking_1_134.jpg
            output = []
            output_subdirs = (os.path.join(output_dir, sub_path)) #TEST_val_results/1--Handshaking
            os.makedirs(output_subdirs, exist_ok = True)
            output_path = os.path.join(output_subdirs, txtname) #TEST_val_results/1--Handshaking/1_Handshaking_Handshaking_1_134.txt

            output.append(txtcontent)
            image = cv2.imread(fullfile)
            start_timestamp = time.time()
            b, g, r = cv2.split(image)
            b_norm = cv2.normalize(b, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
            g_norm = cv2.normalize(g, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
            r_norm = cv2.normalize(r, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
            image_norm = cv2.merge([b_norm, g_norm, r_norm])
            h_image, w_image = image_norm.shape[:2]
            x_alignment = (w_image / input_size[0])
            y_alignment = (h_image / input_size[1])
            image_input = cv2.resize(image_norm, input_size)
            retval, faces = detector.detect(image_input)
            end_timestamp = time.time()
            duration = end_timestamp - start_timestamp

            if faces is not None and len(faces) > 0:
                detected_faces = len(faces)
                latency = duration / detected_faces
            else:
                detected_faces = 0
                latency = duration

            latency = int(latency * 1000)
            print(f"{latency} ms")
            overall_faces += detected_faces
            overall_duration += duration
            overall_files += 1
            output.append(str(detected_faces))

            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    score = float(face[-1])
                    x = int(x * x_alignment)
                    y = int(y * y_alignment)
                    w = int(w * x_alignment)
                    h = int(h * y_alignment)
                    face_report = (f"{x} {y} {w} {h} {score:.4f}")
                    output.append(face_report)

            with open (output_path, "w") as f:
                f.write("\n".join(output))
    overall_output.append(f"main_dir : {main_dir}")
    overall_output.append(f"detector_model : {detector_model}")
    overall_output.append(f"confidence_threshold : {confidence_threshold}")
    overall_output.append(f"overall_files : {overall_files}")
    overall_output.append(f"overall_faces : {overall_faces}")
    overall_output.append(f"overall_duration : {overall_duration:.3f} s")
    if overall_faces > 0:
        latency_per_face = int((overall_duration / overall_faces) * 1000)
    if overall_files > 0:
        latency_per_file = int((overall_duration / overall_files) * 1000)
    overall_output.append(f"latency_per_face : {latency_per_face} ms")
    overall_output.append(f"latency_per_file : {latency_per_file} ms")
    print(overall_output)
    with open('overall_output.pkl', 'wb') as f:
        pickle.dump(overall_output, f)