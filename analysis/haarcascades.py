import os
import cv2
import time

from imports.MLmodels import initialize_detector

detector, detector_model = initialize_detector()
main_dir = "TEST_val"
output_dir = "TEST_val_results"

if __name__ == '__main__':
    for sub_dir, dirs, filenames in os.walk(main_dir, topdown=True):
        sub_path = os.path.relpath(sub_dir, main_dir)
        for filename in filenames:
            txtname = (os.path.splitext(filename)[0]+".txt") #3_Riot_Riot_3_604.txt
            dirfile = (os.path.join(sub_path, filename)) #3--Riot/3_Riot_Riot_3_958.jpg
            fullfile = (os.path.join(sub_dir, filename)) #TEST_val/1--Handshaking/1_Handshaking_Handshaking_1_134.jpg
            output = []
            output_subdirs = (os.path.join(output_dir, sub_path)) #TEST_val_results/1--Handshaking
            os.makedirs(output_subdirs, exist_ok = True)
            output_path = os.path.join(output_subdirs, txtname) #TEST_val_results/1--Handshaking/1_Handshaking_Handshaking_1_134.txt

            output.append(dirfile)

            image = cv2.imread(fullfile)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #PYTANIE: Czy czas mierzony powinien być łącznie z koniecznym/opcjonalnym procesowaniem obrazu?
            start_timestamp = time.time()
            faces = detector.detectMultiScale(gray, 1.3, 5)
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

            output.append(str(detected_faces))

            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    score = 1.0
                    face_report = (f"{x} {y} {w} {h} {score}")
                    output.append(face_report)

            with open (output_path, "w") as f:
                f.write("\n".join(output))
            
"""
docelowy output przyklad
0--Parade/0_Parade_marchingband_1_869.jpg
5
70 194 61 80 
306 212 53 66 
520 205 53 69 
704 324 45 60 
883 332 46 59 
"""