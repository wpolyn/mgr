import os
import cv2
import time

from imports.MLmodels import initialize_detector

detector, detector_model = initialize_detector()
main_dir = "WIDER_val"

if __name__ == '__main__':
    image = cv2.imread("testimage.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_timestamp = time.time()
    faces = detector.detectMultiScale(gray, 1.3, 5)
    end_timestamp = time.time()
    duration = end_timestamp - start_timestamp
    if faces is not None:
        detected_faces = len(faces)
        latency = duration / detected_faces
    else:
        detected_faces = 0
        latency = duration
    latency = int(latency * 1000)
    print(f"{latency} ms")
    with open ("testimage.txt", "a") as f:
            f.write(f"{detected_faces}\n")
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            score = 1.0
            face_report = (f"{x} {y} {w} {h} {score}")
            with open ("testimage.txt", "a") as f:
                f.write(f"{face_report}\n")

for sub_dir, dirs, filenames in os.walk(main_dir, topdown=True):
     sub_path = os.path.relpath(sub_dir, main_dir)
     for filename in filenames:
          plep = (os.path.join(sub_path, filename))
          print(plep)
          print(os.path.splitext(plep)[0]+".txt")
     
"""
0--Parade/0_Parade_marchingband_1_869.jpg
5
70 194 61 80 
306 212 53 66 
520 205 53 69 
704 324 45 60 
883 332 46 59 
"""
