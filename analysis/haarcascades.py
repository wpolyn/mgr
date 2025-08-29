import cv2
import time

from imports.MLmodels import initialize_detector

detector, detector_model = initialize_detector()
face_counter = 0
main_report = []
if __name__ == '__main__':
    image = cv2.imread("testimage.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = open("testimage.txt", "a")
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for face in faces:
            face_timestamp = time.time()
            face_counter +=1
            x, y, w, h = map(int, face[:4])
            face_report = f"{x} {y} {w} {h}"
            with open ("testimage.txt", "a") as f:
                f.write(f"{face_report}\n")
        with open ("testimage.txt", "a") as f:
            f.write(f"{face_counter}")



"""
0--Parade/0_Parade_marchingband_1_869.jpg
5
70 194 61 80 0 0 0 0 0 0 
306 212 53 66 0 0 0 0 0 1 
520 205 53 69 0 0 0 0 0 0 
704 324 45 60 0 0 0 0 0 0 
883 332 46 59 1 0 0 0 2 0 


69 190 74 74 + 
706 325 52 52 + 
2
"""
