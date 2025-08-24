import cv2 
def get_landmarks():
    face_detector = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml")
    nose_detector = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_mcs_nose.xml")
    smile_detector = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_smile.xml")
    return face_detector, eye_detector, nose_detector, smile_detector