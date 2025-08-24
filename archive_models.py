import cv2

#machine learning models
def initialize_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
    detector_model = "haarcascades"
    #detector = cv2.CascadeClassifier(cv2.data.lbpcascades + 'lbpcascade_frontalface.xml')
    #detector_model = "lbpcascades"
    return detector, detector_model

def initialize_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer_model = "LBPH"
    #
    #
    return recognizer, recognizer_model