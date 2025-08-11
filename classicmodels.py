import cv2

#machine learning models
def initialize_detector():
    detector_catalogue = {
        "1" : "Haar cascades"
    }

    for key in detector_catalogue:
        print(f"{key} : {detector_catalogue[key]}")

    choice = input(f"Please choose the detector model by selecting it's corresponding number:\n")

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detector_model = "haarcascades"
    #detector = cv2.CascadeClassifier(cv2.data.lbpcascades + 'lbpcascade_frontalface.xml')
    #detector_model = "lbpcascades"
    return detector, detector_model

def initialize_recognizer():
    recognizer_catalogue = {
        "1" : "LBPH",
        "2" : "Eigen",
        "3" : "Fisher"
    }

    for key in recognizer_catalogue:
        print(f"{key} : {recognizer_catalogue[key]}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer_model = "LBPH"
    #recognizer = cv2.face.EigenFaceRecognizer_create()
    #recognizer_model = "Eigen"
    #recognizer = cv2.face.FisherFaceRecognizer_create()
    #recognizer_model = "Fisher"
    return recognizer, recognizer_model