import cv2

#machine learning models
def initialize_detector():
    detector_catalogue = {
        "1" : "Haar cascades",
        "2" : "LBP (Local Binary Patterns) cascades"
        }

    for key in detector_catalogue:
        print(f"{key} : {detector_catalogue[key]}")

    while True:
        try:
            choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
            if choice == 1:
                detector_name = "HAARrcascades"
                detector_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            
            elif choice == 2:
                detector_name = "LBPcascades"
                detector_path = "/usr/local/share/opencv4/lbpcascades/lbpcascade_frontalface.xml"

            else:
                print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                continue
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    detector = cv2.CascadeClassifier(detector_path)
    return detector, detector_name

def initialize_recognizer():
    recognizer_catalogue = {
        "1" : "LBPH",
        "2" : "Eigen",
        "3" : "Fisher"
    }

    for key in recognizer_catalogue:
        print(f"{key} : {recognizer_catalogue[key]}")

    while True:
        try:
            choice = int(input(f"Please choose the recognizer model by selecting its corresponding number:\n"))
            if choice == 1:
                recognizer_name = "LBPH (Local Binary Patterns Histograms)"
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif choice == 2:
                recognizer_name = "EigenFace"
                recognizer = cv2.face.EigenFaceRecognizer_create()
            elif choice == 3:
                recognizer_name = "FisherFace"
                recognizer = cv2.face.FisherFaceRecognizer_create()
                
            else:
                print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                continue
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    return recognizer, recognizer_name