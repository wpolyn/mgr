import cv2

def initialize_detector():
    input_320 = (320, 320)
    input_640 = (640, 640)
    input_dynamic = (0, 0)

    detector_catalogue = {
        "1" : "Normal-sized YuNet model (320x320 input)",
        "2" : "Small-sized YuNet model (320x320 input)",
        "3" : "Standard YuNet model (640x640 input)",
        "4" : "int8 quantized YuNet model (640x640 input)",
        "5" : "int8 block-quantized YuNet model (640x640 input)",
        "6" : "Normal-sized YuNet model (dynamic input)",
        "7" : "Small-sized YuNet model (dynamic input)"
    }

    for key in detector_catalogue:
        print(f"{key} : {detector_catalogue[key]}")

    while True:
        try:
            choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
            if choice == 1:
                model_path = "models/yunet/yunet_n_320_320.onnx"
                detector_name = "YuNet_n_320x320"
                input_size = input_320
            elif choice == 2:
                model_path = "models/yunet/yunet_s_320_320.onnx"
                detector_name = "YuNet_s_320x320"
                input_size = input_320
            elif choice == 3:
                model_path = "models/yunet/face_detection_yunet_2023mar.onnx"
                detector_name = "YuNet_640x640"
                input_size = input_640
            elif choice == 4:
                model_path = "models/yunet/face_detection_yunet_2023mar_int8.onnx"
                detector_name = "YuNet_int8_640x640"
                input_size = input_640
            elif choice == 5:
                model_path = "models/yunet/face_detection_yunet_2023mar_int8bq.onnx"
                detector_name = "YuNet_int8bq_640x640"
                input_size = input_640
            elif choice == 6:
                model_path = "models/yunet/yunet_n_dynamic.onnx"
                input_size = input_dynamic
            elif choice == 7:
                model_path = "models/yunet/yunet_s_dynamic.onnx"
                input_size = input_dynamic
            else:
                print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                continue
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    
    detector = cv2.FaceDetectorYN.create(model_path, "", input_size)

    return detector, detector_name, input_size

def initialize_recognizer():

    recognizer_catalogue = {
        "1" : "Standard SFace model",
        "2" : "int8 quantized SFace model",
        "3" : "int8 block-quantized SFace model"
    }

    for key in recognizer_catalogue:
        print(f"{key} : {recognizer_catalogue[key]}")

    while True:
        try:
            choice = int(input(f"Please choose the recognizer model by selecting its corresponding number:\n"))
            if choice == 1:
                model_path = "models/sface/face_recognition_sface_2021dec.onnx"
                recognizer_name = "SFace"
            elif choice == 2:
                model_path = "models/sface/face_recognition_sface_2021dec_int8.onnx"
                recognizer_name = "SFace_int8"
            elif choice == 3:
                model_path = "models/sface/face_recognition_sface_2021dec_int8bq.onnx"
                recognizer_name = "SFace_int8bq"
            else:
                print(f"Your input {choice} does not correspond to any recognizer model. Please modify your input.")
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    
    recognizer = cv2.FaceRecognizerSF.create(model_path, "")

    return recognizer, recognizer_name