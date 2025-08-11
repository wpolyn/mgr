import cv2

def initialize_detector():
    input_320 = (320, 320)
    input_640 = (640, 640)
    
    detector_catalogue = {
        "1" : "Standard YuNet model (640x640 input)",
        "2" : "int8 quantized YuNet model (640x640 input)",
        "3" : "int8 block-quantized YuNet model (640x640 input)",
        "4" : "Normal-sized YuNet model (320x320 input)",
        "5" : "Small-sized YuNet model (320x320 input)"
        #,"6" : "Normal-sized dynamic YuNet model",
        #"7" : "Small-sized dynamic YuNet model"
    }

    for key in detector_catalogue:
        print(f"{key} : {detector_catalogue[key]}")

    choice = input(f"Please choose the detector model by selecting it's corresponding number:\n")

    if choice == 1:
        model_path = "models/yunet/face_detection_yunet_2023mar.onnx"
        model_name = "YuNet_640x640"
        input_size = input_640
    elif choice == 2:
        model_path = "models/yunet/face_detection_yunet_2023mar_int8.onnx"
        model_name = "YuNet_int8_640x640"
        input_size = input_640
    elif choice == 3:
        model_path = "models/yunet/face_detection_yunet_2023mar_int8bq.onnx"
        model_name = "YuNet_int8bq_640x640"
        input_size = input_640
    elif choice == 4:
        model_path = "models/yunet/yunet_n_320_320.onnx"
        model_name = "YuNet_n_320x320"
        input_size = input_320
    elif choice == 5:
        model_path = "models/yunet/yunet_s_320_320.onnx"
        model_name = "YuNet_s_320x320"
        input_size = input_320
    #elif choice == 6:
        #model_path = "models/yunet/yunet_n_dynamic.onnx"
    #elif choice == 7:
        #model_path = "models/yunet/yunet_s_dynamic.onnx"
    else:
        print(f"Your input {choice} does not correspond to any model. Please modify your input.")
    
    detector = cv2.FaceDetectorYN.create({model_path}, "", {input_size})

    return detector, model_name, input_size

def initialize_recognizer():

    recognizer_catalogue = {
        "1" : "Standard SFace model",
        "2" : "int8 quantized SFace model",
        "3" : "int8 block-quantized SFace model"
    }

    choice = input(f"Please choose the recognizer model by selecting it's corresponding number:\n{recognizer_catalogue}")

    if choice == 1:
        model_path = "models/sface/face_recognition_sface_2021dec.onnx"
        model_name = "SFace"
    elif choice == 2:
        model_path = "models/sface/face_recognition_sface_2021dec_int8.onnx"
        model_name = "SFace_int8"
    elif choice == 3:
        model_path = "models/sface/face_recognition_sface_2021dec_int8bq.onnx"
        model_name = "SFace_int8bq"
    else:
        print(f"Your input {choice} does not correspond to any recognizer model. Please modify your input.")

    recognizer = cv2.FaceRecognizerSF.create({model_path}, "")

    return recognizer, model_name